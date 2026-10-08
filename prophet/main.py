

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from prophet_pipeline import (
    load_region_csv,
    validate_and_clean,
    print_quality_report,
    split_train_val_test,
    train_prophet,
    forecast_horizon,
    plot_forecast_vs_actual,
    _HAS_REAL_PROPHET,
)


# =============================================================================
# НАСТРОЙКИ
# =============================================================================
CSV_FILE   = "electricity_75.csv"      
CACHE_FILE = "electricity_clean.pkl"   
REGION     = "Челябинская область"
TARGET_COL = "actual_consumption"
OUTPUT_DIR = "output"

TEST_DAYS  = 30   
HIT_RATE_K = 3    


# =============================================================================
# 1. ЗАГРУЗКА + ОЧИСТКА (обработка ровно ОДИН раз, кэш в pickle)
# =============================================================================
def load_and_clean_once(csv_path: str, cache_path: str) -> pd.DataFrame:
    """
    Если рядом уже есть .pkl-кэш очищенного датасета — читаем его.
    Иначе читаем CSV, чистим, сохраняем кэш.
    """
    if os.path.exists(cache_path):
        print(f"[CACHE] Найден кэш '{cache_path}', читаем очищенный датасет...")
        df = pd.read_pickle(cache_path)
        print(f"[CACHE] Загружено {len(df)} строк, "
              f"{df.index.min()} .. {df.index.max()}")
        return df

    print(f"[LOAD] Кэша нет. Читаем сырой CSV: {csv_path}")
    raw = load_region_csv(csv_path)

    print("[CLEAN] Валидация и очистка (это делается ОДИН раз)...")
    clean, report = validate_and_clean(raw, target_col=TARGET_COL)
    print_quality_report(report, REGION)

    print(f"[SAVE] Сохраняем очищенный датасет в кэш: {cache_path}")
    clean.to_pickle(cache_path)
    return clean


# =============================================================================
# 2. МЕТРИКИ В ПРОЦЕНТАХ
# =============================================================================
def metrics_percent(y_true, y_pred, lower, upper):
    """
    Возвращает словарь только с процентными метриками — понятными заказчику.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    lower  = np.asarray(lower,  dtype=float)
    upper  = np.asarray(upper,  dtype=float)

    mape  = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    smape = np.mean(2 * np.abs(y_pred - y_true) /
                    (np.abs(y_true) + np.abs(y_pred))) * 100
    coverage = np.mean((y_true >= lower) & (y_true <= upper)) * 100

    return {
        "Средняя ошибка прогноза (MAPE), %":        round(mape, 2),
        "Симметричная ошибка (sMAPE), %":           round(smape, 2),
        "Покрытие 80% доверительного интервала, %": round(coverage, 2),
    }


# =============================================================================
# 3. TOP-3 HIT RATE (в процентах, по дням)
# =============================================================================
def top3_hit_rate_percent(test_df, forecast_df, target_col=TARGET_COL, k=HIT_RATE_K):
    """
    Для каждого дня test-периода считает:
      сколько из k предсказанных пиковых часов попало в k фактических.
    Возвращает:
      overall   — общий процент попаданий (по всем дням),
      per_day   — DataFrame по дням (date, hits, k, hit_rate_%).
    """
    merged = forecast_df.join(test_df[[target_col]], how="inner")
    merged["date"] = merged.index.date

    rows = []
    for day, day_df in merged.groupby("date"):
        if len(day_df) < k:
            continue
        pred_top = set(day_df.sort_values("yhat_upper", ascending=False).head(k).index)
        actual_top = set(day_df.sort_values(target_col, ascending=False).head(k).index)
        hits = len(pred_top & actual_top)
        rows.append({
            "date": pd.Timestamp(day),
            "hits": hits,
            "k": k,
            "hit_rate_%": hits / k * 100,
        })
    per_day = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    overall = per_day["hits"].sum() / per_day["k"].sum() * 100 if len(per_day) else float("nan")
    return overall, per_day


# =============================================================================
# 4. ГЛАВНАЯ ФУНКЦИЯ
# =============================================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("ПРОГНОЗ ТОП-3 ПИКОВЫХ ЧАСОВ — РЕЗУЛЬТАТЫ")
    print(f"Движок: {'Prophet (настоящий)' if _HAS_REAL_PROPHET else '_FourierGAM (fallback)'}")
    print("=" * 70)

    # --- 1. Загрузка + очистка (ровно ОДИН раз) -------------------------------
    df = load_and_clean_once(CSV_FILE, CACHE_FILE)

    cols = [TARGET_COL] + (["average_price"] if "average_price" in df.columns else [])
    model_df = df[cols].copy().sort_index()

    # --- 2. Train / Test разбиение --------------------------------------------
    train_df, _val_df, test_df = split_train_val_test(
        model_df, val_days=30, test_days=TEST_DAYS
    )
    print(f"\n[SPLIT] train: {len(train_df)} ч | test: {len(test_df)} ч")
    print(f"[SPLIT] test:  {test_df.index.min()} .. {test_df.index.max()}")

    # --- 3. Обучение на train -------------------------------------------------
    print("\n[TRAIN] Обучаем модель...")
    model = train_prophet(train_df, target_col=TARGET_COL, price_regressor=True)

    # --- 4. Прогноз на весь test ---------------------------------------------
    print("[EVAL] Оцениваем на тестовой выборке...")
    price = test_df["average_price"] if "average_price" in test_df.columns else None
    forecast = forecast_horizon(model, test_df.index, price_series=price)

    # --- 5. Числовые метрики в процентах -------------------------------------
    y_true = test_df[TARGET_COL]
    pct = metrics_percent(
        y_true, forecast["yhat"], forecast["yhat_lower"], forecast["yhat_upper"]
    )
    overall_hit, per_day = top3_hit_rate_percent(
        test_df, forecast, target_col=TARGET_COL, k=HIT_RATE_K
    )

    print("\n" + "=" * 70)
    print("ЧИСЛОВЫЕ РЕЗУЛЬТАТЫ (в процентах)")
    print("=" * 70)
    for name, value in pct.items():
        print(f"  {name:<45} {value:>8.2f} %")
    print(f"  {'Попадание в Top-3 (за весь тест)':<45} {overall_hit:>8.2f} %")
    if len(per_day):
        print(f"  {'Дней, где совпали все Top-3':<45} "
              f"{(per_day['hits'] == HIT_RATE_K).mean() * 100:>8.2f} %")
        print(f"  {'Дней, где не совпал ни один Top-3':<45} "
              f"{(per_day['hits'] == 0).mean() * 100:>8.2f} %")

    # --- 6. Единственный график: Прогноз vs факт + Top-3 + остатки -----------
    plot_forecast_vs_actual(
        test_df, forecast,
        target_col=TARGET_COL,
        title=f"Прогноз vs факт — {REGION} (test)",
        save_path=os.path.join(OUTPUT_DIR, "forecast_vs_actual.png"),
    )
    print(f"\n[DONE] Результаты в папке: {os.path.abspath(OUTPUT_DIR)}/")


# =============================================================================
if __name__ == "__main__":
    main()
