"""
Прогноз топ-3 пиковых часов на сутки вперёд — Prophet, регион 75 (Челябинская область).

Часть 2: переобучение каждый день (расширяющееся окно).
Для каждого дня D модель обучается на всей истории строго до 0:00 дня D
и прогнозирует сутки D. Так система работала бы в реальной эксплуатации.

Запуск:  python main.py
Результаты — в папке output/.
"""

import os

import pandas as pd

from prophet_pipeline import (
    _HAS_REAL_PROPHET,
    expanding_window_daily_backtest,
    format_backtest_table,
    load_clean_cached,
    load_hourly_forecast,
    monthly_rollup,
    plot_forecast_vs_actual,
    plot_year_overview,
    top3_hit_rate_by_day,
)

# =============================================================================
# НАСТРОЙКИ
# =============================================================================
CSV_FILE = "electricity_75.csv"
CACHE_FILE = "electricity_clean.pkl"
ENCODING = None                      # None = авто (utf-8-sig, затем cp1251)
REGION = "Челябинская область (75)"
TARGET_COL = "actual_consumption"
OUTPUT_DIR = "output"

# Период бэктеста (включительно). None = последние BACKTEST_DAYS дней данных.
BACKTEST_START = "2025-06-01"
BACKTEST_END = "2025-06-30"
BACKTEST_DAYS = 30

N_JOBS = 1          # -1 = все ядра. Фолды независимы, на настоящем Prophet ускорение ~ в число ядер
RESUME = False      # True = продолжить прерванный прогон из output/expanding_checkpoint.csv
HIT_RATE_K = 3

CHECKPOINT_CSV = os.path.join(OUTPUT_DIR, "expanding_checkpoint.csv")
HOURLY_CSV = os.path.join(OUTPUT_DIR, "expanding_hourly.csv")


def pick_dates(df: pd.DataFrame) -> pd.DatetimeIndex:
    """Дни для прогноза: из настроек или последние BACKTEST_DAYS полных суток."""
    last_full_day = df.index.max().normalize()
    if df.index.max().hour != 23:
        last_full_day -= pd.Timedelta(days=1)
    if BACKTEST_START and BACKTEST_END:
        dates = pd.date_range(BACKTEST_START, BACKTEST_END, freq="D")
        if dates.min() < df.index.min().normalize() + pd.Timedelta(days=180):
            raise ValueError("Перед первым днём бэктеста должно быть хотя бы ~180 дней истории.")
        if dates.max() > last_full_day:
            raise ValueError(f"В данных нет фактов до {BACKTEST_END}: последние полные сутки — {last_full_day.date()}.")
        return dates
    return pd.date_range(end=last_full_day, periods=BACKTEST_DAYS, freq="D")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print(f"ПРОГНОЗ ТОП-3 ПИКОВЫХ ЧАСОВ — {REGION}")
    print(f"Движок: {'Prophet (настоящий)' if _HAS_REAL_PROPHET else '_FourierGAM (fallback, НЕ настоящий Prophet)'}")
    print("=" * 70)

    df = load_clean_cached(CSV_FILE, CACHE_FILE, target_col=TARGET_COL, region=REGION, encoding=ENCODING)
    cols = [TARGET_COL] + (["average_price"] if "average_price" in df.columns else [])
    model_df = df[cols].sort_index()

    dates = pick_dates(model_df)
    print(f"\n[BACKTEST] {len(dates)} дн.: {dates.min().date()} .. {dates.max().date()}, "
          f"переобучение каждый день, n_jobs={N_JOBS}")

    table = expanding_window_daily_backtest(
        model_df, dates, target_col=TARGET_COL,
        checkpoint_csv=CHECKPOINT_CSV, hourly_csv=HOURLY_CSV,
        n_jobs=N_JOBS, resume=RESUME,
    )
    if table.empty:
        raise SystemExit("Бэктест не дал ни одного дня — проверь период и данные.")

    # Файл с итогами пишем отдельно, чтобы не затирать чекпоинт строкой «СРЕДНЕЕ»
    table.to_csv(os.path.join(OUTPUT_DIR, "expanding_results.csv"), index=False)
    pretty = format_backtest_table(table)
    pretty.to_csv(os.path.join(OUTPUT_DIR, "expanding_results_pretty.csv"), index=False, encoding="utf-8-sig")

    # --- итоги ----------------------------------------------------------------
    mean = table[table["forecast_date"] == "СРЕДНЕЕ"].iloc[0]
    daily = table[table["forecast_date"] != "СРЕДНЕЕ"]
    print("\n" + "=" * 70)
    print(f"ИТОГИ ({len(daily)} дн.)")
    print("=" * 70)
    print(f"  MAE                      {mean['MAE']:>9.0f} МВт*ч")
    print(f"  RMSE                     {mean['RMSE']:>9.0f} МВт*ч")
    print(f"  MAPE                     {mean['MAPE_%']:>9.2f} %")
    print(f"  Попадание в топ-3        {mean['peak_hour_hit_rate_%']:>9.1f} %")
    print(f"  Покрытие 80%-интервала   {mean['interval_coverage_%']:>9.1f} %")

    # --- графики ----------------------------------------------------------------
    test_h, fc_h = load_hourly_forecast(HOURLY_CSV, dates.min(), dates.max(), TARGET_COL)
    plot_forecast_vs_actual(test_h, fc_h, target_col=TARGET_COL,
                            title=f"Прогноз vs факт — {REGION} (расширяющееся окно)",
                            save_path=os.path.join(OUTPUT_DIR, "forecast_vs_actual.png"))

    if len(daily) > 31:
        rollup = monthly_rollup(table)
        rollup.to_csv(os.path.join(OUTPUT_DIR, "expanding_monthly.csv"), index=False, encoding="utf-8-sig")
        plot_year_overview(table, rollup, period_label=REGION,
                           save_path=os.path.join(OUTPUT_DIR, "year_overview.png"))

    per_day = top3_hit_rate_by_day(test_h, fc_h, TARGET_COL, HIT_RATE_K)
    if len(per_day):
        print(f"  Дней, где совпали все топ-3:   {(per_day['hits'] == HIT_RATE_K).mean() * 100:>5.1f} %")
        print(f"  Дней, где не совпал ни один:   {(per_day['hits'] == 0).mean() * 100:>5.1f} %")

    print(f"\n[DONE] Результаты: {os.path.abspath(OUTPUT_DIR)}/")


if __name__ == "__main__":      # обязательно для n_jobs > 1 (особенно на Windows)
    main()
