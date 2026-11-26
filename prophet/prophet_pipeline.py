"""
Содержимое модуля:
  1. загрузка CSV региона и кэш очищенного датасета
  2. очистка и валидация почасового ряда
  3. разбиения train/val/test и rolling-origin фолды
  4. праздники РФ
  5. модель (Prophet либо _FourierGAM) и прогноз
  6. метрики точности и top-3 hit-rate
  7. бэктесты: rolling-origin и расширяющееся окно (переобучение каждый день)
  8. таблицы результатов
  9. графики
 10. единый протокол оценки Prophet: честный/текущий вариант, пространственный регрессор соседей,
     метрики «пикового режима», парная статистика (задачи 1-2)

"""

import glob
import itertools
import json
import logging
import math
import os
import re
import time
import warnings
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, effective_n_jobs
from scipy.stats import norm, wilcoxon
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")
# cmdstanpy на каждый fit пишет по несколько INFO-строк; на 365 переобучениях
# они забивают консоль и прячут строки прогресса.
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)
H1 = pd.Timedelta(hours=1)


# =============================================================================
# 1. ЗАГРУЗКА
# =============================================================================
_NUMERIC_COLS = ["planned_consumption", "actual_consumption",
                 "planned_generation", "actual_generation",
                 "average_price", "vsvgo_consumption", "vsvgo_avg_price"]

# число с пробелами-разделителями тысяч (обычный, неразрывный, узкий) и
# необязательной десятичной частью; обязательно начинается с цифры
_NUMBER_RE = re.compile(r"[-+]?\d[\d\s]*(?:[.,]\d+)?")


def _extract_number(value):
    """'4 051 МВт*ч' -> 4051.0 ; '979 руб./МВт*ч' -> 979.0 ; 4051 -> 4051.0 ; NaN -> nan"""
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    match = _NUMBER_RE.search(str(value))
    if not match:
        return np.nan
    return float(re.sub(r"\s", "", match.group()).replace(",", "."))


def load_region_csv(path: str, zip_path: str | None = None, encoding: str | None = None) -> pd.DataFrame:
    """
    Читает CSV региона в почасовой DataFrame с индексом datetime.

    encoding: явно указать кодировку. Обычно файлы utf-8-sig, но, например,
    electricity_71.csv (Тюмень) сохранён в cp1251 — если encoding не задан,
    пробуем оба варианта по очереди.
    Колонки с единицами измерения ('4 051 МВт*ч') превращаются в числа;
    уже числовые колонки (как в electricity_75.csv) только приводятся к float.
    Дубли по (дата, час) отбрасываются, остаётся первая запись.
    """
    def _read(enc):
        if zip_path:
            with zipfile.ZipFile(zip_path) as z:
                with z.open(path) as f:
                    return pd.read_csv(f, encoding=enc)
        return pd.read_csv(path, encoding=enc)

    if encoding:
        df = _read(encoding)
    else:
        try:
            df = _read("utf-8-sig")
        except UnicodeDecodeError:
            df = _read("cp1251")

    for col in _NUMERIC_COLS:
        if col in df.columns:
            s = df[col]
            df[col] = s.astype(float) if pd.api.types.is_numeric_dtype(s) else s.apply(_extract_number).astype(float)

    df["datetime"] = pd.to_datetime(df["date"]) + pd.to_timedelta(df["hour"], unit="h")
    df = df.set_index("datetime").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df


def load_clean_cached(csv_path: str, cache_path: str, target_col: str = "actual_consumption",
                      region: str = "", encoding: str | None = None) -> pd.DataFrame:
    """
    Загрузка + очистка РОВНО ОДИН раз: очищенный датасет кэшируется в pickle.
    Кэш считается устаревшим и пересобирается, если CSV новее кэша — иначе
    при замене CSV (например, на версию с данными за 2025 год) скрипт молча
    продолжал бы работать со старыми данными.
    """
    if os.path.exists(cache_path) and (
            not os.path.exists(csv_path) or os.path.getmtime(cache_path) >= os.path.getmtime(csv_path)):
        print(f"[CACHE] Найден кэш '{cache_path}', читаем очищенный датасет...")
        df = pd.read_pickle(cache_path)
        print(f"[CACHE] Загружено {len(df)} строк, {df.index.min()} .. {df.index.max()}")
        return df

    if os.path.exists(cache_path):
        print(f"[CACHE] '{cache_path}' старше '{csv_path}' — пересобираем.")
    print(f"[LOAD] Читаем сырой CSV: {csv_path}")
    raw = load_region_csv(csv_path, encoding=encoding)

    print("[CLEAN] Валидация и очистка (делается один раз)...")
    clean, report = validate_and_clean(raw, target_col=target_col)
    print_quality_report(report, region)

    print(f"[SAVE] Сохраняем очищенный датасет в кэш: {cache_path}")
    clean.to_pickle(cache_path)
    return clean


# =============================================================================
# 2. ПРЕДОБРАБОТКА И ВАЛИДАЦИЯ ДАННЫХ
# =============================================================================
def validate_and_clean(df: pd.DataFrame, target_col: str = "actual_consumption") -> tuple[pd.DataFrame, dict]:
    """
    Полный цикл очистки почасового ряда, как того требует ТЗ проекта:
      1) проверка временной последовательности и восстановление пропущенных часов
      2) проверка полноты суток (РФ не переходит на летнее время с 2014 г.,
         поэтому сутки не из 24 записей — это именно пропуски в данных)
      3) контроль корректности значений (потребление не может быть <= 0)
      4) обнаружение выбросов по профилю конкретного часа суток
      5) заполнение пропусков

    Возвращает (очищенный df, словарь с отчётом о найденных проблемах).
    """
    report = {}

    # --- 2.1 Полнота суток — считаем ДО достройки сетки, иначе проверка
    #         бессмысленна (после reindex в каждых сутках ровно 24 часа) ---
    rows_per_day = df.groupby(df.index.normalize()).size()
    report["incomplete_days"] = int((rows_per_day.iloc[1:-1] != 24).sum())  # крайние сутки могут быть неполными

    # --- 2.2 Временная последовательность: достраиваем ПОЛНУЮ почасовую сетку ---
    full_index = pd.date_range(df.index.min(), df.index.max(), freq="h")
    missing_hours = full_index.difference(df.index)
    report["missing_hours_count"] = len(missing_hours)
    report["missing_hours_pct"] = round(len(missing_hours) / len(full_index) * 100, 3)
    df = df.reindex(full_index)

    # --- 2.3 Контроль корректности значений: потребление должно быть > 0 ---
    invalid_mask = df[target_col] <= 0
    report["invalid_values_count"] = int(invalid_mask.sum())
    df.loc[invalid_mask, target_col] = np.nan

    # --- 2.4 Обнаружение выбросов ---
    # Сравниваем каждое значение с медианой и MAD ТОГО ЖЕ часа суток
    # (у потребления разный масштаб днём и ночью, обычный z-score по всему
    # ряду принял бы нормальный дневной пик за аномалию).
    hours = df.index.hour
    series = df[target_col]
    med = series.groupby(hours).transform("median")
    mad = (series - med).abs().groupby(hours).transform("median").replace(0, np.nan)
    robust_z = (series - med) / (1.4826 * mad)
    outlier_mask = robust_z.abs() > 5  # порог ~5 MAD — явные аномалии, не обычный разброс
    report["outliers_count"] = int(outlier_mask.sum())
    report["outliers_pct"] = round(outlier_mask.sum() / len(df) * 100, 3)
    df.loc[outlier_mask, target_col] = np.nan

    # --- 2.5 Заполнение пропусков (после reindex + invalid + outliers) ---
    report["total_nan_before_fill"] = int(df[target_col].isna().sum())
    # короткие провалы (<=3ч) — линейная интерполяция;
    # длинные провалы — тем же часом/днём недели неделю назад (сезонная заливка)
    df[target_col] = df[target_col].interpolate(limit=3, limit_direction="both")
    still_nan = df[target_col].isna()
    if still_nan.any():
        seasonal_fill = df[target_col].shift(24 * 7)
        df.loc[still_nan, target_col] = seasonal_fill[still_nan]
    df[target_col] = df[target_col].ffill().bfill()  # финальный фолбэк на крайних точках

    if "average_price" in df.columns:
        df["average_price"] = df["average_price"].interpolate(limit=6, limit_direction="both").ffill().bfill()

    report["rows_total"] = len(df)
    return df, report


def print_quality_report(report: dict, region: str):
    print(f"\n=== Отчёт о качестве данных: {region} ===")
    print(f"  Всего часов в периоде:        {report['rows_total']}")
    print(f"  Пропущенных часов (дыры):     {report['missing_hours_count']} ({report['missing_hours_pct']}%)")
    print(f"  Дней с неполными сутками:     {report['incomplete_days']} (≠24 записей; DST в РФ нет)")
    print(f"  Некорректных значений (<=0):  {report['invalid_values_count']}")
    print(f"  Выбросов (>5 MAD от медианы часа): {report['outliers_count']} ({report['outliers_pct']}%)")
    print(f"  Итого NaN перед финальной заливкой: {report['total_nan_before_fill']}")


# =============================================================================
# 3. РАЗБИЕНИЕ ДАННЫХ
# =============================================================================
def split_train_val_test(df: pd.DataFrame, val_days: int = 30, test_days: int = 30):
    """
    Для временных рядов НЕЛЬЗЯ бить случайно (sklearn train_test_split
    перемешает прошлое и будущее и даст завышенную точность). Делим строго
    по времени, в хронологическом порядке:

        [-----------  train  -----------][--- val ---][--- test ---]
        ранние даты                                         последние даты

    train — обучение модели,
    val   — подбор гиперпараметров (например changepoint_prior_scale
            у Prophet, или сила регуляризации у GAM-эквивалента),
    test  — финальная, ни разу не используемая до конца оценка точности.
    """
    test_cutoff = df.index.max() - pd.Timedelta(days=test_days)
    val_cutoff = test_cutoff - pd.Timedelta(days=val_days)

    train_df = df[df.index <= val_cutoff]
    val_df = df[(df.index > val_cutoff) & (df.index <= test_cutoff)]
    test_df = df[df.index > test_cutoff]
    return train_df, val_df, test_df


def rolling_origin_folds(df: pd.DataFrame, n_folds: int = 5,
                          horizon_days: int = 1, step_days: int = 14,
                          min_train_days: int = 180):
    """
    Walk-forward (rolling-origin) бэктест — надёжнее одного train/test
    разбиения: модель переобучается и проверяется несколько раз на разных
    "срезах" времени, что даёт среднюю точность и её разброс, а не
    случайное число по одному удачному/неудачному куску истории.

        Fold 1: train[........] -> test[день 1]
        Fold 2: train[...........] -> test[день 2]   (сдвиг на step_days)
        Fold 3: train[..............] -> test[день 3]
        ...

    Возвращает список (train_df, test_df) по каждому фолду.
    """
    folds = []
    last_day = df.index.max().normalize()
    first_possible_cutoff = df.index.min() + pd.Timedelta(days=min_train_days)

    cutoff = last_day - pd.Timedelta(days=(n_folds - 1) * step_days + horizon_days - 1)
    cutoff = max(cutoff, first_possible_cutoff)

    for i in range(n_folds):
        train_end = cutoff + pd.Timedelta(days=i * step_days) - pd.Timedelta(hours=1)
        test_start = train_end + pd.Timedelta(hours=1)
        test_end = test_start + pd.Timedelta(days=horizon_days) - pd.Timedelta(hours=1)
        if test_end > df.index.max():
            break
        train_df = df[df.index <= train_end]
        test_df = df[(df.index >= test_start) & (df.index <= test_end)]
        if len(train_df) < 24 * min_train_days * 0.5 or test_df.empty:
            continue
        folds.append((train_df, test_df))
    return folds


# =============================================================================
# 4. ПРАЗДНИКИ РФ (заданы вручную, без пакета `holidays`)
# =============================================================================
def russian_holidays(years) -> pd.DataFrame:
    """
    Нерабочие праздничные дни РФ. Фиксированные даты одинаковы из года
    в год; новогодние "длинные выходные" (2-8 января) фактически каждый
    год сдвигаются постановлением правительства на день-два, но ядро
    (1,2,3,7 января) неизменно — для прогноза потребления этого достаточно,
    точный перенос выходных значим только для ±1 дня около праздника.
    """
    fixed_md = ["01-01", "01-02", "01-03", "01-07",  # новогодние каникулы (ядро)
                "02-23", "03-08", "05-01", "05-09", "06-12", "11-04"]
    dates = [f"{y}-{md}" for y in years for md in fixed_md]
    return pd.DataFrame({"holiday": "ru_holiday", "ds": pd.to_datetime(dates)})


# =============================================================================
# 5. МОДЕЛЬ: Prophet, либо Fourier-GAM эквивалент как фолбэк
# =============================================================================
class _FourierGAM:
    """
    Воспроизводит математическую суть Prophet без самой библиотеки:

        y(t) = тренд(t) + сезонность_год(t) + сезонность_нед(t)
               + сезонность_сутки(t) + праздники(t) + b*price(t) + ошибка

    Сезонности — суммы синусов/косинусов (ряд Фурье), как у настоящего
    Prophet. Коэффициенты подбираются Ridge-регрессией (аналог MAP-оценки
    со штрафом, которую Prophet делает через L-BFGS). Доверительный
    интервал строится по эмпирическому разбросу остатков того же часа суток.
    """
    def __init__(self, yearly_order=6, weekly_order=3, daily_order=6,
                 interval_width=0.8, alpha=1.0, holidays=None, **_ignored):
        self.yearly_order = yearly_order
        self.weekly_order = weekly_order
        self.daily_order = daily_order
        self.interval_width = interval_width
        self.alpha = alpha
        self.holidays_df = holidays
        self.extra_regressors = {}
        self._reg_mu, self._reg_sd = {}, {}

    def add_regressor(self, name):
        self.extra_regressors[name] = True

    def _fourier_block(self, t_days, period_days, order):
        cols = {}
        for k in range(1, order + 1):
            arg = 2 * np.pi * k * t_days / period_days
            cols[f"p{period_days:.0f}_sin{k}"] = np.sin(arg)
            cols[f"p{period_days:.0f}_cos{k}"] = np.cos(arg)
        return pd.DataFrame(cols)

    def _design_matrix(self, ds: pd.Series, extra: pd.DataFrame | None, fit_mode: bool):
        t0 = self._t0 if hasattr(self, "_t0") else ds.min()
        t_days = (ds - t0).dt.total_seconds().values / 86400.0

        X = pd.DataFrame({"trend": t_days})
        X = pd.concat([X, self._fourier_block(t_days, 365.25, self.yearly_order)], axis=1)
        X = pd.concat([X, self._fourier_block(t_days, 7.0, self.weekly_order)], axis=1)
        X = pd.concat([X, self._fourier_block(t_days, 1.0, self.daily_order)], axis=1)

        if self.holidays_df is not None:
            hol_dates = set(self.holidays_df["ds"].dt.date)
            X["is_holiday"] = ds.dt.date.isin(hol_dates).astype(float).values

        if extra is not None:
            for name in self.extra_regressors:
                vals = extra[name].astype(float).values
                if fit_mode:
                    mu, sd = np.nanmean(vals), (np.nanstd(vals) or 1.0)
                    self._reg_mu[name], self._reg_sd[name] = mu, sd
                else:
                    mu, sd = self._reg_mu.get(name, 0.0), self._reg_sd.get(name, 1.0)
                X[name] = (vals - mu) / sd
        return X

    def fit(self, train_df: pd.DataFrame):
        self._t0 = train_df["ds"].min()
        X = self._design_matrix(train_df["ds"], train_df, fit_mode=True)
        y = train_df["y"].values

        self.model_ = Ridge(alpha=self.alpha)
        self.model_.fit(X.values, y)

        resid = y - self.model_.predict(X.values)
        hours = train_df["ds"].dt.hour.values
        self._resid_std_by_hour = pd.Series(resid).groupby(hours).std().reindex(range(24)).fillna(np.std(resid))

        self.history = train_df.copy()
        self._raw_regressor_history = train_df.set_index("ds")
        return self

    def predict(self, future_df: pd.DataFrame):
        X = self._design_matrix(future_df["ds"], future_df, fit_mode=False)
        yhat = self.model_.predict(X.values)

        z = norm.ppf(0.5 + self.interval_width / 2)
        hours = future_df["ds"].dt.hour.values
        sigma = self._resid_std_by_hour.reindex(hours).values

        out = future_df.copy()
        out["yhat"] = yhat
        out["yhat_lower"] = yhat - z * sigma
        out["yhat_upper"] = yhat + z * sigma
        return out


try:
    from prophet import Prophet
    _HAS_REAL_PROPHET = True
except ImportError:
    Prophet = _FourierGAM
    _HAS_REAL_PROPHET = False


def train_prophet(df: pd.DataFrame, target_col: str = "actual_consumption",
                   price_regressor: bool = True, changepoint_prior_scale: float = 0.05,
                   ridge_alpha: float = 1.0):
    """df — результат load_region_csv / validate_and_clean (индекс datetime, почасовые данные)."""
    years = range(df.index.year.min(), df.index.year.max() + 2)
    holidays_df = russian_holidays(years)

    train = pd.DataFrame({"ds": df.index, "y": df[target_col].values})
    if price_regressor and "average_price" in df.columns:
        train["price"] = df["average_price"].values

    if _HAS_REAL_PROPHET:
        model = Prophet(growth="linear", daily_seasonality=True, weekly_seasonality=True,
                         yearly_seasonality=True, holidays=holidays_df,
                         interval_width=0.8, changepoint_prior_scale=changepoint_prior_scale)
    else:
        model = Prophet(interval_width=0.8, alpha=ridge_alpha, holidays=holidays_df)

    if price_regressor and "price" in train.columns:
        model.add_regressor("price")

    model.fit(train)
    if not hasattr(model, "_raw_regressor_history"):
        model._raw_regressor_history = train.set_index("ds")
    return model


def apply_price_mode(df: pd.DataFrame, mode: str = "lag24") -> pd.DataFrame:
    """
    Как цена попадает в модель как регрессор.
      'lag24' — цена тех же суток назад (известна в 00:00 суток прогноза) — КОРРЕКТНО, по умолчанию;
      'now'   — цена в тот же час, что и прогноз. В момент прогноза её нет => утечка будущего
                (так работал прежний main.py; оставлено только для сравнения).
    Сдвиг делается по ВСЕЙ истории до нарезки на train/test, поэтому границы фолдов не искажаются.
    """
    if mode == "lag24" and "average_price" in df.columns:
        df = df.copy()
        df["average_price"] = df["average_price"].shift(24).bfill()
    elif mode not in ("lag24", "now"):
        raise ValueError("price_mode: 'lag24' или 'now'")
    return df



def forecast_horizon(model, timestamps: pd.DatetimeIndex, price_series: pd.Series | None = None):
    """Прогноз на заданные часы -> DataFrame[yhat, yhat_lower, yhat_upper] с индексом ds."""
    future = pd.DataFrame({"ds": timestamps})
    if "price" in model.extra_regressors:
        if price_series is not None:
            future["price"] = price_series.reindex(timestamps).values
        else:
            future["price"] = model._raw_regressor_history["price"].iloc[-1]
    forecast = model.predict(future)
    return forecast.set_index("ds")[["yhat", "yhat_lower", "yhat_upper"]]


def top3_peak_hours(forecast_df: pd.DataFrame, risk_col: str = "yhat_upper"):
    return forecast_df.sort_values(risk_col, ascending=False).head(3)


# =============================================================================
# 6. МЕТРИКИ ТОЧНОСТИ
# =============================================================================
def regression_accuracy(y_true, y_pred) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true))) * 100
    smape = float(np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))) * 100
    r2 = r2_score(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "MAPE_%": mape, "sMAPE_%": smape, "R2": r2}


def interval_coverage(y_true, lower, upper) -> float:
    inside = (y_true >= lower) & (y_true <= upper)
    return float(inside.mean()) * 100


def top3_hit_rate_by_day(test_df: pd.DataFrame, forecast_df: pd.DataFrame,
                          target_col: str = "actual_consumption", k: int = 3,
                          risk_col: str = "yhat_upper") -> pd.DataFrame:
    """
    Для каждых суток: сколько из k предсказанных пиковых часов (по risk_col)
    попало в k фактических. Колонки: date, hits, k, hit_rate_%.
    """
    merged = forecast_df.join(test_df[[target_col]], how="inner")
    rows = []
    for day, day_df in merged.groupby(merged.index.date):
        if len(day_df) < k:
            continue
        predicted_top = set(day_df.sort_values(risk_col, ascending=False).head(k).index)
        actual_top = set(day_df.sort_values(target_col, ascending=False).head(k).index)
        hits = len(predicted_top & actual_top)
        rows.append({"date": pd.Timestamp(day), "hits": hits, "k": k, "hit_rate_%": hits / k * 100})
    return pd.DataFrame(rows, columns=["date", "hits", "k", "hit_rate_%"])


def peak_hour_hit_rate(test_df: pd.DataFrame, forecast_df: pd.DataFrame,
                        target_col: str = "actual_consumption", k: int = 3) -> float:
    """Общая доля попаданий в топ-k по всем суткам (доля 0..1, не проценты)."""
    per_day = top3_hit_rate_by_day(test_df, forecast_df, target_col, k)
    return per_day["hits"].sum() / per_day["k"].sum() if len(per_day) else float("nan")


def evaluate_fold(train_df, test_df, target_col: str = "actual_consumption", **model_kwargs):
    """Обучает модель на train_df, прогнозирует test_df. Возвращает (метрики, прогноз)."""
    t0 = time.perf_counter()
    model = train_prophet(train_df, target_col=target_col, **model_kwargs)
    train_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    forecast_df = forecast_horizon(model, test_df.index, price_series=test_df.get("average_price"))
    predict_time = time.perf_counter() - t0

    y_true = test_df[target_col]
    metrics = regression_accuracy(y_true, forecast_df["yhat"])
    metrics["peak_hour_hit_rate_%"] = peak_hour_hit_rate(test_df, forecast_df, target_col) * 100
    metrics["interval_coverage_%"] = interval_coverage(y_true, forecast_df["yhat_lower"], forecast_df["yhat_upper"])
    metrics["train_time_sec"] = train_time
    metrics["predict_time_sec"] = predict_time
    return metrics, forecast_df


COMPARISON_CRITERIA = ["MAE", "RMSE", "MAPE_%", "sMAPE_%", "R2",
                        "peak_hour_hit_rate_%", "interval_coverage_%",
                        "train_time_sec", "predict_time_sec"]


# =============================================================================
# 7. БЭКТЕСТЫ
# =============================================================================
def rolling_backtest(df: pd.DataFrame, n_folds: int = 5, horizon_days: int = 1,
                      step_days: int = 14, price_mode: str = "lag24", **model_kwargs) -> pd.DataFrame:
    """Прогоняет evaluate_fold по нескольким walk-forward фолдам и собирает метрики в таблицу."""
    df = apply_price_mode(df, price_mode)
    folds = rolling_origin_folds(df, n_folds=n_folds, horizon_days=horizon_days, step_days=step_days)
    rows = []
    for i, (train_df, test_df) in enumerate(folds, 1):
        metrics, _ = evaluate_fold(train_df, test_df, **model_kwargs)
        metrics["fold"] = i
        metrics["test_start"] = test_df.index.min()
        rows.append(metrics)
    return pd.DataFrame(rows)


def _append_csv(path: str, frame: pd.DataFrame):
    """Дописывает строки в конец CSV (заголовок — только в новый файл), без перезаписи всего файла."""
    if os.path.exists(path) and os.path.getsize(path) > 0:
        header = list(pd.read_csv(path, nrows=0).columns)
        frame.reindex(columns=header).to_csv(path, mode="a", header=False, index=False)
    else:
        frame.to_csv(path, index=False)


def _backtest_one_day(df, day_start, target_col, price_regressor, model_kwargs):
    """
    Один фолд расширяющегося окна: train = всё строго ДО day_start, test = сутки day_start.
    Возвращает (день, метрики | None, почасовой факт/прогноз | None, замечание | None).
    Функция самодостаточна (ничего не печатает и не пишет на диск), поэтому
    безопасно запускается в отдельных процессах.
    """
    day_end = day_start + pd.Timedelta(hours=23)
    train_df = df[df.index < day_start]
    test_df = df[(df.index >= day_start) & (df.index <= day_end)]

    if train_df.empty or test_df.empty:
        return day_start, None, None, (f"  [пропуск] {day_start.date()}: нет данных "
                                         f"(train={len(train_df)}, test={len(test_df)})")
    note = None
    if len(test_df) < 24:
        note = f"  [предупреждение] {day_start.date()}: в тесте только {len(test_df)} ч из 24 (неполные сутки)"

    # Prophet оценивает интервал Монте-Карло на глобальном numpy RNG, а топ-3 часа
    # ранжируются по yhat_upper. Фиксируем seed от даты: результат не зависит ни от
    # порядка дней, ни от числа процессов, и повторяется от запуска к запуску.
    rng_state = np.random.get_state()
    np.random.seed(int(day_start.strftime("%Y%m%d")))
    try:
        metrics, forecast_df = evaluate_fold(train_df, test_df, target_col=target_col,
                                              price_regressor=price_regressor, **model_kwargs)
    finally:
        np.random.set_state(rng_state)

    metrics["forecast_date"] = str(day_start.date())
    metrics["train_hours"] = len(train_df)
    metrics["test_hours"] = len(test_df)
    hourly = forecast_df.join(test_df[[target_col]], how="inner").reset_index()  # колонки: ds, yhat*, target
    return day_start, metrics, hourly, note


def expanding_window_daily_backtest(df: pd.DataFrame, forecast_dates,
                                     target_col: str = "actual_consumption",
                                     price_regressor: bool = True,
                                     progress_every: int = 10,
                                     checkpoint_csv: str | None = None,
                                     hourly_csv: str | None = None,
                                     n_jobs: int = 1,
                                     resume: bool = False,
                                     price_mode: str = "lag24",
                                     **model_kwargs) -> pd.DataFrame:
    """
    "Расширяющееся окно" (expanding window), день за днём — так, как это нужно
    в продакшене: каждые сутки модель переобучается на ВСЕЙ накопленной
    истории и прогнозирует ровно следующие сутки:

        Фолд 1: train = [начало датасета .. D-1 23:00]  -> прогноз на день D
        Фолд 2: train = [начало датасета .. D   23:00]  -> прогноз на день D+1
        ...

    forecast_dates — даты, которые нужно спрогнозировать ("YYYY-MM-DD" или
    pd.Timestamp); для каждой train строится из ВСЕХ данных строго до 0:00 этой даты.

    n_jobs         — число процессов (как в joblib: -1 = все ядра). Фолды независимы
                     (actual уже известны), поэтому считаются параллельно; результат
                     тот же, что при n_jobs=1. Один день на настоящем Prophet — 20-30 с,
                     год — часы, с N ядрами примерно в N раз быстрее.
    checkpoint_csv — путь, куда после каждой порции дней ДОПИСЫВАЮТСЯ метрики
                     (без перезаписи всего файла) — прерванный прогон не теряет посчитанного.
    hourly_csv     — туда же дописывается почасовой факт/прогноз (ds, yhat, yhat_lower,
                     yhat_upper, <target_col>) по каждому дню — бесплатно, без лишних
                     переобучений. Из него потом можно вырезать любой период для
                     графика (см. load_hourly_forecast).
    resume         — True: если checkpoint_csv уже есть, уже посчитанные дни пропускаются
                     и прогон продолжается с места остановки. Если между запусками
                     менялась модель/параметры — поставь False (или удали файлы),
                     иначе старые дни останутся от прошлой версии.
    progress_every — печатать строку прогресса примерно раз в N дней.
    price_mode     — 'lag24' (по умолчанию): регрессор цены = цена сутки назад, известна на момент прогноза.
                     'now': цена того же часа (прежнее поведение, УТЕЧКА будущего — только для сравнения).

    Возвращает DataFrame с метриками по каждому дню + итоговую строку
    "СРЕДНЕЕ" — готовую таблицу для сравнения с другими моделями проекта.
    """
    df = apply_price_mode(df, price_mode)
    dates = pd.DatetimeIndex(pd.to_datetime(list(forecast_dates))).normalize().unique().sort_values()
    n_total = len(dates)
    wanted_iso = {str(d.date()) for d in dates}

    # --- возобновление / чистый старт ---------------------------------------
    rows: list[dict] = []
    if resume and checkpoint_csv and os.path.exists(checkpoint_csv):
        prev = pd.read_csv(checkpoint_csv)
        # строку "СРЕДНЕЕ" (если прошлый прогон дописал итог в этот же файл) и чужие даты отбрасываем
        prev = prev[prev["forecast_date"].astype(str).isin(wanted_iso)].copy()
        for col in ("train_hours", "test_hours"):
            if col in prev.columns:
                prev[col] = prev[col].astype(int)
        prev.to_csv(checkpoint_csv, index=False)  # файл без итоговой строки — можно дописывать
        rows = prev.to_dict("records")
        done_iso = set(prev["forecast_date"].astype(str))
        print(f"[RESUME] Уже посчитано {len(done_iso)} из {n_total} дней в '{checkpoint_csv}' — они пропускаются. "
              f"(Менял модель? Удали файл или поставь resume=False.)")
        if hourly_csv and os.path.exists(hourly_csv):
            hourly_prev = pd.read_csv(hourly_csv)
            day_of_row = pd.to_datetime(hourly_prev["ds"]).dt.strftime("%Y-%m-%d")
            keep = day_of_row.isin(done_iso)  # хвост недописанного дня выкидываем — он пересчитается
            if not keep.all():
                hourly_prev[keep].to_csv(hourly_csv, index=False)
            if len(done_iso - set(day_of_row[keep])):
                print(f"[RESUME] В '{hourly_csv}' нет почасовых данных для части уже посчитанных дней "
                      f"— для графиков по ним перезапусти с resume=False.")
        elif hourly_csv and done_iso:
            print(f"[RESUME] '{hourly_csv}' не найден — для уже посчитанных дней почасовых данных не будет.")
    else:
        for path in (checkpoint_csv, hourly_csv):
            if path and os.path.exists(path):
                os.remove(path)
        done_iso = set()

    todo = [d for d in dates if str(d.date()) not in done_iso]
    n_workers = max(1, effective_n_jobs(n_jobs))
    chunk_size = 1 if n_workers == 1 else n_workers * 2
    if n_workers > 1:
        print(f"[PARALLEL] {n_workers} процессов, порциями по {chunk_size} дней")

    # --- основной цикл по порциям дней ---------------------------------------
    t_start = time.perf_counter()
    n_done_now, last_printed = 0, 0
    for pos in range(0, len(todo), chunk_size):
        chunk = todo[pos:pos + chunk_size]
        args = (target_col, price_regressor, model_kwargs)
        if n_workers == 1:
            results = [_backtest_one_day(df, d, *args) for d in chunk]
        else:
            results = Parallel(n_jobs=n_workers)(delayed(_backtest_one_day)(df, d, *args) for d in chunk)

        new_metrics, new_hourly = [], []
        for _day, metrics, hourly, note in results:
            if note:
                print(note)
            if metrics is not None:
                new_metrics.append(metrics)
                new_hourly.append(hourly)
        rows.extend(new_metrics)
        if checkpoint_csv and new_metrics:
            _append_csv(checkpoint_csv, pd.DataFrame(new_metrics))
        if hourly_csv and new_hourly:
            _append_csv(hourly_csv, pd.concat(new_hourly, ignore_index=True))

        n_done_now += len(chunk)
        is_last = pos + chunk_size >= len(todo)
        if progress_every and new_metrics and (n_done_now - last_printed >= progress_every or is_last):
            last_printed = n_done_now
            avg_per_day = (time.perf_counter() - t_start) / n_done_now
            eta_min = avg_per_day * (len(todo) - n_done_now) / 60
            last = new_metrics[-1]
            print(f"  [{len(done_iso) + n_done_now}/{n_total}] {last['forecast_date']}  "
                  f"MAPE={last['MAPE_%']:.1f}%  hit-rate={last['peak_hour_hit_rate_%']:.0f}%  "
                  f"(~{avg_per_day:.1f} сек/день, осталось ~{eta_min:.1f} мин)")

    table = pd.DataFrame(rows)
    if len(table):
        table = table.sort_values("forecast_date").reset_index(drop=True)
        numeric_cols = [c for c in COMPARISON_CRITERIA if c in table.columns]
        mean_row = table[numeric_cols].mean()
        mean_row["forecast_date"] = "СРЕДНЕЕ"
        table = pd.concat([table, pd.DataFrame([mean_row])], ignore_index=True)
    return table


# =============================================================================
# 8. ТАБЛИЦЫ РЕЗУЛЬТАТОВ
# =============================================================================
# Человекочитаемые названия метрик для вывода/отчёта
PRETTY_METRIC_NAMES = {
    "forecast_date":          "Дата",
    "train_hours":             "Часов в train",
    "test_hours":               "Часов в test",
    "MAE":                      "Ошибка, МВт*ч (MAE)",
    "RMSE":                     "Ошибка, МВт*ч (RMSE)",
    "MAPE_%":                   "Ошибка, % (MAPE)",
    "sMAPE_%":                  "Ошибка, % (sMAPE)",
    "R2":                       "R² (объясн. дисперсия)",
    "peak_hour_hit_rate_%":     "Попадание в топ-3, %",
    "interval_coverage_%":      "Покрытие 80%-интервала, %",
    "train_time_sec":           "Время обучения, сек",
    "predict_time_sec":         "Время прогноза, сек",
}


def format_backtest_table(table: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """
    Готовит таблицу результатов backtest'а для чтения человеком:
    переименовывает колонки на русский, округляет числа до разумной
    точности. Исходный table (с техническими именами колонок) не меняет —
    возвращает копию.
    """
    columns = columns or ["forecast_date", "MAE", "MAPE_%", "sMAPE_%", "R2",
                           "peak_hour_hit_rate_%", "interval_coverage_%"]
    out = table[columns].copy()

    round_map = {"MAE": 0, "RMSE": 0, "MAPE_%": 2, "sMAPE_%": 2, "R2": 3,
                 "peak_hour_hit_rate_%": 1, "interval_coverage_%": 1,
                 "train_time_sec": 1, "predict_time_sec": 2}
    for col, ndigits in round_map.items():
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").round(ndigits)

    return out.rename(columns=PRETTY_METRIC_NAMES)


def monthly_rollup(table: pd.DataFrame) -> pd.DataFrame:
    """
    Для длинного бэктеста (например, на весь год) 365 строк читать
    неудобно — сворачивает дневные метрики по месяцам (среднее по
    месяцу), плюс отдельно строку "ГОД" со средним по всем дням.
    """
    daily = table[table["forecast_date"] != "СРЕДНЕЕ"].copy()
    daily["month"] = pd.to_datetime(daily["forecast_date"]).dt.to_period("M").astype(str)

    numeric_cols = [c for c in COMPARISON_CRITERIA if c in daily.columns]
    rollup = daily.groupby("month")[numeric_cols].mean().reset_index()
    rollup = rollup.rename(columns={"month": "forecast_date"})

    year_row = daily[numeric_cols].mean()
    year_row["forecast_date"] = "ГОД (среднее)"
    return pd.concat([rollup, pd.DataFrame([year_row])], ignore_index=True)


def load_hourly_forecast(hourly_csv: str, start=None, end=None, target_col: str = "actual_consumption"):
    """
    Читает почасовой файл расширяющегося окна и вырезает период [start, end]
    (даты включительно) -> (test_df, forecast_df), готовые для plot_forecast_vs_actual.
    """
    h = pd.read_csv(hourly_csv, parse_dates=["ds"]).set_index("ds").sort_index()
    if start is not None:
        h = h[h.index >= pd.Timestamp(start)]
    if end is not None:
        h = h[h.index < pd.Timestamp(end) + pd.Timedelta(days=1)]
    return h[[target_col]], h[["yhat", "yhat_lower", "yhat_upper"]]


# =============================================================================
# 9. ГРАФИКИ
# =============================================================================
def plot_data_quality(raw_df, clean_df, target_col="actual_consumption", save_path=None):
    """Сырой ряд с подсвеченными выбросами/пропусками vs очищенный ряд."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    axes[0].plot(raw_df.index, raw_df[target_col], color="tab:red", linewidth=0.7, label="Сырые данные")
    axes[0].set_title("До очистки")
    axes[0].legend(); axes[0].set_ylabel("МВт*ч")

    axes[1].plot(clean_df.index, clean_df[target_col], color="tab:green", linewidth=0.7, label="После очистки")
    axes[1].set_title("После очистки (заполнены пропуски/выбросы)")
    axes[1].legend(); axes[1].set_ylabel("МВт*ч"); axes[1].set_xlabel("Дата")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_seasonal_profile(df, target_col="actual_consumption", save_path=None):
    """Средний суточный профиль потребления: будни vs выходные."""
    d = df.copy()
    d["hour"] = d.index.hour
    d["is_weekend"] = d.index.dayofweek >= 5
    profile = d.groupby(["is_weekend", "hour"])[target_col].mean().unstack(0)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(profile.index, profile[False], marker="o", label="Будни", color="tab:blue")
    ax.plot(profile.index, profile[True], marker="o", label="Выходные", color="tab:orange")
    ax.set_title("Средний суточный профиль потребления")
    ax.set_xlabel("Час суток"); ax.set_ylabel("Потребление, МВт*ч")
    ax.set_xticks(range(0, 24, 2))
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_forecast_vs_actual(test_df, forecast_df, target_col="actual_consumption",
                             title="Прогноз vs факт", save_path=None):
    """Прогноз + интервал + факт, с отметкой фактических и предсказанных топ-3 часов пиков."""
    merged = forecast_df.join(test_df[[target_col]], how="inner")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True,
                                    gridspec_kw={"height_ratios": [3, 1]})
    ax1.fill_between(merged.index, merged["yhat_lower"], merged["yhat_upper"],
                      color="tab:blue", alpha=0.2, label="80% доверительный интервал")
    ax1.plot(merged.index, merged["yhat"], color="tab:blue", label="Прогноз (yhat)")
    ax1.plot(merged.index, merged[target_col], color="tab:red", linewidth=1.2, label="Факт")

    for i, (_, day_df) in enumerate(merged.groupby(merged.index.date)):
        if len(day_df) < 3:
            continue
        pred_top3 = day_df.sort_values("yhat_upper", ascending=False).head(3)
        actual_top3 = day_df.sort_values(target_col, ascending=False).head(3)
        ax1.scatter(pred_top3.index, pred_top3["yhat"], marker="v", color="tab:blue", s=60, zorder=5,
                    label="Предсказанный топ-3" if i == 0 else None)
        ax1.scatter(actual_top3.index, actual_top3[target_col], marker="^", color="tab:red", s=60, zorder=5,
                    label="Фактический топ-3" if i == 0 else None)

    ax1.set_title(title); ax1.set_ylabel("МВт*ч")
    ax1.legend(loc="upper left", fontsize=8)

    residual = merged[target_col] - merged["yhat"]
    ax2.bar(merged.index, residual, width=0.03, color="gray")
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_ylabel("Остаток\n(факт-прогноз)"); ax2.set_xlabel("Дата")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_fold_metrics(fold_metrics: pd.DataFrame, criteria=("MAPE_%", "peak_hour_hit_rate_%"), save_path=None):
    """Как метрики ведут себя по фолдам walk-forward бэктеста — средняя точность и её разброс."""
    fig, axes = plt.subplots(1, len(criteria), figsize=(6 * len(criteria), 4.5))
    if len(criteria) == 1:
        axes = [axes]
    for ax, crit in zip(axes, criteria):
        ax.plot(fold_metrics["fold"], fold_metrics[crit], marker="o", color="tab:blue")
        mean_v = fold_metrics[crit].mean()
        ax.axhline(mean_v, color="tab:red", linestyle="--", label=f"среднее = {mean_v:.2f}")
        ax.set_title(crit); ax.set_xlabel("Фолд (walk-forward)")
        ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_year_overview(table: pd.DataFrame, rollup: pd.DataFrame, period_label: str = "", save_path=None):
    """Дневная динамика MAPE и hit-rate + помесячная сводка расширяющегося окна (4 графика)."""
    daily = table[table["forecast_date"] != "СРЕДНЕЕ"].copy()
    daily["forecast_date"] = pd.to_datetime(daily["forecast_date"])
    monthly = rollup[rollup["forecast_date"] != "ГОД (среднее)"]
    suffix = f" ({period_label})" if period_label else ""

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    axes[0, 0].plot(daily["forecast_date"], daily["MAPE_%"], color="tab:blue", linewidth=0.8)
    axes[0, 0].axhline(daily["MAPE_%"].mean(), color="tab:red", linestyle="--",
                        label=f"среднее = {daily['MAPE_%'].mean():.2f}%")
    axes[0, 0].set_title(f"MAPE по дням{suffix}"); axes[0, 0].set_ylabel("%"); axes[0, 0].legend()

    axes[0, 1].plot(daily["forecast_date"], daily["peak_hour_hit_rate_%"], color="tab:green", linewidth=0.8)
    axes[0, 1].axhline(daily["peak_hour_hit_rate_%"].mean(), color="tab:red", linestyle="--",
                        label=f"среднее = {daily['peak_hour_hit_rate_%'].mean():.1f}%")
    axes[0, 1].set_title("Попадание в топ-3 по дням"); axes[0, 1].set_ylabel("%"); axes[0, 1].legend()

    axes[1, 0].bar(monthly["forecast_date"], monthly["MAPE_%"], color="tab:blue")
    axes[1, 0].set_title("MAPE по месяцам"); axes[1, 0].set_ylabel("%")
    axes[1, 0].tick_params(axis="x", rotation=45)

    axes[1, 1].bar(monthly["forecast_date"], monthly["peak_hour_hit_rate_%"], color="tab:green")
    axes[1, 1].set_title("Попадание в топ-3 по месяцам"); axes[1, 1].set_ylabel("%")
    axes[1, 1].tick_params(axis="x", rotation=45)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


# #############################################################################
# 10. ЕДИНЫЙ ПРОТОКОЛ ОЦЕНКИ PROPHET (задачи 1-2)
#     Точка прогноза — 00:00 суток D, горизонт 24 ч, переобучение каждый день (расширяющееся окно);
#     один тест-период и одна предобработка для всех методов; гиперпараметры — только на валидации до теста;
#     оценка «по всем часам» и отдельно «пиковый режим» (топ-k часов суток).
# #############################################################################
# =============================================================================
# 10.1 ПРОТОКОЛ: ДАННЫЕ ПАНЕЛИ РЕГИОНОВ И ПРОСТРАНСТВЕННАЯ СТРУКТУРА
# =============================================================================
def discover_region_files(folder: str = ".", pattern: str = "electricity_*.csv") -> dict[str, str]:
    """{'75': 'electricity_75.csv', '71': ...} по шаблону имени."""
    out = {}
    for p in sorted(glob.glob(os.path.join(folder, pattern))):
        m = re.search(r"electricity_(\d+)\.csv$", os.path.basename(p))
        if m:
            out[m.group(1)] = p
    return out


def load_panel(region_files: dict[str, str], cache_dir: str, target_col: str = "actual_consumption",
               encoding: str | None = None):
    """
    Каждый регион проходит ОДНУ И ТУ ЖЕ предобработку (prophet_pipeline.validate_and_clean),
    затем все регионы выравниваются на общую почасовую сетку (пересечение периодов).
    Возвращает (cons: DataFrame[время x регион], price: DataFrame | None).
    """
    os.makedirs(cache_dir, exist_ok=True)
    cons, price = {}, {}
    for code, path in region_files.items():
        df = load_clean_cached(path, os.path.join(cache_dir, f"clean_{code}.pkl"),
                               target_col=target_col, region=code, encoding=encoding)
        cons[code] = df[target_col]
        if "average_price" in df.columns:
            price[code] = df["average_price"]
    start = max(s.index.min() for s in cons.values())
    end = min(s.index.max() for s in cons.values())
    cons_df = pd.DataFrame(cons).loc[start:end].asfreq("h").ffill()
    price_df = None
    if len(price) == len(cons):
        price_df = pd.DataFrame(price).loc[start:end].asfreq("h").ffill()
    return cons_df, price_df


def spatial_neighbors(cons: pd.DataFrame, fit_end: pd.Timestamp, k: int = 3):
    """
    Пространственная связь по ДАННЫМ (не по географии): корреляция остатков потребления
    между регионами после удаления профиля «день недели x час» и недельного тренда.
    Считается ТОЛЬКО по данным до начала теста (fit_end) — утечки нет.
    Возвращает (матрица корреляций, {регион: [k самых связанных регионов]}).
    """
    tr = np.log(cons.loc[:fit_end])
    prof = tr.groupby([tr.index.dayofweek, tr.index.hour]).transform("mean")
    resid = tr - prof
    resid = resid - resid.rolling(168, min_periods=24).mean()
    corr = resid.corr()
    nbrs = {}
    for r in corr.columns:
        nbrs[r] = list(corr[r].drop(r).sort_values(ascending=False).index[:k])
    return corr, nbrs


# =============================================================================
# 10.2 ПРИЗНАКИ (все лаги >= 24 ч относительно целевого часа)
# =============================================================================
def _region_features(y: pd.Series) -> pd.DataFrame:
    """
    Единственный признак — r_l24 = y(t-24ч) / scale, где scale = среднее за 168 ч, оканчивающихся в t-24ч.
    Прогноз делается в 00:00 суток t, поэтому для t известно только y(s) при s <= t-24ч => утечки нет.
    Отношение к scale делает ряды разных регионов сопоставимыми (используется для регрессора соседей).
    """
    scale = y.rolling(168, min_periods=168).mean().shift(24)
    f = pd.DataFrame({"r_l24": y.shift(24) / scale}, index=y.index)
    return f.replace([np.inf, -np.inf], np.nan)


class Context:
    """Общие данные: панель регионов, цены, соседи, лаг-24 соседей, настройки."""

    def __init__(self, cons: pd.DataFrame, price: pd.DataFrame | None, neighbors: dict,
                 max_train_days: int | None = None, seed: int = 42):
        self.cons, self.price, self.neighbors = cons, price, neighbors
        self.regions = list(cons.columns)
        self.max_train_days, self.seed = max_train_days, seed
        self.feats = {r: _region_features(cons[r]) for r in self.regions}
        for r in self.regions:                                                  # лаг-24 соседей (среднее по k соседям)
            nb = neighbors.get(r, [])
            self.feats[r]["nb_r_l24"] = np.mean([self.feats[n]["r_l24"].values for n in nb], axis=0) if nb else np.nan

    def day_index(self, day: pd.Timestamp) -> pd.DatetimeIndex:
        return pd.date_range(day, periods=24, freq="h")

    def window(self, obj, day: pd.Timestamp, mtd: int | None):
        """Данные строго ДО day 00:00; при mtd — только последние mtd дней."""
        lo = None if mtd is None else day - pd.Timedelta(days=mtd)
        return obj.loc[lo: day - H1]


# =============================================================================
# 10.3 МЕТОД (Prophet и его варианты) (единый интерфейс: predict_day(day, regions) -> {регион: DataFrame[yhat, lower, upper]})
# =============================================================================
def _frame(idx, yhat, lo=None, hi=None) -> pd.DataFrame:
    nan = np.full(len(idx), np.nan)
    return pd.DataFrame({"yhat": np.asarray(yhat, float),
                         "yhat_lower": nan if lo is None else np.asarray(lo, float),
                         "yhat_upper": nan if hi is None else np.asarray(hi, float)}, index=idx)


class Method:
    DEFAULT_PARAMS: dict = {}
    PARAM_GRID: dict = {}

    def __init__(self, ctx: Context, **params):
        self.ctx = ctx
        self.params = {**self.DEFAULT_PARAMS, **params}
        self.mtd = self.params.pop("max_train_days", ctx.max_train_days)

    def predict_day(self, day, regions):
        raise NotImplementedError


class ProphetM(Method):
    """
    Prophet (если установлен; иначе Fourier-GAM из prophet_pipeline — тогда это НЕ Prophet).
    reg — список внешних регрессоров:
        'price_now'   — цена в тот же час (как в текущем main.py; в момент прогноза её нет => УТЕЧКА, для справки)
        'price_lag24' — цена те же сутки назад (известна на момент прогноза)
        'nbr'         — лаг-24 потребления соседних регионов (пространственный регрессор)
    """
    DEFAULT_PARAMS = {"cps": 0.05, "ridge_alpha": 1.0, "reg": ()}
    PARAM_GRID = {"cps": [0.01, 0.05, 0.1]} if _HAS_REAL_PROPHET else {"ridge_alpha": [0.1, 1.0, 10.0, 100.0]}

    def _regressors(self, r) -> pd.DataFrame:
        c, cols = self.ctx, {}
        for name in self.params["reg"]:
            if name in ("price_now", "price_lag24"):
                if c.price is None:
                    continue                                    # нет цены хотя бы в одном регионе — регрессор не используем
                cols["price"] = c.price[r] if name == "price_now" else c.price[r].shift(24)
            elif name == "nbr":
                cols["nbr"] = c.feats[r]["nb_r_l24"]
        return pd.DataFrame(cols, index=c.cons.index)

    def predict_day(self, day, regions):
        c, p, idx, out = self.ctx, self.params, self.ctx.day_index(day), {}
        for r in regions:
            y = c.window(c.cons[r], day, self.mtd)
            regs = self._regressors(r)
            train = pd.DataFrame({"ds": y.index, "y": y.values})
            future = pd.DataFrame({"ds": idx})
            for col in regs:
                train[col] = regs[col].reindex(y.index).values
                future[col] = regs[col].reindex(idx).ffill().bfill().values
            train = train.dropna()
            hol = russian_holidays(range(train["ds"].dt.year.min(), day.year + 2))
            if _HAS_REAL_PROPHET:
                m = Prophet(growth="linear", daily_seasonality=True, weekly_seasonality=True,
                            yearly_seasonality=True, holidays=hol, interval_width=0.8,
                            changepoint_prior_scale=p["cps"])
            else:
                m = Prophet(interval_width=0.8, alpha=p["ridge_alpha"], holidays=hol)
            for col in regs:
                m.add_regressor(col)
            state = np.random.get_state()                       # как в prophet_pipeline: seed от даты => воспроизводимо
            np.random.seed(int(day.strftime("%Y%m%d")))
            try:
                m.fit(train)
                fc = m.predict(future)
            finally:
                np.random.set_state(state)
            out[r] = _frame(idx, fc["yhat"], fc["yhat_lower"], fc["yhat_upper"])
        return out


# имя -> (класс, фиксированные параметры). Эти параметры НЕ подбираются, подбирается PARAM_GRID класса.
REGISTRY = {
    "prophet_current": (ProphetM, {"reg": ("price_now",)}),            # как в прежнем main.py (утечка цены), для справки
    "prophet":         (ProphetM, {"reg": ("price_lag24",)}),          # честный: цена сутки назад
    "prophet_nbr":     (ProphetM, {"reg": ("price_lag24", "nbr")}),    # + лаг-24 соседних регионов (пространство)
}


def make_method(name: str, ctx: Context, tuned: dict | None = None, **extra) -> Method:
    cls, fixed = REGISTRY[name]
    return cls(ctx, **{**fixed, **(tuned or {}), **extra})


# =============================================================================
# 10.4 ЗАПУСК: параллелизм, чекпоинты, подбор гиперпараметров
# =============================================================================
def _run_chunk(method: Method, name: str, jobs: list):
    """jobs: [(день, [регионы])]. Возвращает (DataFrame прогнозов | None, [ошибки])."""
    frames, errors = [], []
    for day, regions in jobs:
        try:
            for r, fc in method.predict_day(day, regions).items():
                f = fc.copy()
                f.index.name = "ds"
                f = f.reset_index()
                f["region"], f["method"] = r, name
                f["y"] = method.ctx.cons[r].reindex(f["ds"]).values
                frames.append(f)
        except Exception as e:                                   # один сбойный день не должен ронять год расчёта
            errors.append((str(day.date()), repr(e)))
    return (pd.concat(frames, ignore_index=True) if frames else None), errors


FC_COLS = ["method", "region", "ds", "y", "yhat", "yhat_lower", "yhat_upper"]


def _predict_days(method, name, jobs, n_jobs, chunk_days, on_batch=None):
    n_workers = max(1, effective_n_jobs(n_jobs))
    chunks = [jobs[i:i + chunk_days] for i in range(0, len(jobs), chunk_days)]
    batch = n_workers * 2 if n_workers > 1 else 1
    all_frames, all_err, done, t0 = [], [], 0, time.perf_counter()
    for b in range(0, len(chunks), batch):
        group = chunks[b:b + batch]
        if n_workers > 1:
            res = Parallel(n_jobs=n_workers)(delayed(_run_chunk)(method, name, c) for c in group)
        else:
            res = [_run_chunk(method, name, c) for c in group]
        new = [f for f, _ in res if f is not None]
        for _, e in res:
            all_err += e
        if new:
            df = pd.concat(new, ignore_index=True)[FC_COLS]
            all_frames.append(df)
            if on_batch:
                on_batch(df)
        done += sum(len(c) for c in group)
        el = time.perf_counter() - t0
        print(f"    [{name}] {done}/{len(jobs)} дн.  (~{el / done:.1f} с/день, осталось ~{el / done * (len(jobs) - done) / 60:.1f} мин)")
    out = pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame(columns=FC_COLS)
    return out, all_err


def run_method(name, ctx, days, regions, out_dir, tuned=None, n_jobs=1, chunk_days=5, resume=True):
    """Прогноз method на days x regions; почасовые прогнозы дописываются в out_dir/forecasts/<method>.csv."""
    os.makedirs(os.path.join(out_dir, "forecasts"), exist_ok=True)
    path = os.path.join(out_dir, "forecasts", f"{name}.csv")
    sig = json.loads(json.dumps({"params": tuned or {}, "max_train_days": ctx.max_train_days}))
    sig_path = os.path.join(out_dir, "forecasts", f"{name}.sig.json")
    if resume and os.path.exists(path):
        old_sig = json.load(open(sig_path)) if os.path.exists(sig_path) else None
        if old_sig != sig:
            print(f"  [{name}] параметры или окно обучения изменились — старые прогнозы удалены, считаем заново")
            os.remove(path)
    json.dump(sig, open(sig_path, "w"))
    have = set()
    if resume and os.path.exists(path):
        old = pd.read_csv(path, usecols=["region", "ds"], parse_dates=["ds"], dtype={"region": str})
        cnt = old.groupby(["region", old["ds"].dt.normalize()]).size()
        have = set(cnt[cnt == 24].index)
    elif os.path.exists(path):
        os.remove(path)
    jobs = []
    for d in days:
        miss = [r for r in regions if (r, d) not in have]
        if miss:
            jobs.append((d, miss))
    if not jobs:
        print(f"  [{name}] всё уже посчитано — пропуск")
        return
    method = make_method(name, ctx, tuned)
    print(f"  [{name}] к расчёту {len(jobs)} дн. (регионов: {len(regions)}), параметры: {tuned or 'по умолчанию'}")

    def _append(df):
        df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)

    _, errors = _predict_days(method, name, jobs, n_jobs, chunk_days, on_batch=_append)
    if errors:
        print(f"  [{name}] ОШИБКИ в {len(errors)} дн., они выпадут из общей выборки: {errors[:3]}")


def load_forecasts(out_dir: str, methods=None) -> pd.DataFrame:
    parts = []
    for p in sorted(glob.glob(os.path.join(out_dir, "forecasts", "*.csv"))):
        if methods is None or os.path.splitext(os.path.basename(p))[0] in methods:
            parts.append(pd.read_csv(p, parse_dates=["ds"], dtype={"region": str}))
    fc = pd.concat(parts, ignore_index=True)
    return fc.drop_duplicates(["method", "region", "ds"], keep="last")


def tune_method(name, ctx, val_days, regions, n_jobs=1, criterion="mape+peak_mape"):
    """
    Перебор PARAM_GRID класса метода на валидации (дни строго до теста, то же расширяющееся окно).
    Критерий: MAPE по всем часам + MAPE в фактические топ-3 часа (среднее по регионам) — баланс «в целом»
    и «в пиках»; чистый peak-критерий поощрял бы завышение прогноза. Тест при этом не используется.
    """
    cls, fixed = REGISTRY[name]
    grid = cls.PARAM_GRID
    if not grid:
        return {}, pd.DataFrame()
    keys, rows = list(grid), []
    jobs = [(d, list(regions)) for d in val_days]
    for vals in itertools.product(*grid.values()):
        combo = dict(zip(keys, vals))
        method = make_method(name, ctx, combo)
        fc, err = _predict_days(method, name, jobs, n_jobs, chunk_days=max(1, len(jobs) // max(1, 2 * effective_n_jobs(n_jobs))))
        if fc.empty:
            continue
        daily = evaluate_forecasts(common_days(fc))
        s = daily[["mape", "peak_mape"]].mean()
        rows.append({"method": name, **combo, "val_MAPE": s["mape"], "val_peak_MAPE": s["peak_mape"],
                     "criterion": s["mape"] + s["peak_mape"], "val_days": daily["date"].nunique()})
        print(f"    tune[{name}] {combo}: MAPE={s['mape']:.2f}  peakMAPE={s['peak_mape']:.2f}")
    log = pd.DataFrame(rows).sort_values("criterion").reset_index(drop=True)
    best = {k: log.loc[0, k] for k in keys}
    best = {k: (v.item() if hasattr(v, "item") else v) for k, v in best.items()}
    return best, log


# =============================================================================
# 10.5 МЕТРИКИ И СТАТИСТИКА
# =============================================================================
def common_days(fc: pd.DataFrame) -> pd.DataFrame:
    """Оставляет только пары (регион, сутки), которые есть у ВСЕХ методов, целиком (24 ч) и без NaN."""
    fc = fc.dropna(subset=["y", "yhat"]).copy()
    fc["date"] = fc["ds"].dt.normalize()
    cnt = fc.groupby(["region", "date", "method"]).size().unstack("method")
    ok = (cnt == 24).all(axis=1)
    keep = ok[ok].index.to_frame(index=False)
    return fc.merge(keep, on=["region", "date"]).sort_values(["method", "region", "ds"]).reset_index(drop=True)


def _daily_metrics(Y, P, R, L, U, k: int) -> dict:
    """Y,P,R,L,U: (дни x 24). R — по чему ранжируем часы (yhat или yhat_upper)."""
    err = np.abs(P - Y)
    mt = np.zeros_like(Y, bool)
    mp = np.zeros_like(Y, bool)
    np.put_along_axis(mt, np.argsort(-Y, axis=1, kind="stable")[:, :k], True, axis=1)
    np.put_along_axis(mp, np.argsort(-R, axis=1, kind="stable")[:, :k], True, axis=1)
    dil = mt.copy()                                    # фактические пики +-1 час
    dil[:, 1:] |= mt[:, :-1]
    dil[:, :-1] |= mt[:, 1:]
    at, ap = Y.argmax(1), R.argmax(1)
    return {
        "mae": err.mean(1), "mse": (err ** 2).mean(1), "mape": (err / Y).mean(1) * 100,
        "smape": (2 * err / (np.abs(P) + np.abs(Y))).mean(1) * 100,
        # --- пиковый режим ---
        "hits": (mt & mp).sum(1),                                          # сколько из k предсказанных часов угадано точно
        "hits_tol": (dil & mp).sum(1),                                     # ... с допуском +-1 час
        "top1": (at == ap).astype(float), "top1_tol": (np.abs(at - ap) <= 1).astype(float),
        "capture": np.where(mp, Y, 0).sum(1) / np.where(mt, Y, 0).sum(1) * 100,  # какую долю пиковой нагрузки «поймали»
        "peak_mae": (err * mt).sum(1) / k, "peak_mape": (err / Y * mt).sum(1) / k * 100,
        "peak_bias": ((P - Y) * mt).sum(1) / k,                            # знак: <0 — пики занижаются
        "dmax_mape": np.abs(P.max(1) - Y.max(1)) / Y.max(1) * 100,
        "coverage": ((Y >= L) & (Y <= U)).mean(1) * 100 if not np.isnan(L).any() else np.full(len(Y), np.nan),
    }


def evaluate_forecasts(fc: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    """fc — результат common_days. Возвращает дневную таблицу метрик (method, region, date, ...)."""
    rows = []
    for (m, r), g in fc.groupby(["method", "region"], sort=False):
        g = g.sort_values("ds")
        arr = lambda c: g[c].to_numpy(float).reshape(-1, 24)
        Y, P, L, U = arr("y"), arr("yhat"), arr("yhat_lower"), arr("yhat_upper")
        days = pd.DatetimeIndex(g["ds"].dt.normalize().iloc[::24])
        variants = [(P, m)]
        if not np.isnan(U).any():
            variants.append((U, f"{m} [ранг по upper]"))
        for R, label in variants:
            d = pd.DataFrame(_daily_metrics(Y, P, R, L, U, k))
            d["method"], d["region"], d["date"] = label, r, days
            rows.append(d)
    return pd.concat(rows, ignore_index=True)


def summarize(daily: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    daily = daily.assign(all3=(daily["hits"] == k).astype(float) * 100, zero=(daily["hits"] == 0).astype(float) * 100)
    s = daily.groupby(["method", "region"], sort=False).agg(
        days=("mae", "size"), MAE=("mae", "mean"), MSE=("mse", "mean"), MAPE=("mape", "mean"),
        sMAPE=("smape", "mean"), top3_hit=("hits", "mean"), top3_tol1=("hits_tol", "mean"),
        top1_hit=("top1", "mean"), top1_tol1=("top1_tol", "mean"), capture=("capture", "mean"),
        peak_MAE=("peak_mae", "mean"), peak_MAPE=("peak_mape", "mean"), peak_bias=("peak_bias", "mean"),
        dmax_MAPE=("dmax_mape", "mean"), coverage80=("coverage", "mean"),
        all3_days=("all3", "mean"), zero_days=("zero", "mean")).reset_index()
    s["RMSE"] = np.sqrt(s.pop("MSE"))
    for c in ("top3_hit", "top3_tol1"):
        s[c] = s[c] / k * 100
    for c in ("top1_hit", "top1_tol1"):
        s[c] *= 100
    cols = ["method", "region", "days", "top3_hit", "top3_tol1", "all3_days", "zero_days", "top1_hit", "top1_tol1",
            "capture", "peak_MAE", "peak_MAPE", "peak_bias", "dmax_MAPE", "MAE", "RMSE", "MAPE", "sMAPE", "coverage80"]
    return s[cols]


def summarize_all(summary: pd.DataFrame) -> pd.DataFrame:
    """Среднее по регионам (макро-среднее) — по одной строке на метод."""
    num = [c for c in summary.columns if c not in ("method", "region")]
    return summary.groupby("method", sort=False)[num].mean().reset_index()


def paired_stat(a: pd.Series, b: pd.Series, n_boot: int = 2000, block: int = 7, seed: int = 0) -> dict:
    """
    Парное сравнение по дням: d = b - a. Среднее d, 95% ДИ блочным бутстрэпом (блок = неделя, ряд
    автокоррелирован) и p-value критерия Уилкоксона (ненулевые d, двусторонний).
    """
    d = (b - a).dropna().to_numpy(float)
    n = len(d)
    if n < 5:
        return {"diff": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "p": np.nan, "n": n}
    rng = np.random.default_rng(seed)
    block = max(1, min(block, n // 4))
    nb = math.ceil(n / block)
    starts = rng.integers(0, n, size=(n_boot, nb))
    idx = (starts[:, :, None] + np.arange(block)) % n
    means = d[idx].reshape(n_boot, -1)[:, :n].mean(1)
    try:
        p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
    except ValueError:
        p = np.nan
    return {"diff": d.mean(), "ci_lo": np.quantile(means, 0.025), "ci_hi": np.quantile(means, 0.975), "p": p, "n": n}


MIN_DAYS_FOR_VERDICT = 30
LOWER_IS_BETTER = {"mae": True, "mape": True, "peak_mape": True, "peak_mae": True, "dmax_mape": True,
                   "hits": False, "hits_tol": False, "capture": False, "top1": False}


def pairs_table(daily: pd.DataFrame, pairs: list[tuple[str, str]],
                metrics=("hits", "hits_tol", "top1", "capture", "peak_mae", "peak_mape", "mae", "mape"),
                regions=None) -> pd.DataFrame:
    """Для каждой пары (база, новый) и метрики: разница (новый - база), ДИ, p, вывод. По регионам и «ВСЕ» (среднее по регионам по дням)."""
    out = []
    regs = regions or sorted(daily["region"].unique())
    for base, new in pairs:
        for metric in metrics:
            piv = daily.pivot_table(index=["region", "date"], columns="method", values=metric)
            if base not in piv or new not in piv:
                continue
            for scope in regs + ["ВСЕ"]:
                if scope == "ВСЕ":
                    sub = piv[[base, new]].groupby(level="date").mean()
                else:
                    sub = piv[[base, new]].xs(scope, level="region")
                st = paired_stat(sub[base], sub[new])
                lower = LOWER_IS_BETTER[metric]
                better = (st["ci_hi"] < 0) if lower else (st["ci_lo"] > 0)
                worse = (st["ci_lo"] > 0) if lower else (st["ci_hi"] < 0)
                if st["n"] < MIN_DAYS_FOR_VERDICT:
                    verdict = f"мало дней (n<{MIN_DAYS_FOR_VERDICT}) — вывод не делаем"
                else:
                    verdict = "лучше" if better else "хуже" if worse else "без значимой разницы"
                out.append({"база": base, "новый": new, "регион": scope, "метрика": metric,
                            "среднее_база": sub[base].mean(), "среднее_новый": sub[new].mean(), **st,
                            "вывод(95% ДИ)": verdict})
    return pd.DataFrame(out)


# =============================================================================
# 10.6 ВЫВОД: графики и markdown
# =============================================================================
def df_to_md(df: pd.DataFrame, digits: int = 2) -> str:
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else f"{v:.{digits}f}"
        return str(v)
    head = "| " + " | ".join(map(str, df.columns)) + " |\n|" + "---|" * len(df.columns) + "\n"
    return head + "\n".join("| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)) + "\n"


def plot_summary_bars(s_all: pd.DataFrame, path: str, title: str = ""):
    s = s_all.sort_values("peak_MAPE")
    fig, axes = plt.subplots(1, 3, figsize=(16, max(4, 0.35 * len(s) + 1.5)), sharey=True)
    for ax, col, lab in zip(axes, ["MAPE", "peak_MAPE", "top3_hit"],
                            ["MAPE, все часы, %", "MAPE в фактические топ-3 часа, %", "Попадание в топ-3 (точно), %"]):
        ax.barh(s["method"], s[col], color="tab:green" if col == "top3_hit" else "tab:blue")
        ax.set_title(lab)
        ax.grid(axis="x", alpha=0.3)
    axes[0].invert_yaxis()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_corr(corr: pd.DataFrame, path: str):
    fig, ax = plt.subplots(figsize=(1 + 0.8 * len(corr), 1 + 0.7 * len(corr)))
    im = ax.imshow(corr.values, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns)
    ax.set_yticks(range(len(corr)), corr.index)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", color="w", fontsize=8)
    ax.set_title("Корреляция остатков потребления между регионами (до начала теста)")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_spatial_delta(tbl: pd.DataFrame, metric: str, path: str):
    t = tbl[(tbl["метрика"] == metric) & (tbl["регион"] != "ВСЕ")]
    if t.empty:
        return
    t = t.assign(пара=t["база"] + " → " + t["новый"]).pivot(index="регион", columns="пара", values="diff")
    fig, ax = plt.subplots(figsize=(2 + 2.2 * t.shape[1], 1.5 + 0.5 * len(t)))
    v = np.nanmax(np.abs(t.values)) or 1
    im = ax.imshow(t.values, cmap="RdBu_r" if LOWER_IS_BETTER[metric] else "RdBu", vmin=-v, vmax=v, aspect="auto")
    ax.set_xticks(range(t.shape[1]), t.columns, rotation=20, ha="right")
    ax.set_yticks(range(len(t)), t.index)
    for i in range(t.shape[0]):
        for j in range(t.shape[1]):
            ax.text(j, i, f"{t.values[i, j]:+.2f}", ha="center", va="center", fontsize=8,
                    color="w" if abs(t.values[i, j]) > 0.6 * v else "k")
    ax.set_title(f"Δ {metric} (новый − база), синий = лучше" if LOWER_IS_BETTER[metric] else f"Δ {metric} (новый − база)")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


RU_WEEKDAYS = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"]


def plot_daily_forecasts_pdf(fc: pd.DataFrame, path: str, title: str, main_method: str = "prophet", k: int = 3):
    """
    Один PDF, по странице на месяц; каждая клетка — ОДИН день: факт и прогноз по 24 часам.
    Красные кружки — фактические топ-k часов, треугольники — топ-k часов по прогнозу main_method;
    в заголовке клетки — сколько из k часов угадано. fc — прогнозы ОДНОГО региона (после common_days).
    """
    from matplotlib.backends.backend_pdf import PdfPages
    methods = sorted(fc["method"].unique(), key=lambda m: (m != main_method, m))
    palette = {main_method: dict(color="tab:blue", lw=1.5), "prophet_current": dict(color="tab:gray", lw=1.1, ls="--"),
               "prophet_nbr": dict(color="tab:green", lw=1.3)}
    fc = fc.copy()
    fc["date"] = fc["ds"].dt.normalize()
    with PdfPages(path) as pdf:
        for (yr, mo), g in fc.groupby([fc["date"].dt.year, fc["date"].dt.month]):
            days = sorted(g["date"].unique())
            ncol = 6
            nrow = math.ceil(len(days) / ncol)
            fig, axes = plt.subplots(nrow, ncol, figsize=(24, 3.3 * nrow + 1.3), squeeze=False)
            for ax in axes.ravel():
                ax.axis("off")
            for ax, d in zip(axes.ravel(), days):
                ax.axis("on")
                gd = g[g["date"] == d]
                act = gd.drop_duplicates("ds").sort_values("ds")
                hrs = act["ds"].dt.hour.to_numpy()
                ya = act["y"].to_numpy()
                ax.plot(hrs, ya, "k-", lw=1.8, label="факт")
                top_a = np.argsort(-ya)[:k]
                ax.scatter(hrs[top_a], ya[top_a], s=55, facecolors="none", edgecolors="red", linewidths=1.5, zorder=5,
                           label=f"факт: топ-{k} часа")
                hit = None
                for m in methods:
                    gm = gd[gd["method"] == m].sort_values("ds")
                    st = palette.get(m, dict(color="tab:orange", lw=1.2))
                    ax.plot(gm["ds"].dt.hour, gm["yhat"], label=m, **st)
                    if m == main_method:
                        yp = gm["yhat"].to_numpy()
                        top_p = np.argsort(-yp)[:k]
                        ax.scatter(hrs[top_p], yp[top_p], marker="^", s=45, color=st["color"], zorder=6,
                                   label=f"прогноз: топ-{k} часа ({m})")
                        hit = len(set(top_a) & set(top_p))
                ax.set_title(f"{pd.Timestamp(d):%d.%m} {RU_WEEKDAYS[pd.Timestamp(d).dayofweek]}"
                             + (f"   угадано {hit}/{k}" if hit is not None else ""), fontsize=10)
                ax.set_xticks([0, 6, 12, 18, 23])
                ax.tick_params(labelsize=7)
                ax.grid(alpha=0.3)
            h, l = axes[0, 0].get_legend_handles_labels()
            fig_h = fig.get_size_inches()[1]
            fig.suptitle(f"{title} — {yr}-{mo:02d}", y=1 - 0.12 / fig_h, va="top", fontsize=13)
            fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 1 - 0.55 / fig_h), ncol=len(l), fontsize=10)
            fig.tight_layout(rect=(0, 0, 1, 1 - 1.0 / fig_h))
            pdf.savefig(fig)
            plt.close(fig)