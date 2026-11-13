"""
Прогнозирование пиковых часов электропотребления — модель Prophet
(или эквивалентный Fourier-GAM, если пакет `prophet` не установлен).

Проект: ООО «ТЕРА» / СО ЕЭС.

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

Если `prophet` установлен, он подхватывается автоматически (интерфейс
_FourierGAM повторяет нужную часть интерфейса Prophet) — менять код не нужно.
Какой движок работает, показывает флаг _HAS_REAL_PROPHET. Fourier-GAM —
математический аналог Prophet (линейный тренд + Fourier-сезонности +
праздники + регрессор + интервал по остаткам); числа с ним НЕ равны числам
настоящего Prophet, для отчётов используй настоящий.
"""

import logging
import os
import re
import time
import warnings
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, effective_n_jobs
from scipy.stats import norm
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")
# cmdstanpy на каждый fit пишет по несколько INFO-строк; на 365 переобучениях
# они забивают консоль и прячут строки прогресса.
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)


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
                      step_days: int = 14, **model_kwargs) -> pd.DataFrame:
    """Прогоняет evaluate_fold по нескольким walk-forward фолдам и собирает метрики в таблицу."""
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

    Возвращает DataFrame с метриками по каждому дню + итоговую строку
    "СРЕДНЕЕ" — готовую таблицу для сравнения с другими моделями проекта.
    """
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
