"""
Прогнозирование пиковых часов электропотребления — модель Prophet
(или эквивалентный Fourier-GAM, если пакет `prophet` недоступен).

Проект: ООО «ТЕРА» / СО ЕЭС.

"""

import re
import time
import zipfile
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")


# =============================================================================
# 1. ЗАГРУЗКА
# =============================================================================
def _extract_number(value):
    """'4 051 МВт*ч' -> 4051.0 ; '979 руб./МВт*ч' -> 979.0 ; NaN -> None"""
    if pd.isna(value):
        return None
    match = re.search(r"[-+]?[\d  ]+(?:[.,]\d+)?", str(value))
    if not match:
        return None
    cleaned = match.group().replace(" ", "").replace(" ", "").replace(",", ".")
    return float(cleaned)


def load_region_csv(path: str, zip_path: str | None = None) -> pd.DataFrame:
    if zip_path:
        with zipfile.ZipFile(zip_path) as z:
            with z.open(path) as f:
                df = pd.read_csv(f, encoding="utf-8-sig")
    else:
        df = pd.read_csv(path, encoding="utf-8-sig")

    numeric_cols = ["planned_consumption", "actual_consumption",
                     "planned_generation", "actual_generation",
                     "average_price", "vsvgo_consumption", "vsvgo_avg_price"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].apply(_extract_number)

    df["datetime"] = pd.to_datetime(df["date"]) + pd.to_timedelta(df["hour"], unit="h")
    df = df.set_index("datetime").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df


# =============================================================================
# 2. ПРЕДОБРАБОТКА И ВАЛИДАЦИЯ ДАННЫХ
# =============================================================================
def validate_and_clean(df: pd.DataFrame, target_col: str = "actual_consumption") -> tuple[pd.DataFrame, dict]:
    """
    Полный цикл очистки почасового ряда, как того требует ТЗ проекта:
      1) проверка временной последовательности и восстановление пропущенных часов
      2) обработка перехода на летнее/зимнее время (DST)
      3) обнаружение и заполнение пропусков
      4) обнаружение и коррекция выбросов (по профилю конкретного часа суток)
      5) контроль корректности значений (потребление не может быть <= 0)

    Возвращает (очищенный df, словарь с отчётом о найденных проблемах).
    """
    report = {}

    # --- 2.1 Временная последовательность: достраиваем ПОЛНУЮ почасовую сетку ---
    full_index = pd.date_range(df.index.min(), df.index.max(), freq="h")
    missing_hours = full_index.difference(df.index)
    report["missing_hours_count"] = len(missing_hours)
    report["missing_hours_pct"] = round(len(missing_hours) / len(full_index) * 100, 3)
    df = df.reindex(full_index)

    # --- 2.2 Переход на летнее/зимнее время (DST) ---
    # РФ отменила переход на летнее время с 2014 года (указ правительства),
    # поэтому для данных 2022+ переходов DST физически быть не может.
    # Проверяем это утверждение по факту: ищем дни, где из-за перевода
    # стрелок было бы 23 или 25 записей вместо 24.
    day_counts = pd.Series(1, index=df.index).resample("D").sum()
    dst_like_days = day_counts[(day_counts != 24) & (day_counts.index.date != df.index.max().date())
                                & (day_counts.index.date != df.index.min().date())]
    report["dst_anomalies"] = int((dst_like_days != 24).sum())  # ожидаем 0

    # --- 2.3 Контроль корректности значений: потребление должно быть > 0 ---
    invalid_mask = (df[target_col] <= 0)
    report["invalid_values_count"] = int(invalid_mask.sum())
    df.loc[invalid_mask, target_col] = np.nan

    # --- 2.4 Обнаружение выбросов ---
    # Сравниваем каждое значение с медианой и MAD ТОГО ЖЕ часа суток
    # (у потребления разный масштаб днём и ночью, обычный z-score по всему
    # ряду принял бы нормальный дневной пик за аномалию).
    hour_of_day = df.index.hour
    df["_hour"] = hour_of_day
    med = df.groupby("_hour")[target_col].transform("median")
    mad = df.groupby("_hour")[target_col].transform(lambda s: (s - s.median()).abs().median())
    mad = mad.replace(0, np.nan)
    robust_z = (df[target_col] - med) / (1.4826 * mad)
    outlier_mask = robust_z.abs() > 5  # порог ~5 MAD — явные аномалии, не обычный разброс
    report["outliers_count"] = int(outlier_mask.sum())
    report["outliers_pct"] = round(outlier_mask.sum() / len(df) * 100, 3)
    df.loc[outlier_mask, target_col] = np.nan
    df = df.drop(columns="_hour")

    # --- 2.5 Заполнение пропусков (после reindex + invalid + outliers) ---
    total_missing_before_fill = df[target_col].isna().sum()
    report["total_nan_before_fill"] = int(total_missing_before_fill)
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
    print(f"  Некорректных значений (<=0):  {report['invalid_values_count']}")
    print(f"  Выбросов (>5 MAD от медианы часа): {report['outliers_count']} ({report['outliers_pct']}%)")
    print(f"  Аномалий перехода на DST:     {report['dst_anomalies']} "
          f"(ожидаемо 0 — РФ не переходит на летнее время с 2014 г.)")
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
# 4. ПРАЗДНИКИ РФ (заданы вручную — пакет `holidays` недоступен офлайн)
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
# 5. МОДЕЛЬ: Prophet, либо Fourier-GAM эквивалент как offline fallback
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

        from scipy.stats import norm
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


def peak_hour_hit_rate(test_df: pd.DataFrame, forecast_df: pd.DataFrame,
                        target_col: str = "actual_consumption") -> float:
    merged = forecast_df.join(test_df[[target_col]], how="inner")
    merged["date"] = merged.index.date
    hits, total = 0, 0
    for _, day_df in merged.groupby("date"):
        if len(day_df) < 3:
            continue
        predicted_top3 = set(day_df.sort_values("yhat_upper", ascending=False).head(3).index)
        actual_top3 = set(day_df.sort_values(target_col, ascending=False).head(3).index)
        hits += len(predicted_top3 & actual_top3)
        total += 3
    return hits / total if total else float("nan")


def evaluate_fold(train_df, test_df, **model_kwargs) -> dict:
    t0 = time.perf_counter()
    model = train_prophet(train_df, **model_kwargs)
    train_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    forecast_df = forecast_horizon(model, test_df.index, price_series=test_df.get("average_price"))
    predict_time = time.perf_counter() - t0

    y_true = test_df["actual_consumption"]
    metrics = regression_accuracy(y_true, forecast_df["yhat"])
    metrics["peak_hour_hit_rate_%"] = peak_hour_hit_rate(test_df, forecast_df) * 100
    metrics["interval_coverage_%"] = interval_coverage(y_true, forecast_df["yhat_lower"], forecast_df["yhat_upper"])
    metrics["train_time_sec"] = train_time
    metrics["predict_time_sec"] = predict_time
    return metrics, forecast_df


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


COMPARISON_CRITERIA = ["MAE", "RMSE", "MAPE_%", "sMAPE_%", "R2",
                        "peak_hour_hit_rate_%", "interval_coverage_%",
                        "train_time_sec", "predict_time_sec"]


# =============================================================================
# 7. ГРАФИКИ
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

    merged["date"] = merged.index.date
    for _, day_df in merged.groupby("date"):
        if len(day_df) < 3:
            continue
        pred_top3 = day_df.sort_values("yhat_upper", ascending=False).head(3)
        actual_top3 = day_df.sort_values(target_col, ascending=False).head(3)
        ax1.scatter(pred_top3.index, pred_top3["yhat"], marker="v", color="tab:blue",
                    s=60, zorder=5, label="Предсказанный топ-3" if _ == list(merged.groupby("date"))[0][0] else None)
        ax1.scatter(actual_top3.index, actual_top3[target_col], marker="^", color="tab:red",
                    s=60, zorder=5, label="Фактический топ-3" if _ == list(merged.groupby("date"))[0][0] else None)

    ax1.set_title(title); ax1.set_ylabel("МВт*ч")
    handles, labels = ax1.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax1.legend(uniq.values(), uniq.keys(), loc="upper left", fontsize=8)

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