from pathlib import Path

import holidays
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lightgbm import LGBMRegressor
from mlforecast import MLForecast
from mlforecast.lag_transforms import RollingMean, RollingStd

ROOT = Path(__file__).resolve().parent
DATASETS = ROOT.parents[1] / "datasets"

REGION = "Челябинская область"
TRAIN_START = pd.Timestamp("2022-01-01")
FORECAST_START = pd.Timestamp("2026-01-01")
FORECAST_STOP = pd.Timestamp("2027-01-01")
ONE_DAY = pd.Timedelta(days=1)

EXOG = ["is_weekend", "is_holiday", "is_day_off", "is_pre_holiday",
        "is_post_holiday", "days_to_holiday", "days_from_holiday",
        "is_new_year_period", "is_may_holidays"]

BASE_PARAMS = dict(boosting_type="gbdt", objective="regression", n_estimators=200,
                   learning_rate=0.05, num_leaves=31, max_depth=-1,
                   min_child_samples=20, min_child_weight=0.001,
                   min_split_gain=0.0, subsample=1.0, subsample_freq=0,
                   colsample_bytree=1.0, reg_alpha=0.0, reg_lambda=0.0,
                   max_bin=255, subsample_for_bin=200000, random_state=42,
                   n_jobs=2, deterministic=True, force_col_wise=True, verbosity=-1)

CONFIGS = {
    "original": {"params": BASE_PARAMS, "lags": [1, 24, 25, 168, 169],
                 "date_features": ["hour", "dayofweek", "month"], "rolling": False},
    "extended_200": {"params": {**BASE_PARAMS, "min_child_samples": 40, "reg_lambda": 1.0},
                     "lags": [1, 2, 24, 25, 48, 72, 168, 169, 336],
                     "date_features": ["hour", "dayofweek", "month", "dayofyear"], "rolling": True},
    "extended_400": {"params": {**BASE_PARAMS, "n_estimators": 400, "learning_rate": 0.03,
                                  "min_child_samples": 40, "reg_lambda": 1.0},
                     "lags": [1, 2, 24, 25, 48, 72, 168, 169, 336],
                     "date_features": ["hour", "dayofweek", "month", "dayofyear"], "rolling": True},
}

PRE_HOLIDAYS = ["2022-02-22", "2022-03-05", "2022-04-30", "2022-05-07", "2022-06-11", "2022-11-03",
                "2023-02-22", "2023-03-07", "2023-04-28", "2023-05-05", "2023-06-09", "2023-11-03",
                "2024-02-22", "2024-03-07", "2024-04-27", "2024-05-08", "2024-06-11", "2024-11-02",
                "2025-01-08", "2025-02-22", "2025-03-07", "2025-04-30", "2025-05-08", "2025-06-11", "2025-11-03"]

def numeric(s):
    return pd.to_numeric(s.astype(str).str.replace("МВт*ч", "", regex=False)
                         .str.replace(r"\s+", "", regex=True)
                         .str.replace(",", ".", regex=False), errors="coerce")

def add_calendar(frame, years):
    frame = frame.copy()
    cal = holidays.Russia(years=years)
    hd = np.array(sorted(cal.keys()), dtype="datetime64[D]")
    days = frame.ds.to_numpy().astype("datetime64[D]")
    left = np.searchsorted(hd, days, side="left")
    right = np.searchsorted(hd, days, side="right") - 1
    frame["is_weekend"] = frame.ds.dt.dayofweek.ge(5).astype(int)
    frame["is_holiday"] = np.isin(days, hd).astype(int)
    frame["is_day_off"] = (frame.is_weekend.eq(1) | frame.is_holiday.eq(1)).astype(int)
    frame["is_pre_holiday"] = np.isin(days, np.array(PRE_HOLIDAYS, dtype="datetime64[D]")).astype(int)
    frame["is_post_holiday"] = np.isin(days, hd + np.timedelta64(1, "D")).astype(int)
    frame["days_to_holiday"] = np.where(left < len(hd), (hd[np.minimum(left, len(hd) - 1)] - days).astype(int), 365)
    frame["days_from_holiday"] = np.where(right >= 0, (days - hd[np.maximum(right, 0)]).astype(int), 365)
    frame["is_new_year_period"] = ((frame.ds.dt.month.eq(1) & frame.ds.dt.day.le(8)) |
                                     (frame.ds.dt.month.eq(12) & frame.ds.dt.day.eq(31))).astype(int)
    frame["is_may_holidays"] = (frame.ds.dt.month.eq(5) & frame.ds.dt.day.le(10)).astype(int)
    return frame

def build_forecaster(name):
    c = CONFIGS[name]
    transforms = {1: [RollingMean(24), RollingMean(168), RollingStd(24)],
                  24: [RollingMean(24)], 168: [RollingMean(24)]} if c["rolling"] else None
    return MLForecast(models={"forecast": LGBMRegressor(**c["params"])}, freq="h",
                      lags=c["lags"], date_features=c["date_features"],
                      lag_transforms=transforms, num_threads=1)

def metrics(y, pred):
    y, pred = np.asarray(y, dtype=float), np.asarray(pred, dtype=float)
    valid = np.isfinite(y) & np.isfinite(pred)
    y, pred = y[valid], pred[valid]
    err = np.abs(y - pred)
    if not len(y):
        return {"n": 0, "MAE": np.nan, "RMSE": np.nan, "WAPE_pct": np.nan, "MAPE_pct": np.nan}
    nonzero = y != 0
    return {"n": len(y), "MAE": float(err.mean()), "RMSE": float(np.sqrt(np.mean(err ** 2))),
            "WAPE_pct": float(err.sum() / np.abs(y).sum() * 100) if np.abs(y).sum() else np.nan,
            "MAPE_pct": float(np.mean(err[nonzero] / np.abs(y[nonzero])) * 100) if nonzero.any() else np.nan}

def load_data(path=DATASETS / "electricity_75.csv", region=REGION, stop=FORECAST_STOP):
    try:
        raw = pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        raw = pd.read_csv(path, encoding="cp1251")

    raw = raw.loc[raw.region_name.eq(region), ["date", "hour", "actual_consumption"]].copy()
    if raw.empty:
        raise ValueError(f"В {path} нет данных для региона «{region}»")

    raw["ds"] = pd.to_datetime(raw.date) + pd.to_timedelta(raw.hour, unit="h")
    raw["observed"] = numeric(raw.actual_consumption)
    raw["observed"] = raw.observed.where(raw.observed.gt(0))
    observed_by_hour = raw.groupby("ds")["observed"]
    if observed_by_hour.nunique().gt(1).any():
        raise ValueError(f"В {path} есть разные факты за один и тот же час")

    hours = pd.date_range(TRAIN_START, stop, freq="h", inclusive="left", name="ds")
    data = observed_by_hour.first().reindex(hours).reset_index()
    data["y"] = data.observed.ffill()
    data["unique_id"] = region
    return data


def forecast_period(data, start, stop, mode):
    train = data.loc[data.ds.lt(start), ["unique_id", "ds", "y"] + EXOG]
    future = data.loc[data.ds.ge(start) & data.ds.lt(stop)]
    model = build_forecaster("extended_400")
    model.fit(train, static_features=[])
    predictions = model.predict(len(future), X_df=future[["unique_id", "ds"] + EXOG])

    actual = future[["ds", "observed"]].rename(columns={"observed": "y"})
    output = actual.merge(
        predictions[["ds", "forecast"]].rename(columns={"forecast": "y_pred"}),
        on="ds", how="left", validate="one_to_one",
    )
    if output.y_pred.isna().any():
        raise ValueError(f"Получен неполный прогноз с {start} до {stop}")
    output["mode"] = mode
    return output


def peak_results(frame):
    values = []
    for day, g in frame.groupby(frame.ds.dt.normalize()):
        if len(g) != 24 or g.y.isna().any():
            continue
        a = g.sort_values(["y", "ds"], ascending=[False, True]).head(3)
        b = g.sort_values(["y_pred", "ds"], ascending=[False, True]).head(3)
        overlap = len(set(a.ds) & set(b.ds))
        values.append({"date": day, "actual": " | ".join(a.ds.dt.strftime("%H:%M")),
                       "predicted": " | ".join(b.ds.dt.strftime("%H:%M")), "hits": overlap,
                       "MAE": abs(a.y - a.y_pred).mean()})
    return pd.DataFrame(values, columns=["date", "actual", "predicted", "hits", "MAE"])


def save_forecast_plot(output, last_actual):
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(output.ds, output.y, label="Факт", linewidth=0.8)
    ax.plot(output.ds, output.y_pred, label="Прогноз", linewidth=0.8)
    ax.axvline(last_actual, color="darkred", linestyle="--", label="Последний доступный факт")
    ax.set(title=f"{REGION}: факт и прогноз за 2026 год", xlabel="Дата", ylabel="МВт·ч")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(ROOT / "forecast.png", dpi=140)
    plt.close(fig)


def save_errors_plot(observed):
    daily = pd.DataFrame([
        {"date": day, **metrics(g.y, g.y_pred)}
        for day, g in observed.groupby(observed.ds.dt.normalize())
    ])
    labels = {
        "MAE": "MAE, МВт·ч",
        "RMSE": "RMSE, МВт·ч",
        "WAPE_pct": "WAPE, %",
        "MAPE_pct": "MAPE, %",
    }
    fig, axes = plt.subplots(2, 2, figsize=(15, 7), sharex=True, layout="constrained")
    for ax, (column, label) in zip(axes.flat, labels.items()):
        ax.plot(daily.date, daily[column], color="steelblue", linewidth=1)
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    fig.suptitle(f"Ошибки почасового прогноза по дням — {REGION}")
    fig.supxlabel("Дата")
    fig.savefig(ROOT / "errors.png", dpi=140)
    plt.close(fig)


def main():
    data = add_calendar(load_data(), range(TRAIN_START.year, FORECAST_STOP.year))
    last_actual = data.loc[data.observed.notna(), "ds"].max()
    if pd.isna(last_actual) or last_actual < FORECAST_START:
        raise ValueError("В исходных данных нет фактических наблюдений за 2026 год")

    days = pd.date_range(FORECAST_START, last_actual.normalize(), freq="D")
    result = []
    for i, day in enumerate(days, 1):
        result.append(forecast_period(data, day, day + ONE_DAY, "forecast_with_actual_2026"))
        if i == 1 or i % 20 == 0 or i == len(days):
            print(f"[{i}/{len(days)}] Прогноз за {day.date()} готов", flush=True)

    recursive_start = days[-1] + ONE_DAY
    if recursive_start < FORECAST_STOP:
        print(f"Рекурсивный прогноз с {recursive_start.date()} до конца года", flush=True)
        after_last_actual = last_actual + pd.Timedelta(hours=1)
        recursive = forecast_period(data, after_last_actual, FORECAST_STOP, "recursive_without_actual_2026")
        result.append(recursive.loc[recursive.ds.ge(recursive_start)])

    output = pd.concat(result, ignore_index=True)
    observed = output.loc[output.y.notna()]
    peaks = peak_results(observed)
    summary = pd.DataFrame([{
        "year": FORECAST_START.year,
        **metrics(observed.y, observed.y_pred),
        "complete_days": len(peaks),
        "peak_MAE": peaks.MAE.mean(),
        "peak_top3_hit_pct": peaks.hits.mean() / 3 * 100,
    }])

    output.to_csv(ROOT / "forecast_2026.csv", index=False)
    summary.to_csv(ROOT / "metrics.csv", index=False)
    save_forecast_plot(output, last_actual)
    save_errors_plot(observed)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
