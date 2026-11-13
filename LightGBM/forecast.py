import argparse
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

REGION = "Челябинская область"

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

def load_data():
    raw = pd.read_csv(ROOT / "electricity_75.csv")
    raw = raw.loc[raw.region_name.eq(REGION)].copy()
    raw["ds"] = pd.to_datetime(raw.date) + pd.to_timedelta(raw.hour, unit="h")
    raw["observed"] = numeric(raw.actual_consumption)
    if raw.ds.duplicated().any():
        raise ValueError("В данных есть дубликаты дат и часов")
    raw = raw.set_index("ds").reindex(pd.date_range("2022-01-01", "2026-12-31 23:00", freq="h"))
    raw.index.name = "ds"
    data = raw.reset_index()
    data["observed"] = data.observed.where(data.observed.gt(0))
    data["y"] = data.observed.ffill()
    data["unique_id"] = REGION
    return data


def forecast_day(data, day, years):
    data = add_calendar(data, years)
    train = data.loc[data.ds.lt(day), ["unique_id", "ds", "y"] + EXOG]
    future = data.loc[data.ds.ge(day) & data.ds.lt(day + pd.Timedelta(days=1))]
    model = build_forecaster("extended_400")
    model.fit(train, static_features=[])
    predictions = model.predict(24, X_df=future[["unique_id", "ds"] + EXOG])
    return predictions.forecast.to_numpy()


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
    return pd.DataFrame(values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2025-01-01", choices=["2025-01-01", "2026-01-01"])
    args = parser.parse_args()
    data = load_data()
    last_actual = data.loc[data.observed.notna(), "ds"].max()
    last_day = last_actual.normalize()
    if last_actual.hour != 23:
        raise ValueError("Последний день с фактом должен быть полным")
    result = []
    start = pd.Timestamp(args.start)
    for i, day in enumerate(pd.date_range(start, last_day), 1):
        years = [2022, 2023, 2024, 2025] if day.year == 2025 else [2022, 2023, 2024, 2025, 2026]
        current = data.loc[data.ds.lt(day + pd.Timedelta(days=1))].copy()
        pred = forecast_day(current, day, years)
        test = data.loc[data.ds.ge(day) & data.ds.lt(day + pd.Timedelta(days=1)), ["ds", "observed"]].copy()
        test.columns = ["ds", "y"]
        test["y_pred"] = pred
        test["mode"] = "test_2025" if day.year == 2025 else "forecast_with_actual_2026"
        result.append(test)
        if i == 1 or i % 10 == 0:
            pd.concat(result).to_csv(ROOT / "progress.csv", index=False)
            print(day.date(), i, flush=True)
    frame = add_calendar(data, [2022, 2023, 2024, 2025, 2026])
    train = frame.loc[frame.ds.le(last_actual), ["unique_id", "ds", "y"] + EXOG]
    future = frame.loc[frame.ds.gt(last_actual)]
    model = build_forecaster("extended_400")
    model.fit(train, static_features=[])
    forecast = model.predict(len(future), X_df=future[["unique_id", "ds"] + EXOG])
    tail = future[["ds", "observed"]].copy()
    tail.columns = ["ds", "y"]
    tail["y_pred"] = forecast.forecast.to_numpy()
    tail["mode"] = "recursive_without_actual_2026"
    result.append(tail)
    output = pd.concat(result, ignore_index=True)
    if args.start == "2026-01-01":
        existing = pd.read_csv(ROOT / "test_2025.csv", parse_dates=["ds"])
        output = pd.concat([existing, output], ignore_index=True)
    output.to_csv(ROOT / "forecast_2025_2026.csv", index=False)
    observed = output.loc[output.y.notna()].copy()
    rows = []
    peaks = []
    for year, g in observed.groupby(observed.ds.dt.year):
        rows.append({"year": year, **metrics(g.y, g.y_pred)})
        pk = peak_results(g)
        pk["year"] = year
        peaks.append(pk)
    pd.DataFrame(rows).to_csv(ROOT / "metrics.csv", index=False)
    pd.concat(peaks).to_csv(ROOT / "peak_hours.csv", index=False)
    fig, ax = plt.subplots(figsize=(16, 5))
    for label, col in [("Факт", "y"), ("Прогноз", "y_pred")]:
        ax.plot(output.ds, output[col], label=label, linewidth=0.7)
    ax.axvline(last_actual, color="red", linestyle="--", label="Последний доступный факт")
    ax.set(title="Челябинская область: факт и прогноз", ylabel="МВт·ч")
    ax.grid(alpha=.3); ax.legend(); fig.tight_layout(); fig.savefig(ROOT / "forecast.png", dpi=140); plt.close(fig)
    daily = []
    for day, g in observed.groupby(observed.ds.dt.normalize()):
        daily.append({"date": day, **metrics(g.y, g.y_pred)})
    daily = pd.DataFrame(daily)
    daily.to_csv(ROOT / "daily_metrics.csv", index=False)
    for column in ["MAE", "RMSE", "WAPE_pct", "MAPE_pct"]:
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(daily.date, daily[column], linewidth=.8)
        ax.set(title=f"{column} по дням", xlabel="Дата", ylabel=column)
        ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(ROOT / f"{column}.png", dpi=140); plt.close(fig)
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
