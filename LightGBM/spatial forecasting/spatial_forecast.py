"""Сравнение трёх вариантов прогноза по пяти регионам."""

from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "walk-forward forecasting with an expanding window"))

from forecast import (
    DATASETS,
    EXOG,
    FORECAST_START,
    TRAIN_START,
    add_calendar,
    build_forecaster,
    load_data,
    metrics,
    peak_results,
)

# Идентификатор в выгрузках СО ЕЭС и координаты административного центра.
REGIONS = {
    "Кировская область": ("33", 58.604, 49.668),
    "Пермский край": ("57", 58.011, 56.250),
    "Тюменская область": ("71", 57.153, 65.535),
    "Челябинская область": ("75", 55.164, 61.437),
    "Удмуртская Республика": ("94", 56.853, 53.207),
}
ID_TIME = ["unique_id", "ds"]
TARGET = [*ID_TIME, "y"]
FEATURES = {
    "local": EXOG,
    "interregional": EXOG + ["peer_lag24", "peer_lag168", "peer_change"],
    "spatial": EXOG + ["neighbor_lag24", "neighbor_lag168", "neighbor_change"],
}


def add_peer_features(data, region, normalized_load):
    peers = [name for name in REGIONS if name != region]
    lat, lon = np.radians(REGIONS[region][1:])
    peer_lat, peer_lon = np.radians([REGIONS[name][1:] for name in peers]).T
    haversine = np.sin((peer_lat - lat) / 2) ** 2
    haversine += np.cos(lat) * np.cos(peer_lat) * np.sin((peer_lon - lon) / 2) ** 2
    distance = 2 * 6371 * np.arcsin(np.sqrt(haversine))
    weights = pd.Series(1 / distance, index=peers)
    weights /= weights.sum()
    peer_load = normalized_load[peers]
    averages = {
        "peer": peer_load.mean(axis=1, skipna=False),
        "neighbor": peer_load.mul(weights).sum(axis=1, min_count=len(peers)),
    }

    data = data.copy()
    for prefix, average in averages.items():
        lag24 = data.ds.map(average.shift(24))
        lag168 = data.ds.map(average.shift(168))
        data[f"{prefix}_lag24"] = lag24
        data[f"{prefix}_lag168"] = lag168
        data[f"{prefix}_change"] = lag24 - lag168
    return data


def predict_days(data, features, start, stop):
    """Переобучает модель в начале месяца и прогнозирует сутки по доступной истории."""
    first = data.ds.searchsorted(start)
    end = data.ds.searchsorted(stop)
    dates = data.ds.iloc[first:end:24]
    future_columns = ID_TIME + features
    train_columns = TARGET + features
    forecasts = []
    for _, month in dates.groupby(dates.dt.to_period("M")):
        offsets = data.ds.searchsorted(month)
        model = build_forecaster("extended_400")
        model.models["forecast"].set_params(n_jobs=1)
        model.fit(data.iloc[:offsets[0]][train_columns], static_features=[])

        # Каждый суточный прогноз получает собственную историю до полуночи.
        # MLForecast вычисляет эти независимые прогнозы одним вызовом.
        histories = []
        horizons = []
        history_hours = model.ts.keep_last_n
        for offset, date in zip(offsets, month):
            identifier = date.date().isoformat()
            history_start = max(0, offset - history_hours) if history_hours else 0
            history = data.iloc[history_start:offset][train_columns]
            horizon = data.iloc[offset:offset + 24][future_columns]
            histories.append(history.assign(unique_id=identifier))
            horizons.append(horizon.assign(unique_id=identifier))

        predicted = model.predict(
            24,
            new_df=pd.concat(histories, ignore_index=True),
            X_df=pd.concat(horizons, ignore_index=True),
        )
        forecasts.append(predicted.set_index("ds").forecast)

    return pd.concat(forecasts)


def compare_region(data, region, start, stop):
    output = data.loc[data.ds.ge(start) & data.ds.lt(stop), ["ds", "observed"]].copy()
    output = output.rename(columns={"observed": "y"})
    output.insert(0, "region", region)
    for mode, features in FEATURES.items():
        prediction = output.ds.map(predict_days(data, features, start, stop))
        if not np.isfinite(prediction).all():
            raise ValueError(f"Получен неполный прогноз для региона «{region}» ({mode})")
        output[f"{mode}_y_pred"] = prediction
        print(f"Прогноз готов: {region} ({mode})", flush=True)
    return output


def summarize(predictions):
    rows = []
    for region, frame in predictions.groupby("region", sort=False):
        row = {"region": region, "n": frame.y.notna().sum()}
        for mode in FEATURES:
            scored = frame.rename(columns={f"{mode}_y_pred": "y_pred"})
            errors = metrics(scored.y, scored.y_pred)
            peaks = peak_results(scored)
            row["complete_days"] = len(peaks)
            for name in ("MAE", "RMSE", "WAPE_pct", "MAPE_pct"):
                row[f"{mode}_{name}"] = errors[name]
            row[f"{mode}_top3_hit_pct"] = peaks.hits.mean() / 3 * 100
        row["delta_WAPE_pp"] = row["spatial_WAPE_pct"] - row["local_WAPE_pct"]
        row["delta_top3_pp"] = row["spatial_top3_hit_pct"] - row["local_top3_hit_pct"]
        row["delta_weighted_WAPE_pp"] = row["spatial_WAPE_pct"] - row["interregional_WAPE_pct"]
        row["delta_weighted_top3_pp"] = row["spatial_top3_hit_pct"] - row["interregional_top3_hit_pct"]
        rows.append(row)

    summary = pd.DataFrame(rows)
    average = summary.drop(columns="region").mean().to_dict()
    average["region"] = "Среднее по регионам"
    average["n"] = summary.n.sum()
    average["complete_days"] = summary.complete_days.sum()
    return pd.DataFrame([*rows, average])


def main():
    series = {
        name: load_data(DATASETS / f"electricity_{code}.csv", region=name)
        for name, (code, _, _) in REGIONS.items()
    }
    last_actual = min(data.loc[data.observed.notna(), "ds"].max() for data in series.values())
    stop = (last_actual + pd.Timedelta(hours=1)).normalize()
    if stop <= FORECAST_START:
        raise ValueError("Нет общего периода за 2026 год для сравнения")

    loaded = pd.DataFrame({name: data.set_index("ds").y for name, data in series.items()})
    normalized = loaded.div(loaded.loc[loaded.index < FORECAST_START].mean())
    years = range(TRAIN_START.year, stop.year + 1)
    print(f"Общий период: {FORECAST_START.date()} — {(stop - pd.Timedelta(days=1)).date()}", flush=True)
    with ProcessPoolExecutor(max_workers=2, mp_context=get_context("spawn")) as pool:
        jobs = []
        for name, data in series.items():
            data = add_calendar(data, years)
            data = add_peer_features(data, name, normalized)
            jobs.append(pool.submit(compare_region, data, name, FORECAST_START, stop))
        results = [job.result() for job in jobs]

    output = pd.concat(results, ignore_index=True)
    summary = summarize(output)
    output.to_csv(ROOT / "forecast_2026.csv", index=False, float_format="%.8f")
    summary.to_csv(ROOT / "comparison.csv", index=False, float_format="%.8f")
    print(summary.round(4).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
