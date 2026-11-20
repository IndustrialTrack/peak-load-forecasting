"""Суточный прогноз для Челябинской области с учётом соседних регионов."""

from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from multiprocessing import get_context
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
LOCAL_ROOT = ROOT.parent / "walk-forward forecasting with an expanding window"
sys.path.insert(0, str(LOCAL_ROOT))

from forecast import (
    DATASETS,
    EXOG,
    FORECAST_START,
    ONE_DAY,
    REGION,
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
FEATURES = [*EXOG, "neighbor_lag24", "neighbor_lag168", "neighbor_change"]
ONE_HOUR = pd.Timedelta(hours=1)


def add_peer_features(data, peer_load):
    peers = peer_load.columns
    training_means = peer_load.loc[peer_load.index < FORECAST_START].mean()
    normalized = peer_load.div(training_means)
    lat, lon = np.radians(REGIONS[REGION][1:])
    peer_lat, peer_lon = np.radians([REGIONS[name][1:] for name in peers]).T
    haversine = np.sin((peer_lat - lat) / 2) ** 2
    haversine += np.cos(lat) * np.cos(peer_lat) * np.sin((peer_lon - lon) / 2) ** 2
    distance = 2 * 6371 * np.arcsin(np.sqrt(haversine))
    weights = pd.Series(1 / distance, index=peers)
    weights /= weights.sum()
    weighted_load = normalized.mul(weights).sum(axis=1, min_count=len(peers))

    data = data.copy()
    data["neighbor_lag24"] = data.ds.map(weighted_load.shift(24))
    data["neighbor_lag168"] = data.ds.map(weighted_load.shift(168))
    data["neighbor_change"] = data.neighbor_lag24 - data.neighbor_lag168
    return data


def predict_days(data, days):
    """Переобучает модель перед каждым суточным прогнозом на всей доступной истории."""
    future_columns = ID_TIME + FEATURES
    train_columns = TARGET + FEATURES
    forecasts = []
    for day in days:
        offset = data.ds.searchsorted(day)
        model = build_forecaster("extended_400")
        model.models["forecast"].set_params(n_jobs=1)
        model.fit(data.iloc[:offset][train_columns], static_features=[])
        predicted = model.predict(24, X_df=data.iloc[offset:offset + 24][future_columns])
        forecasts.append(predicted.set_index("ds").forecast)

    print(f"Прогноз готов за {days[0].date()} — {days[-1].date()}", flush=True)
    return pd.concat(forecasts)


def summarize(predictions):
    rows = []
    for mode, description in (("local", "Задача 1, локальный"), ("spatial", "Задача 2, пространственный")):
        scored = predictions.rename(columns={f"{mode}_y_pred": "y_pred"})
        peaks = peak_results(scored)
        rows.append({
            "model": description,
            "region": REGION,
            "start": scored.ds.min().date(),
            "end": scored.ds.max().date(),
            **metrics(scored.y, scored.y_pred),
            "complete_days": len(peaks),
            "peak_MAE": peaks.MAE.mean(),
            "peak_top3_hit_pct": peaks.hits.mean() / 3 * 100,
        })
    return pd.DataFrame(rows)


def main():
    local_forecast = pd.read_csv(
        LOCAL_ROOT / "forecast_2026.csv", usecols=["ds", "y_pred"], parse_dates=["ds"],
    ).rename(columns={"y_pred": "local_y_pred"})
    target = load_data()
    peer_series = {
        name: load_data(DATASETS / f"electricity_{code}.csv", region=name)
        for name, (code, _, _) in REGIONS.items()
        if name != REGION
    }
    target_last = target.loc[target.observed.notna(), "ds"].max()
    peer_last = pd.Series({
        name: frame.loc[frame.observed.notna(), "ds"].max()
        for name, frame in peer_series.items()
    }).min(skipna=False)
    if pd.isna(target_last) or pd.isna(peer_last):
        raise ValueError("Для сравнения нужны фактические данные Челябинской области и всех соседних регионов")

    # Для прогноза нужны факты Челябинской области и история соседей за предыдущие сутки.
    stop = min((target_last + ONE_HOUR).normalize(), (peer_last + ONE_HOUR).normalize() + ONE_DAY)
    if stop <= FORECAST_START:
        raise ValueError("Нет общего периода за 2026 год для сравнения")

    output = target.loc[target.ds.ge(FORECAST_START) & target.ds.lt(stop), ["ds", "observed"]]
    output = output.rename(columns={"observed": "y"}).merge(local_forecast, on="ds", how="left", validate="one_to_one")
    if not np.isfinite(output.local_y_pred).all():
        raise ValueError("В файле первой задачи нет локальных прогнозов за весь период сравнения")

    peers = pd.DataFrame({name: frame.set_index("ds").y for name, frame in peer_series.items()})
    data = add_peer_features(add_calendar(target, range(TRAIN_START.year, stop.year + 1)), peers)
    days = pd.date_range(FORECAST_START, stop, freq="D", inclusive="left")
    months = [days[days.to_period("M") == month] for month in days.to_period("M").unique()]
    print(f"Общий период: {FORECAST_START.date()} — {(stop - pd.Timedelta(days=1)).date()}", flush=True)
    with ProcessPoolExecutor(max_workers=2, mp_context=get_context("spawn")) as pool:
        spatial = pd.concat(list(pool.map(predict_days, repeat(data), months)))
    output["spatial_y_pred"] = output.ds.map(spatial)
    if not np.isfinite(output.spatial_y_pred).all():
        raise ValueError("Получен неполный пространственный прогноз")

    output.insert(0, "region", REGION)
    summary = summarize(output)
    output.to_csv(ROOT / "forecast_2026.csv", index=False)
    summary.to_csv(ROOT / "comparison.csv", index=False, float_format="%.8f")
    print(summary.round(4).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
