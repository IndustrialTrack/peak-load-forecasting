"""Multi-region 24-hour load forecasting with a shared 1D CNN.

Run from any directory: python LearnCNN_MultiRegion.py
Each CSV in datasets is one region. Inputs are the recent loads for all
selected regions; the model forecasts the target region only.
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import random
import re
import warnings

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout

warnings.filterwarnings("ignore", category=FutureWarning)
SEED = 42
WINDOW = 168
HORIZON = 24
EPOCHS = 50
BATCH_SIZE = 32
TRAIN_START = pd.Timestamp("2022-01-01")
TEST_START = pd.Timestamp("2026-01-01")
# Edit these file names to change the set of neighboring regions.
REGION_FILES = ["electricity_33.csv", "electricity_57.csv", "electricity_71.csv",
                "electricity_75.csv", "electricity_94.csv"]
TARGET_FILE = "electricity_75.csv"
BASE = Path(__file__).resolve().parents[3]
DATA_DIR = BASE / "datasets"
OUT = Path(__file__).resolve().parent / "results" / "multi_region_2026"
# Параллелятся независимые прогнозные дни. Для одного процесса задайте 1.
PARALLEL_TRAINING = True
MAX_WORKERS = min(2, os.cpu_count() or 1)

random.seed(SEED)
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)


def number(value):
    if pd.isna(value):
        return np.nan
    value = str(value).replace("\xa0", " ").replace(" ", "").replace(",", ".")
    value = re.sub(r"[^0-9.\-]", "", value)
    try:
        return float(value)
    except ValueError:
        return np.nan


def read_region(filename):
    path = DATA_DIR / filename
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = None
    for encoding in ("utf-8-sig", "cp1251"):
        try:
            frame = pd.read_csv(path, encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    if frame is None:
        raise ValueError(f"Cannot decode {path}")
    frame.columns = [str(c).replace("\ufeff", "").strip().lower() for c in frame.columns]
    required = {"date", "hour", "region_name", "actual_consumption"}
    if not required.issubset(frame.columns):
        raise ValueError(f"{filename}: missing columns {sorted(required - set(frame.columns))}")
    hour = pd.to_numeric(frame["hour"].astype(str).str.extract(r"(\d+)", expand=False), errors="coerce")
    stamp = pd.to_datetime(frame["date"], errors="coerce").dt.normalize() + pd.to_timedelta(hour, unit="h")
    values = frame["actual_consumption"].map(number)
    series = pd.Series(values.to_numpy(), index=stamp, name=filename).dropna(how="all")
    series = series[~series.index.isna()].groupby(level=0).last()
    return series


def windows(matrix, end):
    xs, ys = [], []
    for i in range(len(matrix) - WINDOW - HORIZON + 1):
        x = matrix[i:i + WINDOW]
        y = matrix[i + WINDOW:i + WINDOW + HORIZON, end]
        if np.isfinite(x).all() and np.isfinite(y).all():
            xs.append(x)
            ys.append(y)
    return np.asarray(xs, np.float32), np.asarray(ys, np.float32)


def build_model(features):
    model = Sequential([Input((WINDOW, features)),
        Conv1D(64, 3, activation="relu", padding="same"), MaxPooling1D(2),
        Conv1D(32, 3, activation="relu", padding="same"), Flatten(),
        Dense(128, activation="relu"), Dropout(.2), Dense(HORIZON)])
    model.compile(optimizer=tf.keras.optimizers.Adam(.001), loss="mse")
    return model


def forecast_one_day(args):
    """Train using history available before the day and return its forecast."""
    day, values, index_values, target_idx = args
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(SEED + int(pd.Timestamp(day).dayofyear))
    data = pd.DataFrame(values, index=pd.DatetimeIndex(index_values), columns=REGION_FILES)
    day = pd.Timestamp(day)
    history = data.loc[:day - pd.Timedelta(hours=1)]
    actual = data.loc[day:day + pd.Timedelta(hours=HORIZON - 1), TARGET_FILE]
    if len(history) < WINDOW or len(actual) != HORIZON or actual.isna().any():
        return []
    train_values = history.to_numpy(dtype=np.float32)
    means = np.nanmean(train_values, axis=0)
    stds = np.nanstd(train_values, axis=0)
    stds[~np.isfinite(stds) | (stds == 0)] = 1
    scaled = (train_values - means) / stds
    X, y = windows(scaled, target_idx)
    if len(X) < 10:
        return []
    model = build_model(len(REGION_FILES))
    model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=False,
        validation_split=.1, verbose=0,
        callbacks=[EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
                   ReduceLROnPlateau(monitor="val_loss", patience=4, factor=.5, min_lr=1e-5)])
    latest = train_values[-WINDOW:]
    pred = model.predict(((latest - means) / stds)[None, ...], verbose=0)[0] * stds[target_idx] + means[target_idx]
    del model
    tf.keras.backend.clear_session()
    return [{"date": day.strftime("%Y-%m-%d"), "hour": hour,
             "target_region": TARGET_FILE, "actual_consumption": float(a),
             "predicted_consumption": float(p)}
            for hour, (a, p) in enumerate(zip(actual.to_numpy(), pred))]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if TARGET_FILE not in REGION_FILES:
        raise ValueError("TARGET_FILE must be included in REGION_FILES")
    regions = {name: read_region(name) for name in REGION_FILES}
    target_idx = REGION_FILES.index(TARGET_FILE)
    start = max(s.index.min() for s in regions.values())
    end = min(s.index.max() for s in regions.values())
    index = pd.date_range(start, end, freq="h")
    data = pd.concat([s.reindex(index) for s in regions.values()], axis=1)
    data.columns = REGION_FILES
    data = data.loc[TRAIN_START:]
    dates = pd.date_range(TEST_START.normalize(), data.index.max().normalize(), freq="D")
    payloads = [(day, data.to_numpy(dtype=np.float32), data.index.to_numpy(), target_idx) for day in dates]
    rows = []
    if PARALLEL_TRAINING and MAX_WORKERS > 1:
        print(f"Parallel daily training: {MAX_WORKERS} workers")
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = [pool.submit(forecast_one_day, item) for item in payloads]
            for future in as_completed(futures):
                rows.extend(future.result())
    else:
        print("Sequential daily training")
        for item in payloads:
            rows.extend(forecast_one_day(item))
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "predictions.csv", index=False, encoding="utf-8-sig")
    if result.empty:
        print("No complete evaluation days; predictions file is empty.")
        return
    result = result.sort_values(["date", "hour"])
    err = result.actual_consumption - result.predicted_consumption
    daily = result.assign(sq=err ** 2, ape=np.abs(err / result.actual_consumption.replace(0, np.nan)))        .groupby("date").agg(MSE=("sq", "mean"), MAPE=("ape", "mean"))
    daily["RMSE"] = np.sqrt(daily.MSE)
    daily["MAPE"] *= 100
    daily.to_csv(OUT / "daily_metrics.csv", encoding="utf-8-sig")
    print(f"Target: {TARGET_FILE}; context regions: {len(REGION_FILES)}")
    print(f"Forecast days: {len(daily)}; results: {OUT}")
    print(daily[["MSE", "RMSE", "MAPE"]].mean().to_string())


if __name__ == "__main__":
    main()
