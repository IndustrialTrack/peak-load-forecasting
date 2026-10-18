"""Rolling day-ahead regional electricity-load experiments.

Files expected:
  project_root/datasets/electricity_<region_id>.csv
  project_root/CNN(1D)/research.py

Run:
  python "CNN(1D)/research.py"
  python "CNN(1D)/research.py" --epochs 40 --train-stride 1

Optional CNN(1D)/region_types.csv columns: region_id,region_type.

The test protocol forecasts 24 hours from midnight using the previous 168
hours of observed load. It is rolling day-ahead evaluation, not a one-shot
forecast for a whole year without new observations.
"""
from __future__ import annotations

import argparse
import os
import re
import warnings

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input, Conv1D, Flatten, Dense, Dropout, Concatenate

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = "C:/Users/savva/Desktop/ВУЗ/Репозиторий Проекта/peak-load-forecasting/datasets"
OUT_DIR = os.path.join(SCRIPT_DIR, "results", "research")
HISTORY, HORIZON, SEED = 168, 24, 42
CALENDAR_FEATURES = 4
np.random.seed(SEED)
tf.random.set_seed(SEED)


def number(value):
    """Convert '4 076 МВт*ч' or '2 180,5 руб.' to a number."""
    if pd.isna(value):
        return np.nan
    s = str(value).replace("\xa0", " ").replace(",", ".")
    s = re.sub(r"[^0-9.\-]", "", s.replace(" ", ""))
    try:
        return float(s)
    except ValueError:
        return np.nan


def read_csv(path, **kwargs):
    try:
        return pd.read_csv(path, encoding="utf-8-sig", **kwargs)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="cp1251", **kwargs)


def normalize_columns(df):
    df = df.copy()
    df.columns = df.columns.astype(str).str.replace("\ufeff", "", regex=False).str.strip().str.lower()
    return df


def load_region(path):
    df = normalize_columns(read_csv(path))
    required = {"date", "hour", "actual_consumption"}
    absent = required - set(df.columns)
    if absent:
        raise ValueError(f"Missing columns in {path}: {', '.join(sorted(absent))}")

    hour = df["hour"].astype(str).str.extract(r"(\d+)")[0].str.zfill(2)
    df["datetime"] = pd.to_datetime(df["date"].astype(str) + " " + hour, errors="coerce")
    df["y"] = df["actual_consumption"].map(number)
    df = (df.dropna(subset=["datetime", "y"])
            .drop_duplicates("datetime", keep="last")
            .set_index("datetime").sort_index())
    if df.empty:
        raise ValueError(f"No valid rows in {path}")

    idx = pd.date_range(df.index.min(), df.index.max(), freq="h")
    series = df["y"].reindex(idx).astype(float)
    if series.isna().any():
        series = series.interpolate(method="time", limit=2, limit_area="inside")
    if series.isna().any():
        bad = series.index[series.isna()]
        raise ValueError(f"{path}: {len(bad)} unresolved hourly gaps; first: {list(bad[:5])}")
    series.name = "actual_consumption"
    return series


def region_name(path):
    try:
        df = normalize_columns(read_csv(path, nrows=5))
        if "region_name" in df.columns and df["region_name"].notna().any():
            return str(df["region_name"].dropna().iloc[0])
    except (OSError, ValueError, IndexError):
        pass
    return os.path.basename(path).replace("electricity_", "").replace(".csv", "")


def calendar_features(datetimes):
    idx = pd.DatetimeIndex(datetimes)
    hour = idx.hour.to_numpy(dtype=np.float32)
    dow = idx.dayofweek.to_numpy(dtype=np.float32)
    return np.column_stack([
        np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
        np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7),
    ]).astype(np.float32)


def build_features(series, starts, scaler=None):
    """Both ML models get 168 load lags and the same known calendar features."""
    raw = series.to_numpy(dtype=np.float32)
    idx = series.index
    hist, future_cal, targets = [], [], []
    for i in starts:
        hist.append(raw[i-HISTORY:i])
        future_cal.append(calendar_features(idx[i:i+HORIZON]))
        targets.append(raw[i:i+HORIZON])

    hist = np.asarray(hist, dtype=np.float32)
    future_cal = np.asarray(future_cal, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)
    if scaler is not None:
        hist = scaler.transform(hist.reshape(-1, 1)).reshape(-1, HISTORY).astype(np.float32)
        targets_scaled = scaler.transform(targets.reshape(-1, 1)).reshape(-1, HORIZON).astype(np.float32)
    else:
        targets_scaled = targets

    rf_x = np.concatenate([hist, future_cal.reshape(len(starts), -1)], axis=1)
    return hist[..., None], future_cal, targets_scaled, rf_x.astype(np.float32), targets


def make_cnn():
    history_in = Input(shape=(HISTORY, 1), name="load_history")
    calendar_in = Input(shape=(HORIZON, CALENDAR_FEATURES), name="future_calendar")
    x = Conv1D(32, 5, padding="same", activation="relu")(history_in)
    x = Conv1D(64, 5, padding="same", activation="relu")(x)
    x = Conv1D(64, 3, padding="same", activation="relu")(x)
    x = Flatten()(x)
    c = Flatten()(calendar_in)
    x = Concatenate()([x, c])
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.2)(x)
    out = Dense(HORIZON, name="next_24_hours")(x)
    model = Model([history_in, calendar_in], out)
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def make_starts(stop, stride):
    """Forecast origins with target hours strictly before position stop."""
    return np.arange(HISTORY, stop - HORIZON + 1, stride, dtype=int)


def test_origins(series, approx_start):
    # Each test block begins at midnight and covers a non-overlapping full day.
    return np.asarray([
        i for i in range(max(HISTORY, approx_start), len(series) - HORIZON + 1)
        if series.index[i].hour == 0 and series.index[i].minute == 0
    ], dtype=int)


def top3(values):
    return np.sort(np.argsort(values)[-3:])


def calculate_scores(actual, pred):
    errors = np.abs(np.asarray(actual, dtype=float) - np.asarray(pred, dtype=float))
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    daily_hits = []
    for a, p in zip(actual, pred):
        daily_hits.append(len(set(top3(a)) & set(top3(p))) / 3.0)
    return {
        "MAE": float(errors.mean()),
        "RMSE": float(np.sqrt(np.mean((actual - pred) ** 2))),
        "WAPE_pct": float(errors.sum() / max(np.abs(actual).sum(), 1e-12) * 100),
        "Top3_hit_rate": float(np.mean(daily_hits)),
    }


def train_cnn(series, test_start, epochs, batch_size, stride):
    """Select epochs on a chronological validation segment; then refit pre-test."""
    val_cut = max(int(test_start * 0.9), HISTORY + HORIZON + 1)
    train_i = make_starts(val_cut, stride)
    val_i = np.arange(max(HISTORY, val_cut), test_start - HORIZON + 1, stride, dtype=int)

    if len(train_i) < 10 or len(val_i) < 5:
        warnings.warn("Small train/validation set; using up to 10 epochs.")
        best_epochs = min(epochs, 10)
    else:
        val_scaler = StandardScaler().fit(series.to_numpy()[:val_cut].reshape(-1, 1))
        xh, xc, yt, _, _ = build_features(series, train_i, val_scaler)
        vh, vc, vy, _, _ = build_features(series, val_i, val_scaler)
        probe = make_cnn()
        hist = probe.fit(
            [xh, xc], yt, validation_data=([vh, vc], vy),
            epochs=epochs, batch_size=batch_size, shuffle=False, verbose=0,
            callbacks=[EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)],
        )
        losses = hist.history.get("val_loss", [])
        best_epochs = int(np.argmin(losses) + 1) if losses else min(epochs, 10)
        del probe

    # Final scaling uses pre-test data only. The test period is never used to fit it.
    scaler = StandardScaler().fit(series.to_numpy()[:test_start].reshape(-1, 1))
    final_i = make_starts(test_start, stride)
    xh, xc, y, _, _ = build_features(series, final_i, scaler)
    model = make_cnn()
    model.fit([xh, xc], y, epochs=best_epochs, batch_size=batch_size, shuffle=False, verbose=0)
    return model, scaler, best_epochs, len(final_i)


def analyze_region(path, epochs, batch_size, train_stride, rf_trees):
    series = load_region(path)
    n = len(series)
    if n < HISTORY + 10 * HORIZON:
        raise ValueError(f"Not enough data in {path}: only {n} hourly rows")

    test_i = test_origins(series, int(n * 0.8))
    if len(test_i) < 3:
        raise ValueError(f"Fewer than 3 complete test days in {path}")
    test_start = int(test_i[0])

    train_i = make_starts(test_start, train_stride)
    if len(train_i) < 10:
        raise ValueError(f"Too few training windows in {path}")

    _, _, _, x_test_rf, y_test = build_features(series, test_i)
    raw = series.to_numpy(dtype=np.float32)

    # Baseline 1: repeat the same hours from the previous week.
    weekly = np.stack([raw[i-HISTORY:i-HISTORY+HORIZON] for i in test_i])

    # Baseline 2: average by weekday and hour, computed only before test start.
    prof = pd.DataFrame({"y": raw[:test_start]}, index=series.index[:test_start])
    prof["dow"], prof["hour"] = prof.index.dayofweek, prof.index.hour
    means = prof.groupby(["dow", "hour"])["y"].mean()
    fallback = float(np.mean(raw[:test_start]))
    profile = np.stack([
        np.asarray([means.get((dt.dayofweek, dt.hour), fallback)
                    for dt in series.index[i:i+HORIZON]], dtype=np.float32)
        for i in test_i
    ])

    # Random Forest sees the same past-load and calendar information as the CNN.
    _, _, y_train, x_train_rf, _ = build_features(series, train_i)
    forest = RandomForestRegressor(
        n_estimators=rf_trees, min_samples_leaf=3, max_features=0.7,
        n_jobs=-1, random_state=SEED,
    )
    forest.fit(x_train_rf, y_train)
    rf_pred = forest.predict(x_test_rf).astype(np.float32)

    cnn, scaler, best_epochs, n_train_windows = train_cnn(
        series, test_start, epochs, batch_size, train_stride
    )
    xh_test, xc_test, _, _, _ = build_features(series, test_i, scaler)
    pred_z = cnn.predict([xh_test, xc_test], verbose=0)
    cnn_pred = scaler.inverse_transform(pred_z.reshape(-1, 1)).reshape(-1, HORIZON)

    methods = {
        "weekly_naive": weekly,
        "weekday_hour_mean": profile,
        "random_forest": rf_pred,
        "cnn_1d": cnn_pred,
    }
    rid = os.path.basename(path).replace("electricity_", "").replace(".csv", "")
    rname = region_name(path)

    score_rows = []
    for method, pred in methods.items():
        score_rows.append({
            "region_id": rid, "region_name": rname, "method": method,
            "best_cnn_epochs": best_epochs if method == "cnn_1d" else np.nan,
            **calculate_scores(y_test, pred),
        })

    # Descriptive time-series properties, calculated on pre-test observations only.
    train_series = pd.Series(raw[:test_start], index=series.index[:test_start])
    mean = float(train_series.mean())
    chars = {
        "region_id": rid, "region_name": rname,
        "autocorr_24": float(train_series.autocorr(24)),
        "autocorr_168": float(train_series.autocorr(168)),
        "coefficient_variation": float(train_series.std() / mean) if mean else np.nan,
        "max_to_mean": float(train_series.max() / mean) if mean else np.nan,
        "mean_abs_hourly_change": float(train_series.diff().abs().mean()),
        "train_start": str(series.index[0]),
        "test_start": str(series.index[test_start]),
        "test_days": int(len(test_i)),
        "training_windows": int(n_train_windows),
    }

    daily_rows = []
    for method, pred in methods.items():
        for j, i in enumerate(test_i):
            a_top, p_top = top3(y_test[j]), top3(pred[j])
            daily_rows.append({
                "region_id": rid, "region_name": rname,
                "date": str(series.index[i].date()), "method": method,
                "actual_top3_hours": ";".join(f"{h:02d}:00" for h in a_top),
                "predicted_top3_hours": ";".join(f"{h:02d}:00" for h in p_top),
                "top3_hits": int(len(set(a_top) & set(p_top))),
                "top3_hit_rate": float(len(set(a_top) & set(p_top)) / 3),
            })
    return score_rows, chars, daily_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--train-stride", type=int, default=1,
                        help="Training-window step in hours (1 = every hour)")
    parser.add_argument("--rf-trees", type=int, default=60)
    parser.add_argument("--region-types", default=os.path.join(SCRIPT_DIR, "region_types.csv"))
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.train_stride, args.rf_trees) < 1:
        parser.error("epochs, batch-size, train-stride and rf-trees must be positive")

    os.makedirs(OUT_DIR, exist_ok=True)
    if not os.path.isdir(DATA_DIR):
        raise FileNotFoundError(f"Dataset directory not found: {DATA_DIR}")

    paths = sorted(
        os.path.join(DATA_DIR, f)
        for f in os.listdir(DATA_DIR)
        if re.fullmatch(r"electricity_\d+\.csv", f)
    )

    if not paths:
        raise FileNotFoundError(f"No datasets/electricity_<region_id>.csv files found in {DATA_DIR}")

    all_scores, all_chars, all_days = [], [], []
    for path in paths:
        print(f"Evaluating {os.path.basename(path)} ...", flush=True)
        try:
            scores, chars, days = analyze_region(
                path, args.epochs, args.batch_size, args.train_stride, args.rf_trees
            )
        except Exception as exc:
            raise RuntimeError(f"Failed while evaluating {path}: {exc}") from exc
        all_scores.extend(scores)
        all_chars.append(chars)
        all_days.extend(days)

    score_df = pd.DataFrame(all_scores)
    char_df = pd.DataFrame(all_chars)
    daily_df = pd.DataFrame(all_days)

    if os.path.exists(args.region_types):
        types = normalize_columns(read_csv(args.region_types))
        if not {"region_id", "region_type"}.issubset(types.columns):
            raise ValueError("region_types.csv must contain region_id,region_type")
        types["region_id"] = types["region_id"].astype(str)
        score_df["region_id"] = score_df["region_id"].astype(str)
        char_df["region_id"] = char_df["region_id"].astype(str)
        score_df = score_df.merge(types[["region_id", "region_type"]], on="region_id",
                                  how="left", validate="many_to_one")
        char_df = char_df.merge(types[["region_id", "region_type"]], on="region_id",
                                how="left", validate="many_to_one")
        summary = (score_df.dropna(subset=["region_type"])
                   .groupby(["region_type", "method"], as_index=False)
                   .agg(regions=("region_id", "nunique"), MAE_mean=("MAE", "mean"),
                        WAPE_pct_mean=("WAPE_pct", "mean"),
                        Top3_hit_rate_mean=("Top3_hit_rate", "mean")))
        summary.to_csv(os.path.join(OUT_DIR, "region_type_summary.csv"),
                       index=False, encoding="utf-8-sig")

    score_df.to_csv(os.path.join(OUT_DIR, "model_comparison.csv"),
                    index=False, encoding="utf-8-sig")
    char_df.to_csv(os.path.join(OUT_DIR, "series_characteristics.csv"),
                   index=False, encoding="utf-8-sig")
    daily_df.to_csv(os.path.join(OUT_DIR, "daily_top3.csv"),
                    index=False, encoding="utf-8-sig")

    # Exploratory Pearson correlations across regions; not causal evidence.
    characteristics = ["autocorr_24", "autocorr_168", "coefficient_variation",
                       "max_to_mean", "mean_abs_hourly_change"]
    associations = []
    for method, group in score_df.groupby("method"):
        joined = group.merge(char_df[["region_id", *characteristics]],
                             on="region_id", validate="one_to_one")
        for feature in characteristics:
            for metric in ("MAE", "WAPE_pct", "Top3_hit_rate"):
                pair = joined[[feature, metric]].dropna()
                r = (float(pair[feature].corr(pair[metric]))
                     if len(pair) >= 2 and pair[feature].nunique() > 1
                     and pair[metric].nunique() > 1 else np.nan)
                associations.append({"method": method, "characteristic": feature,
                                     "metric": metric, "pearson_r": r,
                                     "regions": int(len(pair))})
    pd.DataFrame(associations).to_csv(
        os.path.join(OUT_DIR, "characteristic_quality_correlations.csv"),
        index=False, encoding="utf-8-sig"
    )

    print(f"\\nSaved results to: {OUT_DIR}")
    print("\\nModel comparison:")
    print(score_df.to_string(index=False))
    print("\\nNote: test forecasts use the preceding 168 actual hours.")
    print("This is rolling day-ahead evaluation, not a one-shot year-ahead forecast.")


if __name__ == "__main__":
    main()
