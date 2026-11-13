import os
import re
import pickle
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf

###
#commands for start predict : py '.\CNN(1D)\cmd\LearnCNN\learnOneDay.py' --date 2026-09-27
###

# ============================================================
# PATHS AND PARAMETERS
# ============================================================

# This script is expected at:
# CNN(1D)/cmd/LearnCNN/LearnCNN.py
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CNN_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
PROJECT_DIR = os.path.abspath(os.path.join(CNN_DIR, ".."))

DATA_PATH = os.path.join(PROJECT_DIR, "datasets", "electricity_75.csv")
MODEL_DIR = os.path.join(SCRIPT_DIR, "models")
RESULTS_DIR = os.path.join(CNN_DIR, "resultOneDayLearn")

MODEL_PATH = os.path.join(MODEL_DIR, "cnn_168.keras")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

WINDOW_SIZE = 168
FORECAST_HORIZON = 24


# ============================================================
# DATA LOADING
# ============================================================

def parse_numeric(value):
    """Convert values such as '4 051 МВт*ч' to a number."""
    if pd.isna(value):
        return np.nan

    value = str(value).strip()
    if not value:
        return np.nan

    value = value.replace("\xa0", " ")
    value = value.replace(",", ".")
    value = value.replace("МВт*ч", "")
    value = value.replace("МВт·ч", "")
    value = value.replace("руб./МВт*ч", "")
    value = value.replace("руб./МВт·ч", "")
    value = value.replace(" ", "")
    value = re.sub(r"[^0-9.\-]", "", value)

    try:
        return float(value)
    except ValueError:
        return np.nan


def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV file not found: {path}")

    try:
        df = pd.read_csv(
            path,
            sep=None,
            engine="python",
            encoding="utf-8-sig"
        )
    except UnicodeDecodeError:
        df = pd.read_csv(
            path,
            sep=None,
            engine="python",
            encoding="cp1251"
        )

    df.columns = [
        str(column).replace("\ufeff", "").strip().lower()
        for column in df.columns
    ]

    required = ["date", "hour", "actual_consumption"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"Columns found: {list(df.columns)}"
        )

    df["hour"] = (
        df["hour"].astype(str)
        .str.extract(r"(\d+)", expand=False)
        .astype(int)
    )

    df["datetime"] = pd.to_datetime(
        df["date"].astype(str) + " " + df["hour"].astype(str).str.zfill(2),
        format="%Y-%m-%d %H",
        errors="coerce"
    )

    df["actual_consumption"] = df["actual_consumption"].apply(parse_numeric)
    df = df.dropna(subset=["datetime", "actual_consumption"])
    df = df.sort_values("datetime").drop_duplicates("datetime", keep="first")

    result = df[["datetime", "actual_consumption"]].set_index("datetime")

    # Ensure a continuous hourly index. Only short internal gaps are interpolated.
    full_index = pd.date_range(result.index.min(), result.index.max(), freq="h")
    if len(full_index.difference(result.index)) > 0:
        result = result.reindex(full_index)
        result["actual_consumption"] = result["actual_consumption"].interpolate(
            method="time",
            limit=2,
            limit_area="inside"
        )

    if result["actual_consumption"].isna().any():
        missing_times = result.index[result["actual_consumption"].isna()]
        raise ValueError(
            "There are missing consumption values that cannot be safely "
            f"interpolated. First missing timestamps: {list(missing_times[:5])}"
        )

    print(f"Data loaded: {result.index.min()} — {result.index.max()}")
    return result


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

def load_saved_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}\n"
            "Train the model first or correct MODEL_PATH."
        )

    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(
            f"Scaler not found: {SCALER_PATH}\n"
            "Make sure scaler.pkl is from the same training run as the model."
        )

    model = tf.keras.models.load_model(MODEL_PATH)
    with open(SCALER_PATH, "rb") as file:
        scaler = pickle.load(file)

    return model, scaler


# ============================================================
# FORECAST ONE SPECIFIC DAY
# ============================================================

def predict_day(model, scaler, df, target_date):
    target_start = pd.Timestamp(target_date).normalize()
    history_start = target_start - pd.Timedelta(hours=WINDOW_SIZE)
    history_end = target_start - pd.Timedelta(hours=1)

    history = df.loc[history_start:history_end, "actual_consumption"]

    if len(history) != WINDOW_SIZE:
        raise ValueError(
            f"Need {WINDOW_SIZE} hourly historical values before {target_date}, "
            f"but found {len(history)}. Check the CSV and missing hours."
        )

    scaled_history = scaler.transform(
        history.to_numpy().reshape(-1, 1)
    ).flatten()

    X = scaled_history.reshape(1, WINDOW_SIZE, 1)
    prediction_scaled = model.predict(X, verbose=0)[0]
    prediction = scaler.inverse_transform(
        prediction_scaled.reshape(-1, 1)
    ).flatten()

    forecast_index = pd.date_range(
        start=target_start,
        periods=FORECAST_HORIZON,
        freq="h"
    )

    return pd.DataFrame({
        "datetime": forecast_index,
        "predicted_consumption": prediction
    })


# ============================================================
# TOP 3 PREDICTED PEAK HOURS
# ============================================================

def print_top_3_peaks(forecast):
    peaks = forecast.nlargest(3, "predicted_consumption").sort_values("datetime")

    print("\nTOP 3 PREDICTED PEAK HOURS")
    print("-" * 55)
    for _, row in peaks.iterrows():
        print(
            f"{row['datetime']:%Y-%m-%d %H:%M} — "
            f"{row['predicted_consumption']:.2f} MWh"
        )


# ============================================================
# METRICS FOR A SPECIFIC DAY
# ============================================================

def save_daily_metrics(forecast, df, target_date):
    """
    Compare the forecast with actual values for the same day.

    Top-3 Hit rate = number of shared actual/predicted top-3 hours / 3.
    F1-macro = mean F1 for peak (top 3) and non-peak hours.
    """
    target_start = pd.Timestamp(target_date).normalize()
    actual_index = pd.date_range(target_start, periods=24, freq="h")

    actual_day = df["actual_consumption"].reindex(actual_index)
    predicted = forecast.set_index("datetime")[
        "predicted_consumption"
    ].reindex(actual_index)

    metrics_path = os.path.join(
        RESULTS_DIR, f"metrics_{target_start:%Y-%m-%d}.txt"
    )

    valid = (
        actual_day.notna()
        & predicted.notna()
        & np.isfinite(actual_day.to_numpy(dtype=float))
        & np.isfinite(predicted.to_numpy(dtype=float))
    )

    lines = [
        "FINAL METRICS\n",
        f"Date: {target_start:%Y-%m-%d}\n",
        f"Actual hours available: {int(valid.sum())}/24\n",
    ]

    # Do not report full-day metrics from an incomplete actual day.
    if int(valid.sum()) != 24:
        lines.extend([
            "\nMetrics not calculated: 24 actual hourly values are required.\n",
            "Top-3 Hit rate and F1-macro also require all 24 actual values.\n",
        ])
        with open(metrics_path, "w", encoding="utf-8") as file:
            file.writelines(lines)
        print("\n" + "".join(lines))
        print(f"Metrics status saved to: {metrics_path}")
        return

    actual_values = actual_day.to_numpy(dtype=float)
    predicted_values = predicted.to_numpy(dtype=float)

    mae = float(np.mean(np.abs(actual_values - predicted_values)))
    rmse = float(np.sqrt(np.mean((actual_values - predicted_values) ** 2)))

    nonzero = actual_values != 0
    mape = (
        float(np.mean(np.abs(
            (actual_values[nonzero] - predicted_values[nonzero])
            / actual_values[nonzero]
        )) * 100)
        if nonzero.any() else float("nan")
    )

    # The three hours with highest actual/predicted consumption.
    actual_top3 = set(np.argsort(actual_values)[-3:].tolist())
    predicted_top3 = set(np.argsort(predicted_values)[-3:].tolist())
    hits = len(actual_top3 & predicted_top3)
    top3_hit_rate = hits / 3.0

    # Binary labels: 1 = one of the top-3 peak hours, 0 = other hours.
    y_true = np.zeros(24, dtype=int)
    y_pred = np.zeros(24, dtype=int)
    y_true[list(actual_top3)] = 1
    y_pred[list(predicted_top3)] = 1

    def f1_for_class(true_labels, predicted_labels, label):
        tp = int(np.sum((true_labels == label) & (predicted_labels == label)))
        fp = int(np.sum((true_labels != label) & (predicted_labels == label)))
        fn = int(np.sum((true_labels == label) & (predicted_labels != label)))
        denominator = 2 * tp + fp + fn
        return 0.0 if denominator == 0 else 2 * tp / denominator

    f1_peak = f1_for_class(y_true, y_pred, 1)
    f1_nonpeak = f1_for_class(y_true, y_pred, 0)
    f1_macro = (f1_peak + f1_nonpeak) / 2.0

    lines.extend([
        f"MAE:  {mae:.2f} MWh\n",
        f"RMSE: {rmse:.2f} MWh\n",
        f"MAPE: {mape:.2f}%\n",
        f"Top-3 Hit rate: {top3_hit_rate:.2%} ({hits}/3 actual peak hours found)\n",
        f"F1-macro: {f1_macro:.4f}\n",
        "\nDefinitions:\n",
        "Top-3 Hit rate = overlap between actual and predicted top-3 hours / 3.\n",
        "F1-macro = mean F1 for peak hours (top 3) and non-peak hours.\n",
    ])

    with open(metrics_path, "w", encoding="utf-8") as file:
        file.writelines(lines)

    print("\n" + "".join(lines))
    print(f"Daily metrics saved to: {metrics_path}")


# ============================================================
# PLOT ACTUAL AND PREDICTED VALUES
# ============================================================

def plot_day(forecast, df, target_date):
    target_start = pd.Timestamp(target_date).normalize()
    target_end = target_start + pd.Timedelta(hours=23)

    actual_day = df.loc[
        target_start:target_end,
        "actual_consumption"
    ]

    hours = np.arange(24)
    forecast_hours = forecast["datetime"].dt.hour.to_numpy()

    # Keep each chart's x-axis independent and show the full day on both.
    fig, axes = plt.subplots(2, 1, figsize=(14, 9))

    # First graph: actual consumption
    if len(actual_day) > 0:
        axes[0].plot(
            actual_day.index.hour,
            actual_day.values,
            marker="o"
        )
    else:
        axes[0].text(
            0.5, 0.5,
            "No actual data for this date in the CSV",
            ha="center", va="center",
            transform=axes[0].transAxes
        )

    axes[0].set_title(f"Актуальное потребление на: {target_start:%Y-%m-%d}")
    axes[0].set_ylabel("Актуальное потребление, МВт*ч")
    axes[0].set_xticks(hours)
    axes[0].set_xlim(-0.5, 23.5)
    axes[0].grid(True)

    # Second graph: CNN prediction
    axes[1].plot(
        forecast_hours,
        forecast["predicted_consumption"].to_numpy(),
        marker="o"
    )
    axes[1].set_title(f"CNN 1D Прогнозирование на: {target_start:%Y-%m-%d}")
    axes[1].set_xlabel("Часы")
    axes[1].set_ylabel("прогнозируемое потребление, МВт*ч")
    axes[1].set_xticks(hours)
    axes[1].set_xlim(-0.5, 23.5)
    axes[1].grid(True)

    values = np.concatenate((
        actual_day.to_numpy(dtype=float),
        forecast["predicted_consumption"].to_numpy(dtype=float)
    ))
    values = values[np.isfinite(values)]
    if len(values):
        y_min, y_max = values.min(), values.max()
        padding = (y_max - y_min) * 0.05 or max(abs(y_min) * 0.05, 1)
        for axis in axes:
            axis.set_ylim(y_min - padding, y_max + padding)

    fig.tight_layout()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    output_path = os.path.join(
        RESULTS_DIR,
        f"forecast_{target_start:%Y-%m-%d}.png"
    )
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    print(f"\nGraph saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Forecast electricity consumption for one selected day."
    )
    parser.add_argument(
        "--date",
        required=True,
        help="Forecast date in YYYY-MM-DD format, e.g. 2026-09-27"
    )
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    df = load_data(DATA_PATH)
    model, scaler = load_saved_model()

    forecast = predict_day(model, scaler, df, args.date)

    print(f"\nFORECAST FOR {args.date}")
    print("-" * 55)
    print(
        forecast.assign(hour=forecast["datetime"].dt.strftime("%H:%M"))
        [["hour", "predicted_consumption"]]
        .to_string(
            index=False,
            formatters={"predicted_consumption": lambda value: f"{value:.2f}"}
        )
    )

    print_top_3_peaks(forecast)
    save_daily_metrics(forecast, df, args.date)
    plot_day(forecast, df, args.date)


if __name__ == "__main__":
    main()
