import os
import re
import pickle
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Conv1D, Dense, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "datasets", "electricity_71.csv")

MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

MODEL_PATH = os.path.join(MODEL_DIR, "cnn_168.keras")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

# Historical data used for training
TRAIN_START = "2022-01-01"
TRAIN_END = "2024-12-31 23:00:00"

# Data used for final evaluation
TEST_START = "2025-01-01"
TEST_END = "2026-06-30 23:00:00"

# CNN parameters
WINDOW_SIZE = 168       # 7 days
FORECAST_HORIZON = 24   # next 24 hours

EPOCHS = 50
BATCH_SIZE = 32

RANDOM_SEED = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def ensure_directories():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)


def parse_numeric(value):
    """
    Converts strings like:

        '4 051 МВт*ч'
        '935 руб./МВт*ч'
        '4,051 МВт*ч'

    into float values.

    Empty values become NaN.
    """

    if pd.isna(value):
        return np.nan

    value = str(value).strip()

    if value == "":
        return np.nan

    # Replace comma decimal separator if present
    value = value.replace(",", ".")

    # Remove units and spaces
    value = value.replace("\xa0", " ")
    value = value.replace("МВт*ч", "")
    value = value.replace("руб./МВт*ч", "")
    value = value.replace("МВт·ч", "")
    value = value.replace("руб./МВт·ч", "")

    # Remove ordinary spaces used as thousand separators
    value = value.replace(" ", "")

    # Leave digits, minus sign and decimal point
    value = re.sub(r"[^0-9.\-]", "", value)

    if value in ("", "-", "."):
        return np.nan

    try:
        return float(value)
    except ValueError:
        return np.nan


# ============================================================
# DATA LOADING
# ============================================================

def load_data(path):
    """
    Loads the original CSV and converts it into a clean
    hourly time series.
    """

    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"CSV file not found: {path}"
        )

    try:
        df = pd.read_csv(
            path,
            sep=None,
            engine="python",
            encoding="utf-8-sig",
        )
    except UnicodeDecodeError:
        # Older exports may be saved in the Windows Cyrillic encoding.
        df = pd.read_csv(
            path,
            sep=None,
            engine="python",
            encoding="cp1251",
        )

    # CSV files may include a BOM or whitespace in the first header,
    # and the source files use either commas or tabs as separators.
    df.columns = [
        str(column).replace("\ufeff", "").strip().lower()
        for column in df.columns
    ]

    required_columns = [
        "date",
        "hour",
        "actual_consumption"
    ]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(
                f"Required column '{column}' was not found in CSV. "
                f"Columns found: {list(df.columns)}"
            )

    # --------------------------------------------------------
    # Date + hour -> datetime
    # --------------------------------------------------------

    df["hour"] = (
        df["hour"]
        .astype(str)
        .str.extract(r"(\d+)", expand=False)
        .astype(int)
    )

    df["datetime"] = pd.to_datetime(
        df["date"] + " " + df["hour"].astype(str).str.zfill(2),
        format="%Y-%m-%d %H"
    )

    # --------------------------------------------------------
    # Convert consumption
    # --------------------------------------------------------

    df["actual_consumption"] = (
        df["actual_consumption"]
        .apply(parse_numeric)
    )

    # Remove invalid datetime / target rows
    df = df.dropna(
        subset=["datetime", "actual_consumption"]
    )

    # Sort chronologically
    df = df.sort_values("datetime")

    # Remove duplicate timestamps
    duplicates = df["datetime"].duplicated().sum()

    if duplicates > 0:
        print(
            f"WARNING: found {duplicates} duplicate timestamps. "
            f"Keeping the first occurrence."
        )

        df = df.drop_duplicates(
            subset=["datetime"],
            keep="first"
        )

    # Keep only required columns
    result = df[
        ["datetime", "actual_consumption"]
    ].copy()

    result = result.set_index("datetime")

    # --------------------------------------------------------
    # Check missing hours
    # --------------------------------------------------------

    full_index = pd.date_range(
        start=result.index.min(),
        end=result.index.max(),
        freq="h"
    )

    missing_hours = full_index.difference(result.index)

    print(f"Rows: {len(result):,}")
    print(f"From: {result.index.min()}")
    print(f"To:   {result.index.max()}")

    print(
        f"Missing hourly timestamps: "
        f"{len(missing_hours):,}"
    )

    if len(missing_hours) > 0:
        # Interpolate short internal gaps only. Longer gaps stay missing
        # and trigger an error rather than introducing unreliable values.
        result = result.reindex(full_index)
        result["actual_consumption"] = result["actual_consumption"].interpolate(
            method="time",
            limit=2,
            limit_area="inside",
        )

        unresolved_hours = result.index[
            result["actual_consumption"].isna()
        ]

        if len(unresolved_hours) > 0:
            print("\nFirst unresolved timestamps:")

            for timestamp in unresolved_hours[:10]:
                print(f"  {timestamp}")

            raise ValueError(
                "\nThe dataset contains gaps longer than 2 hours or "
                "missing values at its boundaries. Fix the source data."
            )

        print(f"Interpolated missing hours: {len(missing_hours):,}")

    print("\nData loaded successfully.")

    return result


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

def split_data(df):
    train = df.loc[
        TRAIN_START:TRAIN_END
    ].copy()

    test = df.loc[
        TEST_START:TEST_END
    ].copy()

    if test.empty:
        # When the requested calendar test period is not present, use the
        # newest 20% as a chronological holdout. Never shuffle time-series data.
        test_size = max(
            FORECAST_HORIZON * 7,
            int(len(df) * 0.2),
        )
        split_index = len(df) - test_size

        if split_index < WINDOW_SIZE + FORECAST_HORIZON:
            raise ValueError(
                "Not enough rows to create chronological training and test "
                f"sets. Found {len(df)} hourly rows."
            )

        train = df.iloc[:split_index].copy()
        test = df.iloc[split_index:].copy()

        print(
            "Configured test period is not present in the CSV; "
            "using the latest 20% of available data as a chronological holdout."
        )

    if train.empty:
        raise ValueError("Training dataset is empty.")

    print("\n" + "=" * 70)
    print("DATA SPLIT")
    print("=" * 70)

    print(
        f"Train: {train.index.min()} -> "
        f"{train.index.max()}"
    )

    print(
        f"Test:  {test.index.min()} -> "
        f"{test.index.max()}"
    )

    print(f"Train rows: {len(train):,}")
    print(f"Test rows:  {len(test):,}")

    return train, test


# ============================================================
# CREATE WINDOWS
# ============================================================

def create_windows(values, window_size, horizon):
    """
    Converts a time series into supervised learning examples.

    Example:

        X:
        [168 previous hours]

        y:
        [next 24 hours]
    """

    X = []
    y = []

    max_start = len(values) - window_size - horizon + 1

    for start in range(max_start):
        end = start + window_size
        target_end = end + horizon

        X.append(values[start:end])
        y.append(values[end:target_end])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    # CNN expects:
    # (samples, timesteps, features)

    X = X[..., np.newaxis]

    return X, y


# ============================================================
# MODEL
# ============================================================

def create_model():
    model = Sequential([
        Input(
            shape=(WINDOW_SIZE, 1)
        ),

        Conv1D(
            filters=32,
            kernel_size=3,
            padding="same",
            activation="relu"
        ),

        Conv1D(
            filters=64,
            kernel_size=3,
            padding="same",
            activation="relu"
        ),

        Conv1D(
            filters=64,
            kernel_size=3,
            padding="same",
            activation="relu"
        ),

        Flatten(),

        Dense(
            128,
            activation="relu"
        ),

        Dropout(0.2),

        Dense(
            FORECAST_HORIZON
        )
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="mse",
        metrics=["mae"]
    )

    return model


# ============================================================
# TRAINING
# ============================================================

def train_model(train_df):
    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)

    values = train_df[
        "actual_consumption"
    ].values.reshape(-1, 1)

    # --------------------------------------------------------
    # IMPORTANT:
    # scaler is fitted ONLY on training data
    # --------------------------------------------------------

    scaler = StandardScaler()

    scaled_values = scaler.fit_transform(
        values
    ).flatten()

    X, y = create_windows(
        scaled_values,
        WINDOW_SIZE,
        FORECAST_HORIZON
    )

    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")

    # --------------------------------------------------------
    # Explicit time-based validation split
    #
    # Last 10% of training examples = validation
    # --------------------------------------------------------

    split_index = int(
        len(X) * 0.9
    )

    X_train = X[:split_index]
    y_train = y[:split_index]

    X_val = X[split_index:]
    y_val = y[split_index:]

    print(f"Training samples:   {len(X_train):,}")
    print(f"Validation samples: {len(X_val):,}")

    model = create_model()

    model.summary()

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=7,
            restore_best_weights=True
        ),

        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6
        )
    ]

    history = model.fit(
        X_train,
        y_train,

        validation_data=(
            X_val,
            y_val
        ),

        epochs=EPOCHS,
        batch_size=BATCH_SIZE,

        # VERY IMPORTANT for time series
        shuffle=False,

        callbacks=callbacks,

        verbose=1
    )

    return model, scaler, history


def plot_training_history(history):
    """Save model loss and MAE curves as a PNG image."""
    metrics = history.history
    epochs = range(1, len(metrics["loss"]) + 1)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(epochs, metrics["loss"], label="Training")
    axes[0].plot(epochs, metrics["val_loss"], label="Validation")
    axes[0].set_title("Training loss (MSE)")
    axes[0].set_ylabel("MSE")
    axes[0].legend()
    axes[0].grid(True)

    if "mae" in metrics and "val_mae" in metrics:
        axes[1].plot(epochs, metrics["mae"], label="Training")
        axes[1].plot(epochs, metrics["val_mae"], label="Validation")
        axes[1].set_title("Mean absolute error")
        axes[1].set_ylabel("MAE")
        axes[1].legend()
        axes[1].grid(True)

    axes[1].set_xlabel("Epoch")
    fig.tight_layout()

    filename = os.path.join(RESULTS_DIR, "training_history.png")
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"Training plot saved to: {filename}")


# ============================================================
# TEST EVALUATION
# ============================================================

def evaluate_model(model, scaler, df, test_df):
    print("\n" + "=" * 70)
    print("TEST EVALUATION")
    print("=" * 70)

    # We need the last WINDOW_SIZE hours before the test
    # to construct the first test example.

    test_start = test_df.index.min()
    test_end = test_df.index.max()

    history_start = (
        test_start
        - pd.Timedelta(hours=WINDOW_SIZE)
    )

    evaluation_df = df.loc[
        history_start:test_end
    ].copy()

    actual_values = (
        evaluation_df[
            "actual_consumption"
        ].values
    )

    scaled_values = scaler.transform(
        actual_values.reshape(-1, 1)
    ).flatten()

    predictions_scaled = []
    actual_scaled = []

    # The first prediction starts after WINDOW_SIZE hours.
    for start in range(
        0,
        len(scaled_values)
        - WINDOW_SIZE
        - FORECAST_HORIZON
        + 1,
        FORECAST_HORIZON
    ):
        X = scaled_values[
            start:
            start + WINDOW_SIZE
        ]

        y = scaled_values[
            start + WINDOW_SIZE:
            start + WINDOW_SIZE + FORECAST_HORIZON
        ]

        X = X.reshape(
            1,
            WINDOW_SIZE,
            1
        )

        prediction = model.predict(
            X,
            verbose=0
        )[0]

        predictions_scaled.extend(
            prediction
        )

        actual_scaled.extend(
            y
        )

    predictions_scaled = np.array(
        predictions_scaled
    )

    actual_scaled = np.array(
        actual_scaled
    )

    predictions = scaler.inverse_transform(
        predictions_scaled.reshape(-1, 1)
    ).flatten()

    actual = scaler.inverse_transform(
        actual_scaled.reshape(-1, 1)
    ).flatten()

    if len(actual) == 0:
        raise ValueError(
            "The test set is too short to evaluate a full "
            f"{FORECAST_HORIZON}-hour forecast."
        )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    mae = mean_absolute_error(
        actual,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predictions
        )
    )

    print(f"MAE:  {mae:.2f} MWh")
    print(f"RMSE: {rmse:.2f} MWh")

    prediction_index = pd.date_range(
        start=test_start,
        periods=len(actual),
        freq="h",
    )
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(prediction_index, actual, label="Actual consumption", linewidth=1)
    ax.plot(prediction_index, predictions, label="CNN prediction", linewidth=1)
    ax.set_title("Actual vs predicted consumption on test data")
    ax.set_xlabel("Time")
    ax.set_ylabel("Consumption, MWh")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()

    filename = os.path.join(RESULTS_DIR, "test_predictions.png")
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"Test prediction plot saved to: {filename}")

    return mae, rmse


# ============================================================
# FORECAST A SPECIFIC DAY
# ============================================================

def predict_day(
    model,
    scaler,
    df,
    target_date
):
    """
    Forecasts 24 hours for target_date.

    Example:
        target_date = '2025-07-01'
    """

    target_start = pd.Timestamp(
        target_date
    )

    target_end = (
        target_start
        + pd.Timedelta(hours=23)
    )

    history_start = (
        target_start
        - pd.Timedelta(hours=WINDOW_SIZE)
    )

    history_end = (
        target_start
        - pd.Timedelta(hours=1)
    )

    # --------------------------------------------------------
    # History
    # --------------------------------------------------------

    history = df.loc[
        history_start:history_end
    ]

    if len(history) != WINDOW_SIZE:
        raise ValueError(
            f"Expected {WINDOW_SIZE} historical hours, "
            f"but found {len(history)}."
        )

    history_values = (
        history[
            "actual_consumption"
        ].values
    )

    scaled_history = scaler.transform(
        history_values.reshape(-1, 1)
    ).flatten()

    X = scaled_history.reshape(
        1,
        WINDOW_SIZE,
        1
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    prediction_scaled = model.predict(
        X,
        verbose=0
    )[0]

    prediction = scaler.inverse_transform(
        prediction_scaled.reshape(-1, 1)
    ).flatten()

    forecast_index = pd.date_range(
        start=target_start,
        periods=FORECAST_HORIZON,
        freq="h"
    )

    forecast = pd.DataFrame({
        "datetime": forecast_index,
        "predicted_consumption": prediction
    })

    print("\n" + "=" * 70)
    print(f"FORECAST FOR {target_date}")
    print("=" * 70)

    print(
        forecast.to_string(
            index=False,
            formatters={
                "predicted_consumption":
                    lambda x: f"{x:.2f}"
            }
        )
    )

    return forecast


# ============================================================
# TOP 3 PEAK HOURS
# ============================================================

def find_top_3_peaks(forecast):
    peaks = (
        forecast
        .sort_values(
            "predicted_consumption",
            ascending=False
        )
        .head(3)
        .copy()
    )

    peaks = peaks.sort_values(
        "datetime"
    )

    print("\n" + "=" * 70)
    print("TOP 3 PREDICTED PEAK HOURS")
    print("=" * 70)

    for _, row in peaks.iterrows():
        print(
            f"{row['datetime']}: "
            f"{row['predicted_consumption']:.2f} MWh"
        )

    return peaks


# ============================================================
# PLOT FORECAST
# ============================================================

def plot_forecast(
    forecast,
    df,
    target_date
):
    target_start = pd.Timestamp(
        target_date
    )

    history_start = (
        target_start
        - pd.Timedelta(hours=48)
    )

    history = df.loc[
        history_start:
        target_start - pd.Timedelta(hours=1)
    ]

    plt.figure(
        figsize=(14, 6)
    )

    plt.plot(
        history.index,
        history["actual_consumption"],
        label="Actual consumption"
    )

    plt.plot(
        forecast["datetime"],
        forecast["predicted_consumption"],
        label="CNN prediction"
    )

    plt.axvline(
        target_start,
        linestyle="--"
    )

    plt.xlabel("Time")
    plt.ylabel("Consumption, MWh")

    plt.title(
        f"Electricity consumption forecast: "
        f"{target_date}"
    )

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    filename = os.path.join(
        RESULTS_DIR,
        f"forecast_{target_date}.png"
    )

    plt.savefig(
        filename,
        dpi=150
    )

    plt.close()

    print(
        f"\nForecast plot saved to: {filename}"
    )


# ============================================================
# SAVE MODEL
# ============================================================

def save_model(model, scaler):
    model.save(
        MODEL_PATH
    )

    with open(
        SCALER_PATH,
        "wb"
    ) as file:
        pickle.dump(
            scaler,
            file
        )

    print("\nModel saved:")
    print(f"  {MODEL_PATH}")
    print(f"  {SCALER_PATH}")


# ============================================================
# LOAD MODEL
# ============================================================

def load_saved_model():
    if not os.path.exists(
        MODEL_PATH
    ):
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    if not os.path.exists(
        SCALER_PATH
    ):
        raise FileNotFoundError(
            f"Scaler not found: {SCALER_PATH}"
        )

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    with open(
        SCALER_PATH,
        "rb"
    ) as file:
        scaler = pickle.load(
            file
        )

    return model, scaler


# ============================================================
# MAIN
# ============================================================

def train_pipeline():
    ensure_directories()

    df = load_data(DATA_PATH)

    train_df, test_df = split_data(
        df
    )

    # Train
    model, scaler, history = train_model(
        train_df
    )

    plot_training_history(history)

    # Test
    evaluate_model(
        model,
        scaler,
        df,
        test_df,
    )

    # Save
    save_model(
        model,
        scaler
    )


def forecast_pipeline(
    target_date
):
    df = load_data(
        DATA_PATH
    )

    model, scaler = load_saved_model()

    forecast = predict_day(
        model,
        scaler,
        df,
        target_date
    )

    find_top_3_peaks(
        forecast
    )

    plot_forecast(
        forecast,
        df,
        target_date
    )


# ============================================================
# COMMAND LINE
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Day-ahead electricity consumption "
            "peak forecasting using CNN 1D."
        )
    )

    parser.add_argument(
        "--mode",
        choices=[
            "train",
            "predict"
        ],
        default="train"
    )

    parser.add_argument(
        "--date",
        type=str,
        help=(
            "Date for prediction, "
            "e.g. 2025-07-01"
        )
    )

    args = parser.parse_args()

    if args.mode == "train":
        train_pipeline()

    elif args.mode == "predict":

        if not args.date:
            raise ValueError(
                "--date is required in predict mode."
            )

        forecast_pipeline(
            args.date
        )


if __name__ == "__main__":
    main()
