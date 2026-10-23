
import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Conv1D, Dense, Flatten, Dropout


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    BASE_DIR, "../../../", "datasets", "electricity_75.csv"
)

RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODEL_DIR = os.path.join(BASE_DIR, "models")

WINDOW_SIZE = 168
FORECAST_HORIZON = 24

# Initial model training
INITIAL_TRAIN_START = "2022-01-01"
INITIAL_TRAIN_END = "2023-12-31 23:00:00"

# Walk-forward evaluation
TEST_START = "2024-01-01"
TEST_END = "2026-07-31 23:00:00"

# Rolling training window: last 2 years of available history
TRAINING_WINDOW_DAYS = 730

# Training parameters
INITIAL_EPOCHS = 20
ROLLING_EPOCHS = 1
BATCH_SIZE = 32

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# DATA LOADING
# ============================================================

def parse_numeric(value):
    if pd.isna(value):
        return np.nan

    value = str(value).strip()
    value = value.replace("\xa0", " ")
    value = value.replace("МВт*ч", "")
    value = value.replace("МВт·ч", "")
    value = value.replace(" ", "")
    value = value.replace(",", ".")

    value = re.sub(r"[^0-9.\-]", "", value)

    if value in ("", "-", "."):
        return np.nan

    try:
        return float(value)
    except ValueError:
        return np.nan


def load_data():
    df = pd.read_csv(
        DATA_PATH,
        sep=",",
        encoding="utf-8-sig",
    )

    df.columns = [
        str(c).replace("\ufeff", "").strip().lower()
        for c in df.columns
    ]

    required = {"date", "hour", "actual_consumption"}

    if not required.issubset(df.columns):
        raise ValueError(
            f"Missing columns: {required - set(df.columns)}"
        )

    df["hour"] = (
        df["hour"]
        .astype(str)
        .str.extract(r"(\d+)", expand=False)
        .astype(int)
    )

    df["datetime"] = pd.to_datetime(
        df["date"].astype(str)
        + " "
        + df["hour"].astype(str).str.zfill(2),
        format="%Y-%m-%d %H",
        errors="coerce",
    )

    df["actual_consumption"] = (
        df["actual_consumption"].apply(parse_numeric)
    )

    df = df.dropna(
        subset=["datetime", "actual_consumption"]
    )

    df = df.sort_values("datetime")
    df = df.drop_duplicates("datetime", keep="first")
    df = df.set_index("datetime")

    # Require continuous hourly data.
    full_index = pd.date_range(
        df.index.min(),
        df.index.max(),
        freq="h",
    )

    df = df.reindex(full_index)
    df.index.name = "datetime"

    # Fill short internal gaps in the hourly series (for example, a few
    # missing consumption values in the source CSV) using time interpolation.
    missing_count = int(df["actual_consumption"].isna().sum())
    if missing_count:
        df["actual_consumption"] = df["actual_consumption"].interpolate(
            method="time",
            limit=3,
            limit_area="inside",
        )
        filled_count = missing_count - int(
            df["actual_consumption"].isna().sum()
        )
        print(f"Interpolated missing hourly values: {filled_count}")

    if df["actual_consumption"].isna().any():
        raise ValueError(
            "Unfilled hourly consumption values remain after interpolation. "
            "Check for long gaps or missing values at the dataset boundaries."
        )

    print(f"Dataset: {df.index.min()} -> {df.index.max()}")
    print(f"Total hours: {len(df):,}")

    return df[["actual_consumption"]]


# ============================================================
# CREATE TRAINING WINDOWS
# ============================================================

def create_windows(values):
    X = []
    y = []

    max_start = (
        len(values) - WINDOW_SIZE - FORECAST_HORIZON + 1
    )

    for start in range(max_start):
        end = start + WINDOW_SIZE

        X.append(values[start:end])
        y.append(values[end:end + FORECAST_HORIZON])

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    X = X[..., np.newaxis]

    return X, y


# ============================================================
# CNN MODEL
# ============================================================

def create_model():
    model = Sequential([
        Input(shape=(WINDOW_SIZE, 1)),

        Conv1D(
            filters=32,
            kernel_size=3,
            padding="same",
            activation="relu",
        ),

        Conv1D(
            filters=64,
            kernel_size=3,
            padding="same",
            activation="relu",
        ),

        Conv1D(
            filters=64,
            kernel_size=3,
            padding="same",
            activation="relu",
        ),

        Flatten(),

        Dense(128, activation="relu"),

        Dropout(0.2),

        Dense(FORECAST_HORIZON),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="mse",
        metrics=["mae"],
    )

    return model


# ============================================================
# ROLLING TRAINING
# ============================================================

def rolling_training(df):
    initial_start = pd.Timestamp(INITIAL_TRAIN_START)
    initial_end = pd.Timestamp(INITIAL_TRAIN_END)

    test_start = pd.Timestamp(TEST_START)
    test_end = pd.Timestamp(TEST_END)

    # The initial training period must precede the test period.
    initial_df = df.loc[initial_start:initial_end].copy()

    if initial_df.empty:
        raise ValueError("Initial training dataset is empty.")

    if initial_df.index.max() >= test_start:
        raise ValueError(
            "Initial training data must end before TEST_START."
        )

    if df.index.min() > initial_start or df.index.max() < test_end:
        raise ValueError(
            "The dataset does not cover the configured date range."
        )

    # Fit the scaler only on initial training data.
    # Keep it fixed throughout the experiment.
    scaler = StandardScaler()

    scaler.fit(
        initial_df[["actual_consumption"]].to_numpy()
    )

    initial_values = scaler.transform(
        initial_df[["actual_consumption"]].to_numpy()
    ).flatten()

    X_initial, y_initial = create_windows(initial_values)

    if len(X_initial) == 0:
        raise ValueError("Not enough data for initial training.")

    model = create_model()

    print("\nINITIAL TRAINING")
    print(f"Training examples: {len(X_initial):,}")

    model.fit(
        X_initial,
        y_initial,
        epochs=INITIAL_EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=False,
        verbose=1,
    )

    predictions = []
    actual_values = []
    prediction_dates = []

    # Start at the first complete forecast day.
    forecast_start = test_start.normalize()

    print("\nROLLING TRAINING AND FORECASTING")

    while (
        forecast_start + pd.Timedelta(hours=23) <= test_end
    ):
        forecast_end = (
            forecast_start
            + pd.Timedelta(hours=FORECAST_HORIZON - 1)
        )

        # Only data available BEFORE the forecast may be used
        # for training and input preparation.
        available_end = forecast_start - pd.Timedelta(hours=1)

        # A fixed-size rolling training period.
        training_start = max(
            initial_start,
            available_end
            - pd.Timedelta(days=TRAINING_WINDOW_DAYS)
            + pd.Timedelta(hours=1),
        )

        training_df = df.loc[
            training_start:available_end
        ]

        # Train using the latest available historical data.
        training_values = scaler.transform(
            training_df[["actual_consumption"]].to_numpy()
        ).flatten()

        X_train, y_train = create_windows(training_values)

        if len(X_train) == 0:
            raise ValueError(
                f"Not enough training data on {forecast_start}."
            )

        # Update existing weights instead of initializing a new CNN.
        model.fit(
            X_train,
            y_train,
            epochs=ROLLING_EPOCHS,
            batch_size=BATCH_SIZE,
            shuffle=False,
            verbose=0,
        )

        # Latest 168 actual hours before the forecast.
        history_start = (
            forecast_start - pd.Timedelta(hours=WINDOW_SIZE)
        )

        history = df.loc[
            history_start:available_end,
            "actual_consumption",
        ]

        if len(history) != WINDOW_SIZE:
            raise ValueError(
                f"Expected {WINDOW_SIZE} input hours on "
                f"{forecast_start}; found {len(history)}."
            )

        X = scaler.transform(
            history.to_numpy().reshape(-1, 1)
        ).flatten().reshape(1, WINDOW_SIZE, 1)

        # Forecast the next 24 hours.
        prediction_scaled = model.predict(
            X, verbose=0
        )[0]

        prediction = scaler.inverse_transform(
            prediction_scaled.reshape(-1, 1)
        ).flatten()

        dates = pd.date_range(
            forecast_start,
            periods=FORECAST_HORIZON,
            freq="h",
        )

        # Actual values are used only AFTER the prediction
        # has been produced, for evaluation and future updates.
        actual = df.loc[
            dates, "actual_consumption"
        ].to_numpy()

        predictions.extend(prediction)
        actual_values.extend(actual)
        prediction_dates.extend(dates)

        print(
            f"{forecast_start.date()} | "
            f"training rows: {len(training_df):,} | "
            f"forecast: {forecast_start.date()} - "
            f"{forecast_end.date()}"
        )

        # Move forward by one day.
        # The next iteration will include newly observed actuals.
        forecast_start += pd.Timedelta(days=1)

    comparison = pd.DataFrame(
        {
            "actual_consumption": actual_values,
            "cnn_prediction": predictions,
        },
        index=pd.DatetimeIndex(
            prediction_dates, name="datetime"
        ),
    )

    return model, scaler, comparison


# ============================================================
# EVALUATION
# ============================================================

def evaluate(comparison):
    actual = comparison["actual_consumption"].to_numpy()
    predicted = comparison["cnn_prediction"].to_numpy()

    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))

    nonzero = actual != 0

    mape = (
        np.mean(
            np.abs(
                (actual[nonzero] - predicted[nonzero])
                / actual[nonzero]
            )
        ) * 100
        if nonzero.any()
        else np.nan
    )

    print("\nFINAL METRICS")
    print(f"MAE:  {mae:.2f} MWh")
    print(f"RMSE: {rmse:.2f} MWh")
    print(f"MAPE: {mape:.2f}%")

    csv_path = os.path.join(
        RESULTS_DIR, "rolling_test_comparison.csv"
    )

    comparison.to_csv(
        csv_path, encoding="utf-8-sig"
    )

    fig, ax = plt.subplots(figsize=(15, 6))

    ax.plot(
        comparison.index,
        actual,
        label="Actual consumption",
        linewidth=1,
    )

    ax.plot(
        comparison.index,
        predicted,
        label="Rolling CNN prediction",
        linewidth=1,
    )

    ax.set_title("Walk-forward CNN: actual vs predicted")
    ax.set_xlabel("Date")
    ax.set_ylabel("Consumption, MWh")
    ax.legend()
    ax.grid(True)

    fig.tight_layout()

    plot_path = os.path.join(
        RESULTS_DIR, "rolling_test_predictions.png"
    )

    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    print(f"\nCSV saved: {csv_path}")
    print(f"Plot saved: {plot_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    df = load_data()

    model, scaler, comparison = rolling_training(df)

    evaluate(comparison)

    model.save(
        os.path.join(MODEL_DIR, "rolling_cnn.keras")
    )

    print("\nRolling training completed.")


if __name__ == "__main__":
    main()
