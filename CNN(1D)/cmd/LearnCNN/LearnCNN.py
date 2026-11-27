import os
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Input,
    Conv1D,
    MaxPooling1D,
    Flatten,
    Dense,
    Dropout,
)
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
)

warnings.filterwarnings("ignore", category=FutureWarning)


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(
    BASE_DIR,
    "../../..",
    "datasets",
    "electricity_75.csv"
)

# Regions used as shared context. The target is TARGET_REGION; every other
# region is an input feature. Set to None to keep the original single-region run.
MULTI_REGION = True
TARGET_REGION = "Челябинская область"
CONTEXT_REGIONS = ["Челябинская область", "Пермский край", "Свердловская область"]

# Historical data used for training
TRAIN_START = "2021-01-01"
TRAIN_END = "2025-12-31 23:00:00"

# Data used for final evaluation
TEST_START = "2026-01-01"
TEST_END = "2026-12-31 23:00:00"

# CNN parameters
WINDOW_SIZE = 168
FORECAST_HORIZON = 24

EPOCHS = 50
BATCH_SIZE = 32

RANDOM_SEED = 42

# Training parameters
VALIDATION_SPLIT = 0.1
EARLY_STOPPING_PATIENCE = 8
REDUCE_LR_PATIENCE = 4

# Output directory
RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results",
    "rolling_2026"
)

METRICS_PATH = os.path.join(
    RESULTS_DIR,
    "rolling_daily_metrics.csv"
)

PREDICTIONS_PATH = os.path.join(
    RESULTS_DIR,
    "rolling_predictions.csv"
)

SUMMARY_PATH = os.path.join(
    RESULTS_DIR,
    "rolling_summary.txt"
)

BOXPLOT_PATH = os.path.join(
    RESULTS_DIR,
    "rolling_boxplot.png"
)

MONTHLY_PATH = os.path.join(
    RESULTS_DIR,
    "monthly_metrics.csv"
)

# ============================================================
# RANDOM SEED
# ============================================================

os.environ["PYTHONHASHSEED"] = str(RANDOM_SEED)

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

tf.keras.utils.set_random_seed(RANDOM_SEED)


# ============================================================
# DATA LOADING
# ============================================================

def load_data(path, regions=None):
    """
    Loads the dataset and creates a continuous hourly time series.
    """

    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found:\n{path}"
        )

    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="cp1251")

    required_columns = [
        "date",
        "hour",
        "actual_consumption"
    ]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(
                f"Required column is missing: {column}"
            )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    df["hour"] = pd.to_numeric(
        df["hour"],
        errors="coerce"
    )

    df["actual_consumption"] = pd.to_numeric(
        df["actual_consumption"],
        errors="coerce"
    )

    df = df.dropna(
        subset=[
            "date",
            "hour",
            "actual_consumption"
        ]
    )

    # Construct hourly timestamp.
    # Example: date=2026-01-01, hour=5
    # becomes 2026-01-01 05:00:00.

    df["timestamp"] = (
        df["date"].dt.normalize()
        + pd.to_timedelta(df["hour"], unit="h")
    )

    if regions and "region_name" in df.columns:
        df = df[df["region_name"].isin(regions)][["timestamp", "region_name", "actual_consumption"]]
        df = df.pivot_table(index="timestamp", columns="region_name", values="actual_consumption", aggfunc="last")
        missing_regions = set(regions) - set(df.columns)
        if missing_regions:
            raise ValueError(f"Regions absent from dataset: {sorted(missing_regions)}")
        df = df.reindex(columns=regions)
    else:
        df = df[["timestamp", "actual_consumption"]]

    df = df.sort_values("timestamp")

    # Remove duplicate timestamps.
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.drop_duplicates(subset="timestamp", keep="last").set_index("timestamp")

    df = df.set_index("timestamp")

    # Convert to a continuous hourly index.
    full_index = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq="h"
    )

    df = df.reindex(full_index)

    df.index.name = "timestamp"

    print(f"Dataset loaded: {len(df)} rows")
    print(f"Start: {df.index.min()}")
    print(f"End:   {df.index.max()}")

    missing = int(df.isna().sum().sum())

    print(f"Missing hourly values: {missing}")

    return df


# ============================================================
# NORMALIZATION
# ============================================================

def fit_scaler(values):
    """
    Fits a simple StandardScaler using NumPy.

    Only training data is used.
    """

    values = np.asarray(
        values,
        dtype=np.float32
    )

    mean = np.mean(values)
    std = np.std(values)

    if std == 0:
        std = 1.0

    return mean, std


def transform(values, mean, std):
    return (values - mean) / std


def inverse_transform(values, mean, std):
    return values * std + mean


# ============================================================
# CREATE TRAINING WINDOWS
# ============================================================

def create_windows(
    values,
    window_size,
    forecast_horizon
):
    """
    Creates input sequences and target sequences.

    X:
        Previous 168 hours.

    y:
        Next 24 hours.

    Windows containing missing values are discarded.
    """

    X = []
    y = []

    values = np.asarray(
        values,
        dtype=np.float32
    )

    total_length = (
        window_size + forecast_horizon
    )

    for i in range(
        len(values) - total_length + 1
    ):

        input_window = values[
            i:i + window_size
        ]

        target_window = values[
            i + window_size:
            i + total_length
        ]

        if (
            np.isnan(input_window).any()
            or np.isnan(target_window).any()
        ):
            continue

        X.append(input_window)
        y.append(target_window)

    if len(X) == 0:
        return (
            np.empty(
                (0, window_size, 1),
                dtype=np.float32
            ),
            np.empty(
                (0, forecast_horizon),
                dtype=np.float32
            )
        )

    X = np.asarray(
        X,
        dtype=np.float32
    )

    y = np.asarray(
        y,
        dtype=np.float32
    )

    # CNN input shape:
    # samples, timesteps, features

    X = X.reshape(
        X.shape[0],
        X.shape[1],
        1
    )

    return X, y


# ============================================================
# CNN MODEL
# ============================================================

def build_model():
    """
    Creates a 1D CNN for 24-hour forecasting.
    """

    model = Sequential(
        [
            Input(
                shape=(
                    WINDOW_SIZE,
                    1
                )
            ),

            Conv1D(
                filters=64,
                kernel_size=3,
                activation="relu",
                padding="same"
            ),

            MaxPooling1D(
                pool_size=2
            ),

            Conv1D(
                filters=32,
                kernel_size=3,
                activation="relu",
                padding="same"
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
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="mse"
    )

    return model


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model(train_values):
    """
    Trains the CNN using all available historical data.

    The scaler is fitted only on the current training history.
    """

    train_values = np.asarray(
        train_values,
        dtype=np.float32
    )

    # Use only non-missing historical observations
    valid_values = train_values[
        ~np.isnan(train_values)
    ]

    if len(valid_values) == 0:
        raise ValueError(
            "No valid historical values for training."
        )

    mean, std = fit_scaler(
        valid_values
    )

    scaled_values = transform(
        train_values,
        mean,
        std
    )

    X_train, y_train = create_windows(
        scaled_values,
        WINDOW_SIZE,
        FORECAST_HORIZON
    )

    if len(X_train) < 10:
        raise ValueError(
            "Not enough training windows."
        )

    print(
        f"Training samples: {len(X_train)}"
    )

    model = build_model()

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=0
        ),

        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=REDUCE_LR_PATIENCE,
            min_lr=0.00001,
            verbose=0
        )
    ]

    model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        shuffle=False,
        callbacks=callbacks,
        verbose=0
    )

    return model, mean, std


# ============================================================
# FORECAST ONE DAY
# ============================================================

def predict_day(
    model,
    history_values,
    mean,
    std
):
    """
    Predicts the next 24 hours using the latest 168 hours.
    """

    history_values = np.asarray(
        history_values,
        dtype=np.float32
    )

    if len(history_values) < WINDOW_SIZE:
        raise ValueError(
            "Not enough history for forecasting."
        )

    input_window = history_values[
        -WINDOW_SIZE:
    ]

    if np.isnan(input_window).any():
        raise ValueError(
            "The last 168 hours contain missing values."
        )

    scaled_input = transform(
        input_window,
        mean,
        std
    )

    X = scaled_input.reshape(
        1,
        WINDOW_SIZE,
        1
    )

    prediction_scaled = model.predict(
        X,
        verbose=0
    )[0]

    prediction = inverse_transform(
        prediction_scaled,
        mean,
        std
    )

    return prediction


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    actual,
    predicted
):
    """
    Calculates daily MSE, RMSE, MAPE and TOP-3 Hit Rate.
    """

    actual = np.asarray(
        actual,
        dtype=np.float64
    )

    predicted = np.asarray(
        predicted,
        dtype=np.float64
    )

    if (
        len(actual) != FORECAST_HORIZON
        or len(predicted) != FORECAST_HORIZON
    ):
        raise ValueError(
            "Actual and predicted arrays "
            "must contain exactly 24 values."
        )

    errors = actual - predicted

    mse = np.mean(
        errors ** 2
    )

    rmse = np.sqrt(mse)

    nonzero_mask = actual != 0

    if nonzero_mask.any():
        mape = np.mean(
            np.abs(
                errors[nonzero_mask]
                / actual[nonzero_mask]
            )
        ) * 100
    else:
        mape = np.nan

    # Indices of the 3 highest actual values
    actual_top3 = set(
        np.argsort(actual)[-3:]
    )

    # Indices of the 3 highest predicted values
    predicted_top3 = set(
        np.argsort(predicted)[-3:]
    )

    intersection = (
        actual_top3
        & predicted_top3
    )

    top3_hit = (
        len(intersection) / 3
    ) * 100

    return {
        "MSE": float(mse),
        "RMSE": float(rmse),
        "MAPE": float(mape),
        "TOP3_HIT": float(top3_hit)
    }


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    metrics_history,
    predictions_history
):
    """
    Saves daily metrics and hourly predictions.
    """

    metrics_df = pd.DataFrame(
        metrics_history
    )

    predictions_df = pd.DataFrame(
        predictions_history
    )

    metrics_df.to_csv(
        METRICS_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    predictions_df.to_csv(
        PREDICTIONS_PATH,
        index=False,
        encoding="utf-8-sig"
    )


# ============================================================
# MONTHLY STATISTICS
# ============================================================

def save_monthly_statistics(
    metrics_df
):
    """
    Calculates monthly average metrics.
    """

    df = metrics_df.copy()

    df["date"] = pd.to_datetime(
        df["date"]
    )

    df["month"] = (
        df["date"].dt.to_period("M")
    )

    monthly = df.groupby(
        "month"
    )[
        [
            "MSE",
            "RMSE",
            "MAPE",
            "TOP3_HIT"
        ]
    ].agg(
        [
            "mean",
            "median",
            "std",
            "min",
            "max"
        ]
    )

    monthly.to_csv(
        MONTHLY_PATH,
        encoding="utf-8-sig"
    )

    print(
        "\nMonthly statistics saved:"
    )

    print(MONTHLY_PATH)


# ============================================================
# BOXPLOT
# ============================================================

def create_boxplot(
    metrics_df
):
    """
    Creates boxplots for daily metrics.
    """

    metrics = [
        "MSE",
        "RMSE",
        "MAPE",
        "TOP3_HIT"
    ]

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14, 10)
    )

    axes = axes.flatten()

    for i, metric in enumerate(
        metrics
    ):

        values = metrics_df[
            metric
        ].dropna()

        axes[i].boxplot(
            values,
            patch_artist=False
        )

        axes[i].set_title(
            f"Daily {metric} distribution"
        )

        axes[i].set_ylabel(
            metric
        )

        axes[i].grid(
            axis="y",
            alpha=0.3
        )

    plt.tight_layout()

    plt.savefig(
        BOXPLOT_PATH,
        dpi=300
    )

    plt.close()

    print(
        f"Boxplot saved: {BOXPLOT_PATH}"
    )


# ============================================================
# SUMMARY
# ============================================================

def save_summary(
    metrics_df
):
    """
    Saves final statistics for the evaluation period.
    """

    metrics = [
        "MSE",
        "RMSE",
        "MAPE",
        "TOP3_HIT"
    ]

    with open(
        SUMMARY_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "ROLLING LEARNING — FINAL METRICS\n"
        )

        file.write(
            "=" * 70 + "\n"
        )

        file.write(
            f"Evaluation period: "
            f"{TEST_START} — {TEST_END}\n"
        )

        file.write(
            f"Days evaluated: "
            f"{len(metrics_df)}\n\n"
        )

        for metric in metrics:

            values = metrics_df[
                metric
            ].dropna()

            file.write(
                f"{metric}\n"
            )

            file.write(
                f"Mean:   {values.mean():.4f}\n"
            )

            file.write(
                f"Median: {values.median():.4f}\n"
            )

            file.write(
                f"Std:    {values.std():.4f}\n"
            )

            file.write(
                f"Min:    {values.min():.4f}\n"
            )

            file.write(
                f"Max:    {values.max():.4f}\n"
            )

            file.write(
                "-" * 50 + "\n"
            )

    print(
        f"Summary saved: {SUMMARY_PATH}"
    )


# ============================================================
# MAIN ROLLING LEARNING
# ============================================================

def main():

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    data = load_data(
        DATA_PATH
    )

    # Historical training period
    train_start = pd.Timestamp(
        TRAIN_START
    )

    train_end = pd.Timestamp(
        TRAIN_END
    )

    # Evaluation period
    test_start = pd.Timestamp(
        TEST_START
    )

    test_end = pd.Timestamp(
        TEST_END
    )

    # Initial historical data
    historical_data = data.loc[
        train_start:train_end,
        "actual_consumption"
    ].copy()

    # Data available for evaluation
    test_data = data.loc[
        test_start:test_end,
        "actual_consumption"
    ].copy()

    if historical_data.empty:
        raise ValueError(
            "Historical training data is empty."
        )

    if test_data.empty:
        raise ValueError(
            "No evaluation data for 2026."
        )

    # Generate all forecast dates
    forecast_dates = pd.date_range(
        start=test_start.normalize(),
        end=test_end.normalize(),
        freq="D"
    )

    metrics_history = []
    predictions_history = []

    print("\n")
    print("=" * 70)
    print("ROLLING LEARNING — 2026")
    print("=" * 70)

    print(
        f"Initial training period: "
        f"{TRAIN_START} — {TRAIN_END}"
    )

    print(
        f"Evaluation period: "
        f"{TEST_START} — {TEST_END}"
    )

    print(
        f"Forecast days: "
        f"{len(forecast_dates)}"
    )

    # --------------------------------------------------------
    # MAIN LOOP
    # --------------------------------------------------------

    for day_number, forecast_date in enumerate(
        forecast_dates,
        start=1
    ):

        print("\n")
        print("-" * 70)

        print(
            f"DAY {day_number}/"
            f"{len(forecast_dates)}"
        )

        print(
            f"Forecast date: "
            f"{forecast_date.date()}"
        )

        # ----------------------------------------------------
        # 1. Get historical data before forecast date
        # ----------------------------------------------------

        available_history = data.loc[
            train_start:
            forecast_date - pd.Timedelta(hours=1),
            "actual_consumption"
        ].copy()

        if available_history.empty:
            print(
                "Skipping: no historical data."
            )
            continue

        # ----------------------------------------------------
        # 2. Get the actual 24 hours for evaluation
        # ----------------------------------------------------

        day_end = (
            forecast_date
            + pd.Timedelta(hours=23)
        )

        actual_day = data.loc[
            forecast_date:day_end,
            "actual_consumption"
        ]

        if len(actual_day) != 24:
            print(
                "Skipping: the day does not "
                "contain 24 hourly observations."
            )
            continue

        if actual_day.isna().any():
            print(
                "Skipping: actual values "
                "contain missing observations."
            )
            continue

        # ----------------------------------------------------
        # 3. Train the model
        # ----------------------------------------------------

        print(
            "Training CNN..."
        )

        model, mean, std = train_model(
            available_history.values
        )

        # ----------------------------------------------------
        # 4. Forecast the next 24 hours
        # ----------------------------------------------------

        print(
            "Predicting next 24 hours..."
        )

        predictions = predict_day(
            model,
            available_history.values,
            mean,
            std
        )

        actual = actual_day.values

        # ----------------------------------------------------
        # 5. Calculate daily metrics
        # ----------------------------------------------------

        daily_metrics = calculate_metrics(
            actual,
            predictions
        )

        daily_metrics["date"] = (
            forecast_date.strftime(
                "%Y-%m-%d"
            )
        )

        metrics_history.append(
            daily_metrics
        )

        # ----------------------------------------------------
        # 6. Save hourly predictions
        # ----------------------------------------------------

        for hour in range(
            FORECAST_HORIZON
        ):

            predictions_history.append(
                {
                    "date": forecast_date.strftime(
                        "%Y-%m-%d"
                    ),

                    "hour": hour,

                    "actual_consumption": float(
                        actual[hour]
                    ),

                    "predicted_consumption": float(
                        predictions[hour]
                    )
                }
            )

        # ----------------------------------------------------
        # 7. Print daily results
        # ----------------------------------------------------

        print(
            f"MSE:      "
            f"{daily_metrics['MSE']:.4f}"
        )

        print(
            f"RMSE:     "
            f"{daily_metrics['RMSE']:.4f}"
        )

        print(
            f"MAPE:     "
            f"{daily_metrics['MAPE']:.4f}%"
        )

        print(
            f"TOP-3 HIT:"
            f" {daily_metrics['TOP3_HIT']:.2f}%"
        )

        # ----------------------------------------------------
        # 8. Save results after every day
        # ----------------------------------------------------

        save_results(
            metrics_history,
            predictions_history
        )

        print(
            "Daily results saved."
        )

        # ----------------------------------------------------
        # 9. The next iteration automatically includes
        #    this day's actual data in historical_data.
        #
        #    No manual append is required because the next
        #    iteration reads all available data before
        #    its forecast date.
        # ----------------------------------------------------

        del model

        tf.keras.backend.clear_session()

    # ========================================================
    # FINAL RESULTS
    # ========================================================

    if len(metrics_history) == 0:
        raise RuntimeError(
            "No days were successfully evaluated."
        )

    metrics_df = pd.DataFrame(
        metrics_history
    )

    print("\n")
    print("=" * 70)
    print("ROLLING LEARNING — FINAL RESULTS")
    print("=" * 70)

    print(
        f"Successfully evaluated days: "
        f"{len(metrics_df)}"
    )

    print(
        metrics_df[
            [
                "MSE",
                "RMSE",
                "MAPE",
                "TOP3_HIT"
            ]
        ].describe()
    )

    # Save final files
    save_results(
        metrics_history,
        predictions_history
    )

    save_monthly_statistics(
        metrics_df
    )

    create_boxplot(
        metrics_df
    )

    save_summary(
        metrics_df
    )

    print("\n")
    print("=" * 70)
    print("PROGRAM FINISHED")
    print("=" * 70)

    print(
        f"Results directory:\n"
        f"{RESULTS_DIR}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
