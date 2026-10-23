
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ============================================================
# CONFIGURATION
# ============================================================

ROOT_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_PATH = os.path.join(
    ROOT_DIR,
    "datasets",
    "electricity_75.csv"
)

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "results",
    "SlideLearn"
)

MODEL_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "models",
    "SlideLearn"
)

WINDOW_SIZE = 168
FORECAST_HORIZON = 24

INITIAL_TRAIN_START = "2022-01-01 00:00:00"
INITIAL_TRAIN_END = "2023-12-31 23:00:00"

TEST_START = "2024-01-01 00:00:00"
TEST_END = "2026-07-31 23:00:00"

TRAINING_WINDOW_DAYS = 730

INITIAL_EPOCHS = 20
ROLLING_EPOCHS = 1
BATCH_SIZE = 32

RANDOM_SEED = 42

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# ============================================================
# DATA LOADING
# ============================================================

def read_csv(path):
    """Читает CSV с поддержкой нескольких распространённых кодировок."""
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="cp1251")


def parse_number(value):
    """Преобразует значения вроде '4 076 МВт*ч' в число 4076."""
    if pd.isna(value):
        return np.nan

    if isinstance(value, (int, float, np.number)):
        return float(value)

    value = str(value).replace("\xa0", " ")
    value = value.replace(",", ".")
    value = re.sub(r"[^0-9.\-]", "", value.replace(" ", ""))

    try:
        return float(value)
    except ValueError:
        return np.nan


def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Файл с данными не найден: {DATA_PATH}"
        )

    df = read_csv(DATA_PATH)

    df.columns = (
        df.columns.astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
        .str.lower()
    )

    required = {"date", "hour", "actual_consumption"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"В CSV отсутствуют столбцы: {sorted(missing)}"
        )

    # Поддерживаем как числовой час, так и строковые значения.
    hours = (
        df["hour"]
        .astype(str)
        .str.extract(r"(\d+)")[0]
    )

    df["hour"] = pd.to_numeric(hours, errors="coerce")
    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    df["actual_consumption"] = (
        df["actual_consumption"].apply(parse_number)
    )

    df = df.dropna(
        subset=["date", "hour", "actual_consumption"]
    ).copy()

    df["hour"] = df["hour"].astype(int)

    df["datetime"] = (
        df["date"]
        + pd.to_timedelta(df["hour"], unit="h")
    )

    df = df.sort_values("datetime")
    df = df.drop_duplicates("datetime", keep="last")
    df = df.set_index("datetime")

    series = df["actual_consumption"].astype(float).sort_index()

    if series.empty:
        raise ValueError("После обработки CSV не осталось данных.")

    # Проверяем непрерывность почасового ряда.
    full_index = pd.date_range(
        start=series.index.min(),
        end=series.index.max(),
        freq="h"
    )

    series = series.reindex(full_index)

    if series.isna().any():
        missing_count = int(series.isna().sum())

        raise ValueError(
            f"В почасовом ряду обнаружено {missing_count} пропусков. "
            "Проверь исходный CSV перед обучением."
        )

    series.name = "actual_consumption"

    print("\nДанные загружены.")
    print(f"Количество часов: {len(series):,}")
    print(f"Начало ряда: {series.index.min()}")
    print(f"Конец ряда:  {series.index.max()}")

    return series


# ============================================================
# SEQUENCE GENERATION
# ============================================================

def create_sequences(values):
    """
    Формирует обучающие примеры:
    X — предыдущие 168 часов;
    y — следующие 24 часа.
    """
    X = []
    y = []

    for i in range(
        len(values) - WINDOW_SIZE - FORECAST_HORIZON + 1
    ):
        X.append(
            values[i:i + WINDOW_SIZE]
        )

        y.append(
            values[
                i + WINDOW_SIZE:
                i + WINDOW_SIZE + FORECAST_HORIZON
            ].flatten()
        )

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    if len(X) == 0:
        raise ValueError(
            "Недостаточно данных для создания обучающих примеров."
        )

    return X, y


# ============================================================
# CNN MODEL
# ============================================================

def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(
            shape=(WINDOW_SIZE, 1)
        ),

        tf.keras.layers.Conv1D(
            filters=64,
            kernel_size=3,
            padding="causal",
            activation="relu"
        ),

        tf.keras.layers.Conv1D(
            filters=32,
            kernel_size=3,
            padding="causal",
            activation="relu"
        ),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(
            64,
            activation="relu"
        ),

        tf.keras.layers.Dense(
            FORECAST_HORIZON
        )
    ])

    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"]
    )

    return model


# ============================================================
# SLIDELEARN: ROLLING TRAINING
# ============================================================

def rolling_training(series):
    initial_start = pd.Timestamp(INITIAL_TRAIN_START)
    initial_end = pd.Timestamp(INITIAL_TRAIN_END)

    test_start = pd.Timestamp(TEST_START)
    test_end = pd.Timestamp(TEST_END)

    initial_series = series.loc[
        (series.index >= initial_start)
        & (series.index <= initial_end)
    ]

    if len(initial_series) < WINDOW_SIZE + FORECAST_HORIZON:
        raise ValueError(
            "Недостаточно данных для начального обучения."
        )

    # Масштабировщик обучается один раз.
    # Далее он используется во всех шагах эксперимента.
    scaler = StandardScaler()

    initial_scaled = scaler.fit_transform(
        initial_series.to_numpy().reshape(-1, 1)
    )

    X_train, y_train = create_sequences(initial_scaled)

    model = create_model()

    print("\nНачальное обучение CNN...")
    print(f"Обучающих примеров: {len(X_train):,}")

    initial_history = model.fit(
        X_train,
        y_train,
        epochs=INITIAL_EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=False,
        verbose=1
    )

    # Сохраняем метрики каждой эпохи.
    history_rows = []

    for epoch in range(len(initial_history.history["loss"])):
        history_rows.append({
            "stage": "initial",
            "step": epoch + 1,
            "loss": initial_history.history["loss"][epoch],
            "mae": initial_history.history["mae"][epoch]
        })

    predictions = []
    actual_values = []
    timestamps = []

    test_days = pd.date_range(
        start=test_start.normalize(),
        end=test_end.normalize(),
        freq="D"
    )

    print("\nНачинается скользящее обучение...")

    for day_number, day in enumerate(test_days, start=1):
        day_start = day
        day_end = day + pd.Timedelta(hours=23)

        # Для каждого прогноза доступны только данные до начала суток.
        available_history = series.loc[
            series.index < day_start
        ]

        if len(available_history) < WINDOW_SIZE:
            continue

        training_start = day_start - pd.Timedelta(
            days=TRAINING_WINDOW_DAYS
        )

        rolling_series = available_history.loc[
            available_history.index >= training_start
        ]

        # Для ежедневного дообучения нужно достаточно примеров.
        if len(rolling_series) < WINDOW_SIZE + FORECAST_HORIZON:
            continue

        # Используем тот же scaler, что и при начальном обучении.
        rolling_scaled = scaler.transform(
            rolling_series.to_numpy().reshape(-1, 1)
        )

        X_roll, y_roll = create_sequences(rolling_scaled)

        # Продолжаем обучение той же модели с текущими весами.
        rolling_history = model.fit(
            X_roll,
            y_roll,
            epochs=ROLLING_EPOCHS,
            batch_size=BATCH_SIZE,
            shuffle=False,
            verbose=0
        )

        for epoch in range(len(rolling_history.history["loss"])):
            history_rows.append({
                "stage": "rolling",
                "step": len(history_rows) + 1,
                "loss": rolling_history.history["loss"][epoch],
                "mae": rolling_history.history["mae"][epoch]
            })

        # Последние 168 часов перед прогнозируемыми сутками.
        input_values = available_history.iloc[
            -WINDOW_SIZE:
        ].to_numpy().reshape(-1, 1)

        scaled_input = scaler.transform(input_values)

        X_input = scaled_input.reshape(
            1,
            WINDOW_SIZE,
            1
        ).astype(np.float32)

        # Прогнозируем 24 часа.
        scaled_prediction = model.predict(
            X_input,
            verbose=0
        )

        prediction = scaler.inverse_transform(
            scaled_prediction.reshape(-1, 1)
        ).flatten()

        # Фактические значения используются только после прогноза.
        actual_day = series.loc[
            (series.index >= day_start)
            & (series.index <= day_end)
        ]

        if len(actual_day) != FORECAST_HORIZON:
            print(
                f"Пропуск {day.date()}: "
                f"получено {len(actual_day)} часов вместо 24."
            )
            continue

        predictions.extend(prediction.tolist())
        actual_values.extend(actual_day.to_numpy().tolist())
        timestamps.extend(actual_day.index.tolist())

        if day_number % 30 == 0 or day_number == len(test_days):
            print(
                f"Обработано дней: {day_number}/{len(test_days)}"
            )

    if not predictions:
        raise ValueError(
            "Прогнозы не получены. Проверь даты и доступный период данных."
        )

    comparison = pd.DataFrame({
        "datetime": timestamps,
        "actual": actual_values,
        "predicted": predictions
    })

    comparison = comparison.sort_values(
        "datetime"
    ).reset_index(drop=True)

    training_history = pd.DataFrame(history_rows)

    return model, comparison, training_history


# ============================================================
# METRICS
# ============================================================

def evaluate(comparison):
    actual = comparison["actual"].to_numpy()
    predicted = comparison["predicted"].to_numpy()

    mae = mean_absolute_error(actual, predicted)

    rmse = np.sqrt(
        mean_squared_error(actual, predicted)
    )

    nonzero = actual != 0

    if nonzero.any():
        mape = np.mean(
            np.abs(
                (actual[nonzero] - predicted[nonzero])
                / actual[nonzero]
            )
        ) * 100
    else:
        mape = np.nan

    print("\n" + "=" * 50)
    print("SLIDELEARN — TEST METRICS")
    print("=" * 50)

    print(f"MAE:  {mae:.2f} MWh")
    print(f"RMSE: {rmse:.2f} MWh")
    print(f"MAPE: {mape:.2f}%")
    print(f"Количество часов: {len(comparison):,}")

    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape
    }


# ============================================================
# PLOTS AND RESULTS
# ============================================================

def save_results(comparison, training_history, metrics):
    # Таблица прогнозов.
    comparison_path = os.path.join(
        RESULTS_DIR,
        "slidelearn_comparison.csv"
    )

    comparison.to_csv(
        comparison_path,
        index=False,
        encoding="utf-8-sig"
    )

    # Метрики в отдельный CSV.
    metrics_path = os.path.join(
        RESULTS_DIR,
        "slidelearn_metrics.csv"
    )

    pd.DataFrame([metrics]).to_csv(
        metrics_path,
        index=False,
        encoding="utf-8-sig"
    )

    # История обучения в CSV.
    history_path = os.path.join(
        RESULTS_DIR,
        "slidelearn_training_history.csv"
    )

    training_history.to_csv(
        history_path,
        index=False,
        encoding="utf-8-sig"
    )

    # График фактического и прогнозируемого потребления.
    fig, ax = plt.subplots(figsize=(15, 6))

    ax.plot(
        comparison["datetime"],
        comparison["actual"],
        label="Фактическое потребление",
        linewidth=1
    )

    ax.plot(
        comparison["datetime"],
        comparison["predicted"],
        label="Прогноз SlideLearn",
        linewidth=1
    )

    ax.set_title("SlideLearn: фактическое и прогнозируемое потребление")
    ax.set_xlabel("Дата")
    ax.set_ylabel("Потребление, МВт·ч")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    prediction_plot = os.path.join(
        RESULTS_DIR,
        "slidelearn_predictions.png"
    )

    fig.savefig(
        prediction_plot,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)

    # График истории обучения.
    if not training_history.empty:
        fig, axes = plt.subplots(
            2,
            1,
            figsize=(12, 8),
            sharex=True
        )

        epochs = np.arange(
            1,
            len(training_history) + 1
        )

        axes[0].plot(
            epochs,
            training_history["loss"],
            label="Training MSE"
        )

        axes[0].set_title("История обучения: MSE")
        axes[0].set_ylabel("MSE")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(
            epochs,
            training_history["mae"],
            label="Training MAE"
        )

        axes[1].set_title("История обучения: MAE")
        axes[1].set_ylabel("MAE")
        axes[1].set_xlabel("Эпоха начального обучения / шаг дообучения")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        initial_count = int(
            (training_history["stage"] == "initial").sum()
        )

        if initial_count < len(training_history):
            for ax in axes:
                ax.axvline(
                    initial_count + 0.5,
                    linestyle="--",
                    alpha=0.7
                )

        fig.tight_layout()

        history_plot = os.path.join(
            RESULTS_DIR,
            "slidelearn_training_history.png"
        )

        fig.savefig(
            history_plot,
            dpi=150,
            bbox_inches="tight"
        )

        plt.close(fig)

    print("\nФайлы сохранены:")
    print(f"Прогнозы: {comparison_path}")
    print(f"Метрики: {metrics_path}")
    print(f"История обучения: {history_path}")
    print(f"График прогнозов: {prediction_plot}")


# ============================================================
# MAIN
# ============================================================

def main():
    series = load_data()

    model, comparison, training_history = rolling_training(
        series
    )

    metrics = evaluate(comparison)

    save_results(
        comparison,
        training_history,
        metrics
    )

    model_path = os.path.join(
        MODEL_DIR,
        "slidelearn_cnn.keras"
    )

    model.save(model_path)

    print(f"\nМодель сохранена: {model_path}")
    print("SlideLearn завершил работу.")


if __name__ == "__main__":
    main()
