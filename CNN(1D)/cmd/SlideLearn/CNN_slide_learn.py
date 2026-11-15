import os
import re
import pickle
import argparse
import gc

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
DATA_PATH = os.path.join(BASE_DIR, "../../..", "datasets", "electricity_75.csv")

MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PARAMETERS_DIR = os.path.join(RESULTS_DIR, "model_parameters")

MODEL_PATH = os.path.join(MODEL_DIR, "cnn_168.keras")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

# Для прогноза на 01.01.2026 обучение заканчивается 31.12.2025.
TRAIN_START = "2021-01-01"
TEST_START = "2026-01-01"
TEST_END = "2026-12-31"

WINDOW_SIZE = 168
FORECAST_HORIZON = 24

EPOCHS = 100
BATCH_SIZE = 32
VALIDATION_SHARE = 0.10

LEARNING_RATE = 0.001
DROPOUT_RATE = 0.20
EARLY_STOPPING_PATIENCE = 7
REDUCE_LR_PATIENCE = 3
REDUCE_LR_FACTOR = 0.5
MIN_LEARNING_RATE = 1e-6

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# ============================================================
# DIRECTORIES AND DATA PARSING
# ============================================================

def ensure_directories():
    for directory in (MODEL_DIR, RESULTS_DIR, PARAMETERS_DIR):
        os.makedirs(directory, exist_ok=True)


def parse_numeric(value):
    if pd.isna(value):
        return np.nan

    value = str(value).strip()
    if not value:
        return np.nan

    value = value.replace("\xa0", " ").replace(",", ".")
    for unit in ("руб./МВт*ч", "МВт*ч", "МВт·ч"):
        value = value.replace(unit, "")

    value = value.replace(" ", "")
    value = re.sub(r"[^0-9.\-]", "", value)

    if value in ("", "-", "."):
        return np.nan

    try:
        return float(value)
    except ValueError:
        return np.nan


def load_data(path):
    print("=" * 70)
    print("ЗАГРУЗКА И ПРЕДОБРАБОТКА ДАННЫХ")
    print("=" * 70)

    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV-файл не найден: {path}")

    try:
        df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(path, sep=None, engine="python", encoding="cp1251")

    df.columns = [
        str(column).replace("\ufeff", "").strip().lower()
        for column in df.columns
    ]

    required = ["date", "hour", "actual_consumption"]
    for column in required:
        if column not in df.columns:
            raise ValueError(
                f"В CSV отсутствует обязательный столбец '{column}'. "
                f"Найдены столбцы: {list(df.columns)}"
            )

    hour_values = df["hour"].astype(str).str.extract(r"(\d+)", expand=False)
    df["hour"] = pd.to_numeric(hour_values, errors="coerce")
    df["date"] = df["date"].astype(str).str.strip()

    df["datetime"] = pd.to_datetime(
        df["date"] + " " + df["hour"].fillna(-1).astype(int).astype(str).str.zfill(2),
        format="%Y-%m-%d %H",
        errors="coerce",
    )

    df["actual_consumption"] = df["actual_consumption"].apply(parse_numeric)

    if "planned_consumption" in df.columns:
        df["planned_consumption"] = df["planned_consumption"].apply(parse_numeric)

    df = df.dropna(subset=["datetime", "actual_consumption"])
    df = df.sort_values("datetime")

    duplicates = int(df["datetime"].duplicated().sum())
    if duplicates:
        print(f"Предупреждение: найдено повторяющихся часов: {duplicates}. Оставлена первая запись.")
        df = df.drop_duplicates(subset=["datetime"], keep="first")

    columns = ["datetime", "actual_consumption"]
    if "planned_consumption" in df.columns:
        columns.append("planned_consumption")

    result = df[columns].set_index("datetime").sort_index()

    if result.empty:
        raise ValueError("После предобработки датасет оказался пустым.")

    full_index = pd.date_range(result.index.min(), result.index.max(), freq="h")
    missing = full_index.difference(result.index)

    print(f"Количество строк до восстановления пропусков: {len(result):,}")
    print(f"Период: {result.index.min()} — {result.index.max()}")
    print(f"Пропущено часовых отметок: {len(missing):,}")

    if len(missing):
        result = result.reindex(full_index)
        result["actual_consumption"] = result["actual_consumption"].interpolate(
            method="time", limit=2, limit_area="inside"
        )

        unresolved = result.index[result["actual_consumption"].isna()]
        if len(unresolved):
            examples = ", ".join(str(x) for x in unresolved[:10])
            raise ValueError(
                "Обнаружены пропуски фактического потребления, которые нельзя "
                "восстановить интерполяцией максимум двух соседних часов. "
                f"Примеры: {examples}"
            )

        print(f"Восстановлено интерполяцией коротких пропусков: {len(missing):,}")

    print(f"Итоговое количество часовых строк: {len(result):,}")
    print("Данные успешно загружены.\\n")
    return result


# ============================================================
# WINDOW GENERATION AND MODEL
# ============================================================

def create_windows(values, window_size=WINDOW_SIZE, horizon=FORECAST_HORIZON):
    X, y = [], []
    max_start = len(values) - window_size - horizon + 1

    for start in range(max_start):
        end = start + window_size
        X.append(values[start:end])
        y.append(values[end:end + horizon])

    X = np.asarray(X, dtype=np.float32)[..., np.newaxis]
    y = np.asarray(y, dtype=np.float32)
    return X, y


def create_model():
    model = Sequential([
        Input(shape=(WINDOW_SIZE, 1)),
        Conv1D(filters=32, kernel_size=3, padding="same", activation="relu"),
        Conv1D(filters=64, kernel_size=3, padding="same", activation="relu"),
        Conv1D(filters=64, kernel_size=3, padding="same", activation="relu"),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(DROPOUT_RATE),
        Dense(FORECAST_HORIZON),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="mse",
        metrics=["mae"],
    )
    return model


# ============================================================
# MODEL TRAINING AND PARAMETER LOG
# ============================================================

def train_model(train_df, target_date=None, verbose=0):
    values = train_df["actual_consumption"].to_numpy(dtype=float).reshape(-1, 1)

    if len(values) < WINDOW_SIZE + FORECAST_HORIZON + 10:
        raise ValueError(
            f"Недостаточно данных для обучения: {len(values)} часов."
        )

    # Scaler обучается только на истории, доступной до даты прогноза.
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(values).flatten()

    X, y = create_windows(scaled_values)

    if len(X) < 2:
        raise ValueError("Недостаточно обучающих окон для train/validation.")

    split_index = int(len(X) * (1.0 - VALIDATION_SHARE))
    split_index = min(max(split_index, 1), len(X) - 1)

    X_train, y_train = X[:split_index], y[:split_index]
    X_val, y_val = X[split_index:], y[split_index:]

    model = create_model()

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=REDUCE_LR_FACTOR,
            patience=REDUCE_LR_PATIENCE,
            min_lr=MIN_LEARNING_RATE,
        ),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=False,
        callbacks=callbacks,
        verbose=verbose,
    )

    best_epoch = int(np.argmin(history.history["val_loss"])) + 1

    parameters = {
        "forecast_date": str(pd.Timestamp(target_date).date()) if target_date is not None else "",
        "train_start": str(train_df.index.min()),
        "train_end": str(train_df.index.max()),
        "train_rows_hours": len(train_df),
        "train_samples": len(X_train),
        "validation_samples": len(X_val),
        "window_size_hours": WINDOW_SIZE,
        "forecast_horizon_hours": FORECAST_HORIZON,
        "epochs_max": EPOCHS,
        "epochs_actual": len(history.history["loss"]),
        "batch_size": BATCH_SIZE,
        "validation_share": VALIDATION_SHARE,
        "optimizer": model.optimizer.__class__.__name__,
        "learning_rate_configured": LEARNING_RATE,
        "learning_rate_final": float(tf.keras.backend.get_value(model.optimizer.learning_rate)),
        "loss": str(model.loss),
        "metrics": "mae",
        "best_epoch": best_epoch,
        "best_val_loss_mse_scaled": float(np.min(history.history["val_loss"])),
        "best_val_mae_scaled": float(history.history["val_mae"][best_epoch - 1]),
        "scaler_mean": float(scaler.mean_[0]),
        "scaler_scale": float(scaler.scale_[0]),
        "architecture": "Conv1D(32,3)-Conv1D(64,3)-Conv1D(64,3)-Flatten-Dense(128)-Dropout(0.2)-Dense(24)",
    }

    return model, scaler, history, parameters


def save_model_configuration():
    path = os.path.join(PARAMETERS_DIR, "model_configuration.txt")
    lines = [
        "КОНФИГУРАЦИЯ CNN 1D",
        "=" * 65,
        f"Размер входного окна: {WINDOW_SIZE} часов (7 суток)",
        f"Горизонт прогноза: {FORECAST_HORIZON} часа",
        "Вход: один признак — фактическое потребление",
        "Архитектура: Conv1D(32, kernel=3) -> Conv1D(64, kernel=3) -> "
        "Conv1D(64, kernel=3) -> Flatten -> Dense(128) -> Dropout(0.2) -> Dense(24)",
        f"Оптимизатор: Adam, начальный learning rate={LEARNING_RATE}",
        "Функция потерь: MSE",
        "Дополнительная метрика обучения: MAE",
        f"Максимум эпох: {EPOCHS}",
        f"Размер пакета: {BATCH_SIZE}",
        f"Доля валидации: {VALIDATION_SHARE:.0%} последних временных окон",
        f"EarlyStopping: patience={EARLY_STOPPING_PATIENCE}, restore_best_weights=True",
        f"ReduceLROnPlateau: factor={REDUCE_LR_FACTOR}, patience={REDUCE_LR_PATIENCE}, min_lr={MIN_LEARNING_RATE}",
        "Перемешивание временных окон: отключено (shuffle=False)",
        "Масштабирование: StandardScaler, заново обучается только на доступной истории",
        "",
        "Параметры архитектуры и обучения фиксированы для всех дат. "
        "Фактическое число эпох и итоговый learning rate могут меняться.",
        "Веса нейронной сети обучаются заново для каждой даты прогноза.",
    ]
    with open(path, "w", encoding="utf-8") as file:
        file.write("\\n".join(lines))
    print(f"Общая конфигурация сохранена: {path}")


# ============================================================
# FORECAST AND METRICS
# ============================================================

def predict_next_day(model, scaler, df, target_date):
    target_start = pd.Timestamp(target_date).normalize()
    history_start = target_start - pd.Timedelta(hours=WINDOW_SIZE)
    history_end = target_start - pd.Timedelta(hours=1)

    history = df.loc[history_start:history_end, "actual_consumption"]
    if len(history) != WINDOW_SIZE:
        raise ValueError(
            f"Для {target_start.date()} ожидалось {WINDOW_SIZE} часов истории, "
            f"получено {len(history)}."
        )

    scaled_history = scaler.transform(history.to_numpy().reshape(-1, 1)).flatten()
    X = scaled_history.reshape(1, WINDOW_SIZE, 1)

    prediction_scaled = model.predict(X, verbose=0)[0]
    prediction = scaler.inverse_transform(prediction_scaled.reshape(-1, 1)).flatten()

    forecast_index = pd.date_range(target_start, periods=FORECAST_HORIZON, freq="h")
    return pd.DataFrame({
        "datetime": forecast_index,
        "predicted_consumption": prediction,
    })


def calculate_regression_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    valid = np.isfinite(actual) & np.isfinite(predicted)
    actual, predicted = actual[valid], predicted[valid]

    if len(actual) == 0:
        return {"MAE": np.nan, "RMSE": np.nan, "MAPE": np.nan, "n_hours": 0}

    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    nonzero = actual != 0
    mape = (
        np.mean(np.abs((actual[nonzero] - predicted[nonzero]) / actual[nonzero])) * 100
        if nonzero.any() else np.nan
    )
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "n_hours": len(actual)}


def calculate_peak_metrics(day_comparison):
    actual_top = set(
        day_comparison.nlargest(3, "actual_consumption")["datetime"]
    )
    predicted_top = set(
        day_comparison.nlargest(3, "predicted_consumption")["datetime"]
    )

    hits = len(actual_top & predicted_top)
    precision = hits / 3.0
    recall = hits / 3.0
    f1 = 2 * precision * recall / (precision + recall) if hits else 0.0

    return {
        "actual_top3_hours": ";".join(str(x) for x in sorted(actual_top)),
        "predicted_top3_hours": ";".join(str(x) for x in sorted(predicted_top)),
        "top3_hits": hits,
        "top3_hit_rate": hits / 3.0,
        "peak_precision": precision,
        "peak_recall": recall,
        "peak_f1": f1,
    }


# ============================================================
# EXPANDING-WINDOW EVALUATION
# ============================================================

def rolling_evaluation(df, start_date=TEST_START, end_date=TEST_END, save_last_model=True):
    """
    Для каждой даты:
      1. Обучение на TRAIN_START — 23:00 предыдущего дня.
      2. Новое обучение модели и scaler.
      3. Прогноз следующих 24 часов.
      4. Сравнение с фактическими значениями.

    Факты прогнозируемого дня попадают в обучение только на следующий день.
    """
    start = pd.Timestamp(start_date).normalize()
    requested_end = pd.Timestamp(end_date).normalize()

    available_last_hour = df.index.max()
    last_complete_day = available_last_hour.normalize()
    if available_last_hour.hour < 23:
        last_complete_day -= pd.Timedelta(days=1)

    end = min(requested_end, last_complete_day)
    if end < start:
        raise ValueError(
            f"Нет полных тестовых суток начиная с {start.date()}. "
            f"Последний час данных: {available_last_hour}."
        )

    dates = pd.date_range(start, end, freq="D")

    print("\\n" + "=" * 80)
    print("СКОЛЬЗЯЩЕЕ ОБУЧЕНИЕ С РАСШИРЯЮЩЕЙСЯ ОБУЧАЮЩЕЙ ВЫБОРКОЙ")
    print("=" * 80)
    print(f"Первая дата прогноза: {start.date()}")
    print(f"Последняя дата прогноза: {end.date()}")
    print(f"Количество дней: {len(dates)}")
    print("Для каждого дня модель обучается заново на доступной к этому моменту истории.")

    all_predictions = []
    daily_metrics = []
    peak_rows = []
    parameter_rows = []
    last_model = None
    last_scaler = None

    for number, target_date in enumerate(dates, start=1):
        target_start = target_date
        target_end = target_start + pd.Timedelta(days=1) - pd.Timedelta(hours=1)
        train_end = target_start - pd.Timedelta(hours=1)

        train_df = df.loc[pd.Timestamp(TRAIN_START):train_end].copy()
        actual_day = df.loc[target_start:target_end].copy()

        if len(actual_day) != FORECAST_HORIZON:
            print(
                f"[{number}/{len(dates)}] Пропуск {target_start.date()}: "
                f"найдено {len(actual_day)} фактических часов вместо 24."
            )
            continue

        if len(train_df) < WINDOW_SIZE + FORECAST_HORIZON + 10:
            print(f"[{number}/{len(dates)}] Недостаточно истории для {target_start.date()}.")
            continue

        print(
            f"\\n[{number}/{len(dates)}] Прогноз на {target_start.date()} | "
            f"обучение: {train_df.index.min()} — {train_df.index.max()} "
            f"({len(train_df):,} часов)"
        )

        if last_model is not None:
            del last_model
            tf.keras.backend.clear_session()
            gc.collect()

        model, scaler, history, params = train_model(
            train_df, target_date=target_start, verbose=0
        )

        forecast = predict_next_day(model, scaler, df, target_start)

        actual_comparison = actual_day[["actual_consumption"]].reset_index()
        actual_comparison = actual_comparison.rename(columns={"datetime": "datetime"})
        day_comparison = actual_comparison.merge(forecast, on="datetime", how="inner")

        if len(day_comparison) != FORECAST_HORIZON:
            raise ValueError(
                f"Для {target_start.date()} после объединения получено "
                f"{len(day_comparison)} часов вместо 24."
            )

        day_metrics = calculate_regression_metrics(
            day_comparison["actual_consumption"],
            day_comparison["predicted_consumption"],
        )
        day_metrics["date"] = str(target_start.date())
        daily_metrics.append(day_metrics)

        peak_metrics = calculate_peak_metrics(day_comparison)
        peak_metrics["date"] = str(target_start.date())
        peak_rows.append(peak_metrics)

        day_comparison["forecast_date"] = str(target_start.date())
        all_predictions.append(day_comparison)

        parameter_rows.append(params)

        print(
            f"MAE={day_metrics['MAE']:.2f} МВт·ч | "
            f"RMSE={day_metrics['RMSE']:.2f} МВт·ч | "
            f"MAPE={day_metrics['MAPE']:.2f}% | "
            f"Top-3 Hit Rate={peak_metrics['top3_hit_rate']:.2%}"
        )

        last_model, last_scaler = model, scaler

    if not all_predictions:
        raise ValueError("Не удалось сформировать ни одного полного суточного прогноза.")

    predictions_df = pd.concat(all_predictions, ignore_index=True)
    daily_metrics_df = pd.DataFrame(daily_metrics)
    peak_metrics_df = pd.DataFrame(peak_rows)
    parameters_df = pd.DataFrame(parameter_rows)

    predictions_path = os.path.join(RESULTS_DIR, "rolling_predictions_2026.csv")
    metrics_path = os.path.join(RESULTS_DIR, "rolling_daily_metrics.csv")
    peaks_path = os.path.join(RESULTS_DIR, "rolling_peak_metrics.csv")
    params_path = os.path.join(PARAMETERS_DIR, "daily_model_parameters.csv")

    predictions_df.to_csv(predictions_path, index=False, encoding="utf-8-sig")
    daily_metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    peak_metrics_df.to_csv(peaks_path, index=False, encoding="utf-8-sig")
    parameters_df.to_csv(params_path, index=False, encoding="utf-8-sig")

    overall = calculate_regression_metrics(
        predictions_df["actual_consumption"],
        predictions_df["predicted_consumption"],
    )
    mean_hit_rate = peak_metrics_df["top3_hit_rate"].mean()
    mean_peak_f1 = peak_metrics_df["peak_f1"].mean()

    summary = [
        "ИТОГОВАЯ ОЦЕНКА CNN 1D: EXPANDING-WINDOW",
        "=" * 70,
        f"Период прогнозирования: {predictions_df['datetime'].min()} — {predictions_df['datetime'].max()}",
        f"Количество прогнозированных часов: {len(predictions_df)}",
        f"Количество полных суток: {len(daily_metrics_df)}",
        f"Размер входного окна: {WINDOW_SIZE} часов",
        f"Горизонт прогноза: {FORECAST_HORIZON} часов",
        "",
        "ОБЩИЕ МЕТРИКИ ПО ВСЕМ ЧАСАМ",
        f"MAE:  {overall['MAE']:.3f} МВт·ч",
        f"RMSE: {overall['RMSE']:.3f} МВт·ч",
        f"MAPE: {overall['MAPE']:.3f}%",
        "",
        "ОЦЕНКА ПИКОВЫХ ЧАСОВ (Top-3 за каждые сутки)",
        f"Средний Top-3 Hit Rate: {mean_hit_rate:.3f} ({mean_hit_rate:.2%})",
        f"Средний Top-3 F1: {mean_peak_f1:.3f}",
        "",
        "Фактические данные прогнозируемого дня добавлялись в обучение только на следующий день.",
        f"Таблица прогнозов: {predictions_path}",
        f"Дневные метрики: {metrics_path}",
        f"Метрики пиков: {peaks_path}",
        f"Параметры моделей: {params_path}",
    ]

    summary_path = os.path.join(RESULTS_DIR, "rolling_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as file:
        file.write("\\n".join(summary))

    print("\\n" + "\\n".join(summary))

    if save_last_model and last_model is not None:
        last_model.save(MODEL_PATH)
        with open(SCALER_PATH, "wb") as file:
            pickle.dump(last_scaler, file)
        print(f"\\nПоследняя обученная модель сохранена: {MODEL_PATH}")
        print(f"Соответствующий scaler сохранён: {SCALER_PATH}")

    save_model_configuration()
    return predictions_df, daily_metrics_df, peak_metrics_df, parameters_df


# ============================================================
# PREDICTION WITH SAVED MODEL
# ============================================================

def load_saved_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Модель не найдена: {MODEL_PATH}")
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f"Scaler не найден: {SCALER_PATH}")

    model = tf.keras.models.load_model(MODEL_PATH)
    with open(SCALER_PATH, "rb") as file:
        scaler = pickle.load(file)
    return model, scaler


def forecast_saved_model(target_date):
    ensure_directories()
    df = load_data(DATA_PATH)
    model, scaler = load_saved_model()

    forecast = predict_next_day(model, scaler, df, target_date)
    print("\\nПРОГНОЗ НА", pd.Timestamp(target_date).date())
    print(forecast.to_string(index=False, formatters={
        "predicted_consumption": lambda value: f"{value:.2f}"
    }))

    peaks = forecast.nlargest(3, "predicted_consumption").sort_values("datetime")
    print("\\nТРИ ПРОГНОЗИРУЕМЫХ ПИКОВЫХ ЧАСА:")
    print(peaks.to_string(index=False))
    return forecast


# ============================================================
# COMMAND LINE
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Суточное прогнозирование электропотребления CNN 1D "
                    "с расширяющимся обучающим окном."
    )
    parser.add_argument(
        "--mode",
        choices=["rolling", "predict"],
        default="rolling",
        help="'rolling' — переобучение для каждого дня; 'predict' — прогноз сохранённой моделью.",
    )
    parser.add_argument(
        "--start-date",
        default=TEST_START,
        help="Первая дата тестового прогноза, по умолчанию 2026-01-01.",
    )
    parser.add_argument(
        "--end-date",
        default=TEST_END,
        help="Последняя дата тестового прогноза, по умолчанию 2026-12-31.",
    )
    parser.add_argument(
        "--date",
        help="Дата для режима predict, например 2026-11-14.",
    )

    args = parser.parse_args()
    ensure_directories()

    if args.mode == "rolling":
        df = load_data(DATA_PATH)
        rolling_evaluation(df, args.start_date, args.end_date)
    else:
        if not args.date:
            parser.error("Для --mode predict необходимо указать --date.")
        forecast_saved_model(args.date)


if __name__ == "__main__":
    main()
