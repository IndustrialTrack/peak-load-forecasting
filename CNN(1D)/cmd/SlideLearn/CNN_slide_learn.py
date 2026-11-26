"""Суточный прогноз CNN: расширяющаяся история, без доступа к будущему.

Запуск: py -3.12 CNN_slide_learn.py
Каждый день после первого веса дообучаются на всей доступной истории.
Годовой эксперимент может быть длительным; импорт модуля обучение не запускает.
"""
from pathlib import Path
import pickle
import re

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR.parents[2] / "datasets" / "electricity_75.csv"
RESULTS_DIR = BASE_DIR / "results" / "2026"
MODEL_DIR = BASE_DIR / "models" / "2026"

TRAIN_START = "2021-01-01"
TEST_START = "2026-01-01"
TEST_END = "2026-12-31"
WINDOW_SIZE = 168
FORECAST_HORIZON = 24
INITIAL_EPOCHS = 50
ROLLING_EPOCHS = 1
BATCH_SIZE = 32
RANDOM_SEED = 42


def parse_numeric(value):
    if pd.isna(value):
        return np.nan
    text = re.sub(r"[^0-9.\-]", "", str(value).replace(",", "."))
    try:
        return float(text)
    except ValueError:
        return np.nan


def load_data(path):
    """Не интерполируем факты: заполнение из будущего создаёт утечку."""
    try:
        df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(path, sep=None, engine="python", encoding="cp1251")
    df.columns = df.columns.str.strip().str.lower()
    required = {"date", "hour", "actual_consumption"}
    if not required.issubset(df.columns):
        raise ValueError(f"В CSV отсутствуют столбцы: {required - set(df.columns)}")
    if "region_name" in df and df["region_name"].nunique() > 1:
        raise ValueError("CSV должен содержать один регион.")
    hours = pd.to_numeric(df["hour"], errors="raise")
    if not (hours.between(0, 23) & (hours % 1 == 0)).all():
        raise ValueError("Часы должны быть целыми числами от 0 до 23.")
    df.index = pd.DatetimeIndex(
        pd.to_datetime(df["date"], errors="raise") + pd.to_timedelta(hours, unit="h"),
        name="datetime",
    )
    columns = ["actual_consumption"]
    if "planned_consumption" in df:
        columns.append("planned_consumption")
    df = df[columns].map(parse_numeric).sort_index(kind="stable")
    df = df.loc[TRAIN_START:TEST_END]
    # Как в исходной CNN, используем первую запись часа, а не позднюю ревизию.
    duplicates = df.index.duplicated(keep="first")
    if duplicates.any():
        print(f"Повторяющихся часов: {duplicates.sum()}; оставлена первая запись.")
        df = df.loc[~duplicates]
    if df.empty or df.index.hasnans:
        raise ValueError("Пустой ряд или некорректные даты в CSV.")
    expected = pd.date_range(df.index.min(), df.index.max(), freq="h")
    if not df.index.equals(expected):
        raise ValueError("В почасовом ряду есть пропуски; исправьте исходный CSV.")
    historical = df.index < pd.Timestamp(TEST_START)
    # Только прошлое и только короткие пропуски обучающего периода.
    df.loc[historical, "actual_consumption"] = df.loc[
        historical, "actual_consumption"
    ].ffill(limit=2)
    if not np.isfinite(df["actual_consumption"]).all():
        raise ValueError("Фактическое потребление содержит пропуски или бесконечности.")
    return df


def create_windows(values):
    """Каждая цель полностью находится внутри переданного прошлого."""
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    size = WINDOW_SIZE + FORECAST_HORIZON
    if len(values) < size:
        raise ValueError(f"Для обучения требуется хотя бы {size} часов.")
    windows = np.lib.stride_tricks.sliding_window_view(values, size)
    return windows[:, :WINDOW_SIZE, None].copy(), windows[:, WINDOW_SIZE:].copy()


def create_model():
    import tensorflow as tf
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Input, Conv1D, Flatten, Dense, Dropout

    model = Sequential([
        Input(shape=(WINDOW_SIZE, 1)),
        Conv1D(32, 3, padding="same", activation="relu"),
        Conv1D(64, 3, padding="same", activation="relu"),
        Conv1D(64, 3, padding="same", activation="relu"),
        Flatten(), Dense(128, activation="relu"), Dropout(0.2),
        Dense(FORECAST_HORIZON),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss="mse")
    return model


def calculate_metrics(actual, predicted):
    """Ошибки по часам; Top-3 — средняя доля совпавших пиковых часов суток.

    При равных значениях приоритет у более раннего часа. MAPE исключает нули.
    """
    actual = np.asarray(actual, dtype=float).reshape(-1, FORECAST_HORIZON)
    predicted = np.asarray(predicted, dtype=float).reshape(actual.shape)
    if not actual.size or not (np.isfinite(actual).all() and np.isfinite(predicted).all()):
        raise ValueError("Для метрик нужны полные сутки с конечными значениями.")
    error = actual - predicted
    nonzero = actual != 0
    actual_top = np.argsort(-actual, axis=1, kind="stable")[:, :3]
    predicted_top = np.argsort(-predicted, axis=1, kind="stable")[:, :3]
    hits = [len(set(a) & set(p)) / 3 for a, p in zip(actual_top, predicted_top)]
    return {
        "MAE": float(np.mean(np.abs(error))),
        "RMSE": float(np.sqrt(np.mean(error ** 2))),
        "MAPE": float(np.mean(np.abs(error[nonzero] / actual[nonzero])) * 100)
        if nonzero.any() else np.nan,
        "top3_hit_rate": float(np.mean(hits)),
    }


def format_metrics(metrics):
    return (f"MAE={metrics['MAE']:.2f} МВт·ч | "
            f"RMSE={metrics['RMSE']:.2f} МВт·ч | "
            f"MAPE={metrics['MAPE']:.2f}% | "
            f"Top-3 Hit Rate={metrics['top3_hit_rate']:.2%}")


def walk_forward(df):
    """Фиксированная CNN и дообучаемая CNN начинают с одинаковых весов.

    Обе получают одни и те же последние 168 фактических часов. Только
    скользящая CNN обновляет веса. Масштабирование фиксируется до теста.
    """
    import tensorflow as tf

    tf.keras.utils.set_random_seed(RANDOM_SEED)
    start = pd.Timestamp(TEST_START)
    initial = df.loc[(df.index >= pd.Timestamp(TRAIN_START)) & (df.index < start)]
    if initial.empty or initial.index[-1] != start - pd.Timedelta(hours=1):
        raise ValueError("Обучающая история должна заканчиваться 31.12.2025 23:00.")
    last_complete = (df.index.max() + pd.Timedelta(hours=1)).normalize() - pd.Timedelta(days=1)
    end = min(pd.Timestamp(TEST_END), last_complete)
    if end < start:
        raise ValueError("В CSV нет полных суток тестового периода 2026 года.")
    scaler = StandardScaler().fit(initial[["actual_consumption"]].to_numpy())
    scaled = scaler.transform(df[["actual_consumption"]].to_numpy()).ravel()
    initial_count = len(initial)
    X, y = create_windows(scaled[:initial_count])
    rolling = create_model()
    rolling.fit(X, y, epochs=INITIAL_EPOCHS, batch_size=BATCH_SIZE, shuffle=False, verbose=2)
    baseline = tf.keras.models.clone_model(rolling)
    baseline.set_weights(rolling.get_weights())
    frames = []
    for day in pd.date_range(start, end, freq="D"):
        # Исключаем весь прогнозируемый день, включая его целевые значения.
        cutoff = df.index.searchsorted(day)
        if day > start:
            X, y = create_windows(scaled[:cutoff])
            rolling.fit(X, y, epochs=ROLLING_EPOCHS, batch_size=BATCH_SIZE,
                        shuffle=False, verbose=0)
        context = scaled[cutoff - WINDOW_SIZE:cutoff].reshape(1, WINDOW_SIZE, 1)
        frame = df.iloc[cutoff:cutoff + FORECAST_HORIZON].copy()
        for name, model in [("cnn_prediction", rolling), ("baseline_prediction", baseline)]:
            prediction = np.asarray(model(context, training=False)).reshape(-1, 1)
            frame[name] = scaler.inverse_transform(prediction).ravel()
        frame["train_end"] = day - pd.Timedelta(hours=1)
        frames.append(frame)
        metrics = calculate_metrics(frame.actual_consumption, frame.cnn_prediction)
        print(f"{day.date()} | {format_metrics(metrics)}", flush=True)
    return pd.concat(frames), rolling, scaler


def save_results(comparison, df, model, scaler):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(RESULTS_DIR / "rolling_test_comparison.csv", encoding="utf-8-sig")
    actual = comparison.actual_consumption
    rolling = calculate_metrics(actual, comparison.cnn_prediction)
    baseline = calculate_metrics(actual, comparison.baseline_prediction)
    lines = [
        f"Настройки: TRAIN_START={TRAIN_START}, TEST_START={TEST_START}, TEST_END={TEST_END}",
        f"Начало доступной истории: {df.index.min()}",
        f"Оценено: {comparison.index.min()} — {comparison.index.max()}; {len(comparison)} часов",
        "Оцениваются только полные сутки. Недоступные даты не прогнозируются.",
        "MAE/RMSE/MAPE рассчитаны по часам; MAPE исключает фактические нули.",
        "Top-3 Hit Rate: средняя доля совпавших трёх пиковых часов в сутках.",
        "CNN со скользящим обучением: " + format_metrics(rolling),
        "CNN без дообучения (те же начальные веса): " + format_metrics(baseline),
    ]
    for key in ("MAE", "RMSE", "MAPE"):
        delta = baseline[key] - rolling[key]
        lines.append(f"Улучшение {key} (базовая минус скользящая; >0 лучше): {delta:.4f}")
    lines.append(f"Изменение Top-3 (>0 лучше): {rolling['top3_hit_rate'] - baseline['top3_hit_rate']:+.2%}")
    if "planned_consumption" in comparison:
        # Сравниваем только общие полные сутки, чтобы Top-3 был сопоставим.
        valid_days = np.isfinite(comparison.planned_consumption.to_numpy().reshape(-1, 24)).all(axis=1)
        paired = comparison.iloc[np.repeat(valid_days, 24)]
        lines.append(f"Сравнение с planned_consumption: {len(paired) // 24} полных суток.")
        if not paired.empty:
            for label, column in [("План CSV", "planned_consumption"), ("CNN на тех же сутках", "cnn_prediction")]:
                lines.append(label + ": " + format_metrics(calculate_metrics(paired.actual_consumption, paired[column])))
    lines.append("\nСуточные метрики CNN:")
    for day, frame in comparison.groupby(comparison.index.normalize()):
        lines.append(f"{day.date()} | " + format_metrics(calculate_metrics(frame.actual_consumption, frame.cnn_prediction)))
    (RESULTS_DIR / "resultMetric.txt").write_text("\n".join(lines), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(18, 6))
    for column, label in [("planned_consumption", "Планируемое потребление"), ("cnn_prediction", "CNN: скользящее обучение")]:
        if column in comparison:
            ax.plot(comparison.index, comparison[column], label=label, linewidth=0.7, alpha=0.8)
    end = pd.Timestamp(TEST_END) + pd.Timedelta(days=1)
    available_end = comparison.index.max() + pd.Timedelta(hours=1)
    if available_end < end:
        ax.axvspan(available_end, end, color="gray", alpha=0.15)
        ax.text(0.99, 0.02, "Серая область: нет полных фактических суток",
                transform=ax.transAxes, ha="right", fontsize=9, color="dimgray")
    ax.set(xlim=(pd.Timestamp(TEST_START), end), title="Прогноз потребления на 2026 год",
           xlabel="Дата", ylabel="Потребление, МВт·ч")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "rolling_test_predictions.png", dpi=150)
    plt.close(fig)
    model.save(MODEL_DIR / "rolling_cnn.keras")
    with (MODEL_DIR / "scaler.pkl").open("wb") as file:
        pickle.dump(scaler, file)
    print(f"Результаты сохранены: {RESULTS_DIR}")


def main():
    df = load_data(DATA_PATH)
    print(f"Доступные данные: {df.index.min()} — {df.index.max()}")
    comparison, model, scaler = walk_forward(df)
    save_results(comparison, df, model, scaler)


if __name__ == "__main__":
    main()
