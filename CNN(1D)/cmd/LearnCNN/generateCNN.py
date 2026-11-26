import os
import re
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf


# ============================================================
# PATHS AND SETTINGS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Скрипт предполагается разместить рядом с исходным LearnCNN.py:
# .../CNN(1D)/cmd/LearnCNN/plot_2026.py
DATA_PATH = os.path.abspath(
    os.path.join(BASE_DIR, "../../..", "datasets", "electricity_75.csv")
)

MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "cnn_168.keras")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

RESULTS_DIR = os.path.join(BASE_DIR, "results")
PLOT_PATH = os.path.join(RESULTS_DIR, "comparison_2026.png")
CSV_PATH = os.path.join(RESULTS_DIR, "comparison_2026.csv")

YEAR_START = pd.Timestamp("2026-01-01 00:00:00")
YEAR_END = pd.Timestamp("2026-12-31 23:00:00")

WINDOW_SIZE = 168
FORECAST_HORIZON = 24


# ============================================================
# CSV PARSING
# ============================================================

def parse_numeric(value):
    """Converts values such as '4 051 МВт*ч' to float."""
    if pd.isna(value):
        return np.nan

    value = str(value).strip()
    if not value:
        return np.nan

    value = value.replace("\xa0", " ").replace(" ", "")
    value = value.replace(",", ".")
    value = value.replace("МВт*ч", "")
    value = value.replace("МВт·ч", "")
    value = re.sub(r"[^0-9.\-]", "", value)

    if value in ("", "-", "."):
        return np.nan

    try:
        return float(value)
    except ValueError:
        return np.nan


def load_data():
    if not os.path.isfile(DATA_PATH):
        raise FileNotFoundError(
            f"CSV не найден:\n{DATA_PATH}\n\n"
            "Проверь расположение plot_2026.py: он должен находиться "
            "в той же папке, что и исходный скрипт обучения."
        )

    try:
        raw = pd.read_csv(
            DATA_PATH, sep=None, engine="python", encoding="utf-8-sig"
        )
    except UnicodeDecodeError:
        raw = pd.read_csv(
            DATA_PATH, sep=None, engine="python", encoding="cp1251"
        )

    raw.columns = [
        str(column).replace("\ufeff", "").strip().lower()
        for column in raw.columns
    ]

    required = {"date", "hour", "actual_consumption", "planned_consumption"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(
            f"В CSV отсутствуют столбцы: {', '.join(sorted(missing))}"
        )

    hour = raw["hour"].astype(str).str.extract(r"(\d+)", expand=False)
    raw["hour"] = pd.to_numeric(hour, errors="coerce")
    raw["datetime"] = pd.to_datetime(
        raw["date"].astype(str).str.strip()
        + " "
        + raw["hour"].astype("Int64").astype(str).str.zfill(2),
        format="%Y-%m-%d %H",
        errors="coerce",
    )

    raw["actual_consumption"] = raw["actual_consumption"].apply(parse_numeric)
    raw["planned_consumption"] = raw["planned_consumption"].apply(parse_numeric)

    raw = raw.dropna(subset=["datetime"]).sort_values("datetime")
    raw = raw.drop_duplicates(subset=["datetime"], keep="first")
    data = raw.set_index("datetime")[
        ["actual_consumption", "planned_consumption"]
    ].sort_index()

    # Восстанавливаем часовую сетку, как в исходном скрипте.
    full_index = pd.date_range(data.index.min(), data.index.max(), freq="h")
    data = data.reindex(full_index)
    data.index.name = "datetime"

    # Интерполируем только короткие пропуски фактического потребления,
    # необходимые для формирования входного окна CNN.
    data["actual_consumption"] = data["actual_consumption"].interpolate(
        method="time", limit=2, limit_area="inside"
    )

    return data


# ============================================================
# LOAD ALREADY TRAINED MODEL AND SCALER
# ============================================================

def load_trained_objects():
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(f"Не найдена обученная модель:\n{MODEL_PATH}")

    if not os.path.isfile(SCALER_PATH):
        raise FileNotFoundError(f"Не найден scaler:\n{SCALER_PATH}")

    # compile=False: для инференса состояние optimizer и метрики не нужны.
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    with open(SCALER_PATH, "rb") as file:
        scaler = pickle.load(file)

    print(f"Загружена модель: {MODEL_PATH}")
    print(f"Загружен scaler:  {SCALER_PATH}")
    return model, scaler


# ============================================================
# BUILD DAILY PREDICTIONS FOR 2026
# ============================================================

def make_2026_comparison(data, model, scaler):
    if "planned_consumption" not in data.columns:
        raise ValueError("В CSV нет столбца planned_consumption.")

    all_hours = pd.date_range(YEAR_START, YEAR_END, freq="h")
    result = pd.DataFrame(index=all_hours)
    result.index.name = "datetime"

    # План берётся непосредственно из CSV, без пересчёта.
    result["planned_consumption"] = data["planned_consumption"].reindex(all_hours)

    # Каждый день CNN прогнозирует следующие 24 часа по предыдущим 168
    # часам фактического потребления. Модель не переобучается.
    daily_starts = pd.date_range(YEAR_START.normalize(), YEAR_END.normalize(), freq="D")
    predicted_parts = []

    for day_start in daily_starts:
        history_start = day_start - pd.Timedelta(hours=WINDOW_SIZE)
        history_end = day_start - pd.Timedelta(hours=1)
        history = data.loc[history_start:history_end, "actual_consumption"]

        if len(history) != WINDOW_SIZE or history.isna().any():
            print(
                f"Пропуск прогноза на {day_start.date()}: "
                "нет полного окна из 168 исторических часов."
            )
            continue

        scaled_history = scaler.transform(
            history.to_numpy(dtype=float).reshape(-1, 1)
        ).astype(np.float32)

        model_input = scaled_history.reshape(1, WINDOW_SIZE, 1)
        prediction_scaled = model.predict(model_input, verbose=0)[0]

        prediction = scaler.inverse_transform(
            np.asarray(prediction_scaled).reshape(-1, 1)
        ).flatten()

        forecast_hours = pd.date_range(
            day_start, periods=FORECAST_HORIZON, freq="h"
        )

        predicted_parts.append(
            pd.Series(prediction, index=forecast_hours)
        )

    if not predicted_parts:
        raise ValueError(
            "Не удалось сформировать прогнозы за 2026 год. "
            "Проверь, что в CSV есть фактическое потребление как минимум "
            "за последние 168 часов 2025 года и далее."
        )

    predicted = pd.concat(predicted_parts)
    # В обычном случае каждый час попадает ровно в один суточный прогноз.
    result["cnn_prediction"] = predicted.reindex(all_hours)

    # Оставляем для графика только часы, где доступны обе сравниваемые линии.
    plot_data = result.dropna(
        subset=["planned_consumption", "cnn_prediction"]
    )

    if plot_data.empty:
        raise ValueError(
            "Нет часов, для которых одновременно доступны план и прогноз CNN."
        )

    return result, plot_data


# ============================================================
# SAVE TABLE AND DRAW ONLY TWO LINES
# ============================================================

def save_results(result, plot_data):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    result.to_csv(CSV_PATH, encoding="utf-8-sig")
    print(f"Таблица сохранена: {CSV_PATH}")

    fig, ax = plt.subplots(figsize=(16, 7))

    ax.plot(
        plot_data.index,
        plot_data["planned_consumption"],
        label="Planned consumption",
        linewidth=1.0,
    )

    ax.plot(
        plot_data.index,
        plot_data["cnn_prediction"],
        label="CNN prediction (Slide Learn)",
        linewidth=1.0,
    )

    ax.set_title("Planned consumption vs CNN prediction (2026)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Consumption, MWh")
    ax.set_xlim(YEAR_START, YEAR_END)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()

    fig.savefig(PLOT_PATH, dpi=160)
    plt.close(fig)

    print(f"График сохранён: {PLOT_PATH}")
    print(
        f"Период графика: {plot_data.index.min()} — "
        f"{plot_data.index.max()}"
    )
    print(f"Количество общих часов: {len(plot_data):,}")


def main():
    print("=" * 70)
    print("2026: PLANNED CONSUMPTION VS CNN PREDICTION")
    print("Используется уже обученная модель. Переобучения нет.")
    print("=" * 70)

    data = load_data()
    model, scaler = load_trained_objects()
    result, plot_data = make_2026_comparison(data, model, scaler)
    save_results(result, plot_data)


if __name__ == "__main__":
    main()