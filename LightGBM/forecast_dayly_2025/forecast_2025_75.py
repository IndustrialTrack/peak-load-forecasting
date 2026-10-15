import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlforecast import MLForecast
from lightgbm import LGBMRegressor
from utilsforecast.preprocessing import fill_gaps
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# НАСТРОЙКИ
# ============================================================
CSV_FILE = "electricity_75_fixed.csv"
REGION_NAME = "Челябинская область"
SUBJECT_ID = 75

TRAIN_START = "2022-01-01"
TRAIN_END_INITIAL = "2024-12-31"
FORECAST_START = "2025-01-01"
FORECAST_END = "2025-12-31"

HORIZON = 24
N_ESTIMATORS = 200
RANDOM_STATE = 42
TARGET_COL = "actual_consumption"
YEAR = FORECAST_START[:4]

METRICS_FILE          = f"chelyabinsk_metrics_{YEAR}.csv"
METRICS_DAILY_FILE    = f"chelyabinsk_metrics_daily_{YEAR}.csv"
METRICS_MONTHLY_FILE  = f"chelyabinsk_metrics_monthly_{YEAR}.csv"
METRICS_PROGRESS_FILE = f"chelyabinsk_metrics_progress_{YEAR}.csv"
FORECAST_FILE         = f"chelyabinsk_forecast_{YEAR}.csv"

# ============================================================
# ФУНКЦИИ
# ============================================================
def clean_numeric(series):
    return pd.to_numeric(
        series.astype(str)
        .str.replace('МВт*ч', '', regex=False)
        .str.replace('руб./МВт*ч', '', regex=False)
        .str.replace('\u00a0', '', regex=False)
        .str.replace(' ', '', regex=False)
        .str.replace(',', '.', regex=False)
        .replace('nan', None)
        .replace('None', None),
        errors='coerce'
    )

def mape(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    mask = y_true != 0
    return np.nan if mask.sum() == 0 else \
        np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def smape(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    mask = denom != 0
    return np.nan if mask.sum() == 0 else \
        np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100

def wape(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    s = np.sum(np.abs(y_true))
    return np.nan if s == 0 else np.sum(np.abs(y_true - y_pred)) / s * 100

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

def mase(y_true, y_pred, y_naive):
    y_true, y_pred, y_naive = map(np.asarray, (y_true, y_pred, y_naive))
    if len(y_naive) == 0:
        return np.nan
    denom = np.mean(np.abs(y_true - y_naive))
    return np.nan if denom == 0 else np.mean(np.abs(y_true - y_pred)) / denom

def compute_metrics(y_true, y_pred, y_naive=None):
    return {
        "MAE":    mae(y_true, y_pred),
        "RMSE":   rmse(y_true, y_pred),
        "MAPE_%": mape(y_true, y_pred),
        "WAPE_%": wape(y_true, y_pred),
        "SMAPE_%":smape(y_true, y_pred),
        "MASE":   mase(y_true, y_pred, y_naive) if y_naive is not None else np.nan,
        "n":      len(y_true),
    }

def save_metrics_csv(rows, path):
    if rows:
        pd.DataFrame(rows).to_csv(path, index=False, encoding='utf-8-sig')

# ============================================================
# ЗАГРУЗКА
# ============================================================
print(f"=== Загрузка: {CSV_FILE} ===")
try:
    raw = pd.read_csv(CSV_FILE, encoding='utf-8-sig')
except UnicodeDecodeError:
    raw = pd.read_csv(CSV_FILE, encoding='cp1251')

raw.columns = raw.columns.str.strip().str.replace('\ufeff', '', regex=False)

if 'region_name' in raw.columns:
    regions = raw['region_name'].astype(str).unique()
    name = REGION_NAME if REGION_NAME in regions else regions[0]
    raw = raw[raw['region_name'].astype(str) == name].copy()
    print(f"Регион: {name}")

raw['ds'] = pd.to_datetime(
    raw['date'].astype(str) + ' ' + raw['hour'].astype(str).str.zfill(2) + ':00:00',
    errors='coerce'
)
raw['y'] = clean_numeric(raw[TARGET_COL])
raw['unique_id'] = REGION_NAME

df = raw[['ds', 'unique_id', 'y']].dropna(subset=['ds', 'unique_id', 'y'])
df = df.drop_duplicates(subset=['unique_id', 'ds'])
df = df.sort_values(['unique_id', 'ds']).reset_index(drop=True)

df = df[(df['ds'] >= TRAIN_START) &
        (df['ds'] < pd.Timestamp(FORECAST_END) + pd.Timedelta(days=1))].copy()

df = fill_gaps(df, freq='h', start='per_serie', end='per_serie')
df['y'] = df.groupby('unique_id')['y'].transform(lambda s: s.ffill().bfill())
df = df.dropna(subset=['y'])

print(f"Строк: {len(df)}, диапазон: {df['ds'].min()} … {df['ds'].max()}")

# Отдельная таблица для naive (не пихаем в df, чтобы MLForecast не ругался)
naive_df = df[['unique_id', 'ds', 'y']].copy()
naive_df['naive'] = naive_df.groupby('unique_id')['y'].shift(24)
naive_df = naive_df[['unique_id', 'ds', 'naive']]

# ============================================================
# WALK-FORWARD
# ============================================================
forecast_days = pd.date_range(FORECAST_START, FORECAST_END, freq='D')
print(f"\n=== Walk-forward: {len(forecast_days)} дней ===")

all_preds = []
progress_rows = []

for i, day in enumerate(forecast_days, 1):
    train = df[df['ds'] < day].copy()
    test = df[(df['ds'] >= day) & (df['ds'] < day + pd.Timedelta(days=1))].copy()
    if len(train) < 100 or test.empty:
        continue

    fcst = MLForecast(
        models=LGBMRegressor(
            n_estimators=N_ESTIMATORS,
            learning_rate=0.05,
            force_col_wise=True,
            verbose=-1,
            random_state=RANDOM_STATE,
        ),
        freq='h',
        lags=[1, 24, 25, 168, 169],
        date_features=['hour', 'dayofweek', 'month'],
    )
    fcst.fit(train)
    preds = fcst.predict(HORIZON).rename(columns={'LGBMRegressor': 'y_pred'})

    merged = test[['unique_id', 'ds', 'y']].merge(
        preds, on=['unique_id', 'ds'], how='left'
    )
    all_preds.append(merged)

    if i % 10 == 0 or i == 1 or i == len(forecast_days):
        so_far = pd.concat(all_preds, ignore_index=True).dropna(subset=['y_pred'])
        so_far = so_far.merge(naive_df, on=['unique_id', 'ds'], how='left')
        m = compute_metrics(so_far['y'], so_far['y_pred'], so_far['naive'].dropna())
        progress_rows.append({
            "iteration": i,
            "last_day": day.date(),
            "train_start": train['ds'].min().date(),
            "train_end": train['ds'].max().date(),
            "train_rows": len(train),
            "forecast_end_so_far": so_far['ds'].max().date(),
            **m,
        })
        save_metrics_csv(progress_rows, METRICS_PROGRESS_FILE)
        print(f"  [{i}/{len(forecast_days)}] {day.date()} — "
              f"MAPE={m['MAPE_%']:.2f}%, MAE={m['MAE']:.1f}, "
              f"WAPE={m['WAPE_%']:.2f}%", flush=True)

# ============================================================
# ИТОГОВЫЕ МЕТРИКИ
# ============================================================
result = pd.concat(all_preds, ignore_index=True)
result = result.sort_values(['unique_id', 'ds']).reset_index(drop=True)
result = result.dropna(subset=['y_pred'])
result = result.merge(naive_df, on=['unique_id', 'ds'], how='left')

overall = compute_metrics(result['y'], result['y_pred'], result['naive'].dropna())
pd.DataFrame([{
    "region": REGION_NAME, "subject_id": SUBJECT_ID,
    "train_start": TRAIN_START, "train_end_initial": TRAIN_END_INITIAL,
    "forecast_start": FORECAST_START, "forecast_end": FORECAST_END,
    **overall,
}]).to_csv(METRICS_FILE, index=False, encoding='utf-8-sig')

print(f"\n✅ Общие метрики: {METRICS_FILE}")
print(f"   MAE={overall['MAE']:.2f}  RMSE={overall['RMSE']:.2f}  "
      f"MAPE={overall['MAPE_%']:.2f}%  WAPE={overall['WAPE_%']:.2f}%  "
      f"SMAPE={overall['SMAPE_%']:.2f}%  MASE={overall['MASE']:.3f}")

# По дням
daily_rows = []
for d, g in result.groupby(result['ds'].dt.date):
    m = compute_metrics(g['y'], g['y_pred'], g['naive'].dropna())
    daily_rows.append({"date": d, **m})
pd.DataFrame(daily_rows).to_csv(METRICS_DAILY_FILE, index=False, encoding='utf-8-sig')
print(f"✅ По дням: {METRICS_DAILY_FILE}")

# По месяцам
monthly_rows = []
for (y_, m_), g in result.groupby([result['ds'].dt.year, result['ds'].dt.month]):
    mm = compute_metrics(g['y'], g['y_pred'], g['naive'].dropna())
    monthly_rows.append({"year": y_, "month": m_, **mm})
pd.DataFrame(monthly_rows).to_csv(METRICS_MONTHLY_FILE, index=False, encoding='utf-8-sig')
print(f"✅ По месяцам: {METRICS_MONTHLY_FILE}")

# Прогнозы
result[['unique_id', 'ds', 'y', 'y_pred', 'naive']].to_csv(
    FORECAST_FILE, index=False, encoding='utf-8-sig'
)
print(f"✅ Прогнозы: {FORECAST_FILE}")

# ============================================================
# ГРАФИКИ
# ============================================================

# --- 1) Весь 2025 год: факт vs прогноз ---
fig, ax = plt.subplots(figsize=(18, 6))
ax.plot(result['ds'], result['y'], label='Факт', color='steelblue',
        linewidth=1.0, alpha=0.9)
ax.plot(result['ds'], result['y_pred'], label='Прогноз (walk-forward)',
        color='orange', linewidth=1.0, alpha=0.9)

ax.set_title(f"Прогноз потребления за весь {YEAR} год — {REGION_NAME}")
ax.set_xlabel('Дата')
ax.set_ylabel('МВт*ч')
ax.legend(loc='upper right')
ax.grid(alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"chelyabinsk_plot_{YEAR}.png", dpi=120)
plt.close()
print(f"✅ График за весь год: chelyabinsk_plot_{YEAR}.png")

# --- 2) Весь год, но по месяцам (12 подграфиков) ---
months = result['ds'].dt.month.unique()
n_months = len(months)
fig, axes = plt.subplots(n_months, 1, figsize=(16, 3 * n_months), sharex=False)

if n_months == 1:
    axes = [axes]

month_names = ['Янв','Фев','Мар','Апр','Май','Июн',
               'Июл','Авг','Сен','Окт','Ноя','Дек']

for ax, m in zip(axes, months):
    g = result[result['ds'].dt.month == m]
    ax.plot(g['ds'], g['y'], label='Факт', color='steelblue', linewidth=1.0)
    ax.plot(g['ds'], g['y_pred'], label='Прогноз', color='orange',
            linewidth=1.0, alpha=0.9)
    ax.set_title(f"{month_names[m-1]} {YEAR}")
    ax.set_ylabel('МВт*ч')
    ax.grid(alpha=0.3)
    ax.legend(loc='upper right', fontsize=8)

plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"chelyabinsk_plot_months_{YEAR}.png", dpi=120)
plt.close()
print(f"✅ График по месяцам: chelyabinsk_plot_months_{YEAR}.png")

# --- 3) MAPE по месяцам ---
monthly_df = pd.DataFrame(monthly_rows)
fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(monthly_df['month'].astype(str), monthly_df['MAPE_%'], color='coral')
ax.set_title(f"MAPE по месяцам — {REGION_NAME}, {YEAR}")
ax.set_xlabel('Месяц')
ax.set_ylabel('MAPE, %')
ax.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(f"chelyabinsk_mape_monthly_{YEAR}.png", dpi=120)
plt.close()
print(f"✅ MAPE по месяцам: chelyabinsk_mape_monthly_{YEAR}.png")

# --- 4) MAPE по дням года ---
daily_df = pd.DataFrame(daily_rows)
fig, ax = plt.subplots(figsize=(18, 5))
ax.plot(pd.to_datetime(daily_df['date']), daily_df['MAPE_%'],
        color='darkred', linewidth=1.0)
ax.axhline(daily_df['MAPE_%'].mean(), color='black', linestyle='--',
           linewidth=1.0, label=f"Среднее = {daily_df['MAPE_%'].mean():.2f}%")
ax.set_title(f"MAPE по дням — {REGION_NAME}, {YEAR}")
ax.set_xlabel('Дата')
ax.set_ylabel('MAPE, %')
ax.legend()
ax.grid(alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"chelyabinsk_mape_daily_{YEAR}.png", dpi=120)
plt.close()
print(f"✅ MAPE по дням: chelyabinsk_mape_daily_{YEAR}.png")