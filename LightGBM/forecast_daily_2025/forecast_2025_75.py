import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import holidays
from mlforecast import MLForecast
from lightgbm import LGBMRegressor
from utilsforecast.preprocessing import fill_gaps
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# НАСТРОЙКИ
# ============================================================
CSV_FILE = "electricity_75.csv"
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

METRICS_FILE  = f"chelyabinsk_metrics_{YEAR}.txt"
FORECAST_FILE = f"chelyabinsk_forecast_{YEAR}.csv"

# ============================================================
# ПРАЗДНИКИ РФ
# ============================================================
RU_HOLIDAYS = holidays.Russia(years=[2022, 2023, 2024, 2025])
HOLIDAY_DATES = set(RU_HOLIDAYS.keys())

PRE_HOLIDAY_DATES = {
    pd.Timestamp("2022-02-22"), pd.Timestamp("2022-03-05"),
    pd.Timestamp("2022-04-30"), pd.Timestamp("2022-05-07"),
    pd.Timestamp("2022-06-11"), pd.Timestamp("2022-11-03"),
    pd.Timestamp("2023-02-22"), pd.Timestamp("2023-03-07"),
    pd.Timestamp("2023-04-28"), pd.Timestamp("2023-05-05"),
    pd.Timestamp("2023-06-09"), pd.Timestamp("2023-11-03"),
    pd.Timestamp("2024-02-22"), pd.Timestamp("2024-03-07"),
    pd.Timestamp("2024-04-27"), pd.Timestamp("2024-05-08"),
    pd.Timestamp("2024-06-11"), pd.Timestamp("2024-11-02"),
    pd.Timestamp("2025-01-08"), pd.Timestamp("2025-02-22"),
    pd.Timestamp("2025-03-07"), pd.Timestamp("2025-04-30"),
    pd.Timestamp("2025-05-08"), pd.Timestamp("2025-06-11"),
    pd.Timestamp("2025-11-03"),
}


def _days_to_nearest(dates, holidays_dt):
    res = []
    for d in dates:
        d_norm = pd.Timestamp(d).normalize()
        future = holidays_dt[holidays_dt >= d_norm]
        res.append(365 if len(future) == 0 else (future.min() - d_norm).days)
    return pd.Series(res, index=dates.index)


def _days_from_nearest(dates, holidays_dt):
    res = []
    for d in dates:
        d_norm = pd.Timestamp(d).normalize()
        past = holidays_dt[holidays_dt <= d_norm]
        res.append(365 if len(past) == 0 else (d_norm - past.max()).days)
    return pd.Series(res, index=dates.index)


def add_holiday_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    dates = df['ds'].dt.date

    df['is_weekend'] = df['ds'].dt.dayofweek.isin([5, 6]).astype(int)
    df['is_holiday'] = dates.isin(HOLIDAY_DATES).astype(int)
    df['is_day_off'] = ((df['is_weekend'] == 1) | (df['is_holiday'] == 1)).astype(int)
    df['is_pre_holiday'] = df['ds'].dt.normalize().isin(PRE_HOLIDAY_DATES).astype(int)

    holiday_norm = pd.to_datetime(list(HOLIDAY_DATES))
    post_holiday = pd.to_datetime([d + pd.Timedelta(days=1) for d in HOLIDAY_DATES])
    df['is_post_holiday'] = df['ds'].dt.normalize().isin(post_holiday).astype(int)

    df['days_to_holiday']   = _days_to_nearest(df['ds'], holiday_norm)
    df['days_from_holiday'] = _days_from_nearest(df['ds'], holiday_norm)

    df['is_new_year_period'] = (
        ((df['ds'].dt.month == 1) & (df['ds'].dt.day <= 8)) |
        ((df['ds'].dt.month == 12) & (df['ds'].dt.day == 31))
    ).astype(int)
    df['is_may_holidays'] = (
        (df['ds'].dt.month == 5) & (df['ds'].dt.day <= 10)
    ).astype(int)

    return df


# ============================================================
# МЕТРИКИ: MAE, RMSE, WAPE, MAPE
# ============================================================
def clean_numeric(series):
    return pd.to_numeric(
        series.astype(str)
        .str.replace('МВт*ч', '', regex=False)
        .str.replace('руб./МВт*ч', '', regex=False)
        .str.replace('\u00a0', '', regex=False)
        .str.replace(' ', '', regex=False)
        .str.replace(',', '.', regex=False)
        .replace('nan', None).replace('None', None),
        errors='coerce'
    )

def mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def wape(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    s = np.sum(np.abs(y_true))
    return np.nan if s == 0 else np.sum(np.abs(y_true - y_pred)) / s * 100

def mape(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    mask = y_true != 0
    return np.nan if mask.sum() == 0 else \
        np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def compute_metrics(y_true, y_pred):
    return {
        "MAE":    mae(y_true, y_pred),
        "RMSE":   rmse(y_true, y_pred),
        "WAPE_%": wape(y_true, y_pred),
        "MAPE_%": mape(y_true, y_pred),
        "n":      len(y_true),
    }


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
df = add_holiday_features(df)

EXOG_FEATURES = [
    'is_weekend', 'is_holiday', 'is_day_off',
    'is_pre_holiday', 'is_post_holiday',
    'days_to_holiday', 'days_from_holiday',
    'is_new_year_period', 'is_may_holidays',
]

print(f"Строк: {len(df)}, диапазон: {df['ds'].min()} … {df['ds'].max()}")

# ============================================================
# WALK-FORWARD
# ============================================================
forecast_days = pd.date_range(FORECAST_START, FORECAST_END, freq='D')
print(f"\n=== Walk-forward: {len(forecast_days)} дней ===")

all_preds = []

for i, day in enumerate(forecast_days, 1):
    train = df[df['ds'] < day].copy()
    test  = df[(df['ds'] >= day) & (df['ds'] < day + pd.Timedelta(days=1))].copy()
    if len(train) < 100 or test.empty:
        continue

    train_exog = train[['unique_id', 'ds', 'y'] + EXOG_FEATURES].copy()
    test_exog  = test[['unique_id', 'ds'] + EXOG_FEATURES].copy()

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
    fcst.fit(train_exog, static_features=[])
    preds = fcst.predict(HORIZON, X_df=test_exog).rename(
        columns={'LGBMRegressor': 'y_pred'}
    )

    merged = test[['unique_id', 'ds', 'y']].merge(
        preds, on=['unique_id', 'ds'], how='left'
    )
    all_preds.append(merged)

    if i % 10 == 0 or i == 1 or i == len(forecast_days):
        so_far = pd.concat(all_preds, ignore_index=True).dropna(subset=['y_pred'])
        m = compute_metrics(so_far['y'], so_far['y_pred'])
        print(f"  [{i}/{len(forecast_days)}] {day.date()} — "
              f"MAPE={m['MAPE_%']:.2f}%, MAE={m['MAE']:.1f}, "
              f"WAPE={m['WAPE_%']:.2f}%", flush=True)

# ============================================================
# ИТОГОВЫЕ МЕТРИКИ ЗА ГОД
# ============================================================
result = pd.concat(all_preds, ignore_index=True)
result = result.sort_values(['unique_id', 'ds']).reset_index(drop=True)
result = result.dropna(subset=['y_pred'])

overall = compute_metrics(result['y'], result['y_pred'])

with open(METRICS_FILE, 'w', encoding='utf-8') as f:
    f.write(f"Регион: {REGION_NAME} (subject_id={SUBJECT_ID})\n")
    f.write(f"Период прогноза: {FORECAST_START} — {FORECAST_END}\n")
    f.write(f"Обучающая выборка: {TRAIN_START} — {TRAIN_END_INITIAL}\n")
    f.write("=" * 45 + "\n")
    f.write("МЕТРИКИ ЗА ВЕСЬ ГОД\n")
    f.write("=" * 45 + "\n")
    f.write(f"MAE:   {overall['MAE']:.2f} МВт*ч\n")
    f.write(f"RMSE:  {overall['RMSE']:.2f} МВт*ч\n")
    f.write(f"WAPE:  {overall['WAPE_%']:.2f} %\n")
    f.write(f"MAPE:  {overall['MAPE_%']:.2f} %\n")
    f.write(f"n:     {overall['n']}\n")

print(f"\n✅ Метрики: {METRICS_FILE}")
print(f"   MAE={overall['MAE']:.2f}  RMSE={overall['RMSE']:.2f}  "
      f"WAPE={overall['WAPE_%']:.2f}%  MAPE={overall['MAPE_%']:.2f}%")

result[['unique_id', 'ds', 'y', 'y_pred']].to_csv(
    FORECAST_FILE, index=False, encoding='utf-8-sig'
)
print(f"✅ Прогнозы: {FORECAST_FILE}")

# ============================================================
# ГРАФИК ПРОГНОЗА
# ============================================================
fig, ax = plt.subplots(figsize=(18, 6))
ax.plot(result['ds'], result['y'], label='Факт', color='steelblue',
        linewidth=1.0, alpha=0.9)
ax.plot(result['ds'], result['y_pred'], label='Прогноз',
        color='orange', linewidth=1.0, alpha=0.9)

holiday_mask = result['ds'].dt.date.isin(HOLIDAY_DATES)
if holiday_mask.any():
    for hd in result[holiday_mask]['ds'].dt.normalize().unique():
        ax.axvline(pd.Timestamp(hd), color='red', alpha=0.15, linewidth=0.8)

ax.set_title(f"Прогноз потребления за {YEAR} год — {REGION_NAME}")
ax.set_xlabel('Дата')
ax.set_ylabel('МВт*ч')
ax.legend(loc='upper right')
ax.grid(alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"chelyabinsk_plot_{YEAR}.png", dpi=120)
plt.close()
print(f"✅ График прогноза: chelyabinsk_plot_{YEAR}.png")

# ============================================================
# ГРАФИКИ ПО КАЖДОЙ МЕТРИКЕ (по дням)
# ============================================================
daily_rows = []
for d, g in result.groupby(result['ds'].dt.date):
    m = compute_metrics(g['y'], g['y_pred'])
    daily_rows.append({"date": pd.Timestamp(d), **m})
daily_df = pd.DataFrame(daily_rows)

METRICS_TO_PLOT = {
    'MAE':    ('MAE, МВт*ч',  'chelyabinsk_MAE'),
    'RMSE':   ('RMSE, МВт*ч', 'chelyabinsk_RMSE'),
    'WAPE_%': ('WAPE, %',     'chelyabinsk_WAPE'),
    'MAPE_%': ('MAPE, %',     'chelyabinsk_MAPE'),
}

for col, (ylabel, prefix) in METRICS_TO_PLOT.items():
    fig, ax = plt.subplots(figsize=(16, 4))
    ax.plot(daily_df['date'], daily_df[col], color='darkred', linewidth=1.0)
    ax.axhline(daily_df[col].mean(), color='black', linestyle='--',
               linewidth=1.0,
               label=f"Среднее = {daily_df[col].mean():.2f}")
    ax.set_title(f"{ylabel} по дням — {REGION_NAME}, {YEAR}")
    ax.set_xlabel('Дата')
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    fname = f"{prefix}_{YEAR}.png"
    plt.savefig(fname, dpi=120)
    plt.close()
    print(f"✅ График {col}: {fname}")

# ============================================================
# АНАЛИЗ ОШИБОК ПО ТИПАМ ДНЕЙ
# ============================================================
result['is_holiday_day'] = result['ds'].dt.date.isin(HOLIDAY_DATES).astype(int)
result['is_weekend_day'] = result['ds'].dt.dayofweek.isin([5, 6]).astype(int)

print("\n" + "=" * 55)
print("ОШИБКИ: ПРАЗДНИКИ vs ОБЫЧНЫЕ ДНИ")
print("=" * 55)

for label, mask in [
    ("Праздничные дни", result['is_holiday_day'] == 1),
    ("Выходные (сб/вс)", (result['is_weekend_day'] == 1) & (result['is_holiday_day'] == 0)),
    ("Рабочие дни", (result['is_weekend_day'] == 0) & (result['is_holiday_day'] == 0)),
]:
    g = result[mask]
    if len(g) > 0:
        m = compute_metrics(g['y'], g['y_pred'])
        print(f"  {label:20s}: MAPE={m['MAPE_%']:.2f}%  "
              f"WAPE={m['WAPE_%']:.2f}%  MAE={m['MAE']:.1f}  n={m['n']}")