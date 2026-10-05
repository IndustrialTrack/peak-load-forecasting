import pandas as pd
import glob
import matplotlib.pyplot as plt
from mlforecast import MLForecast
from lightgbm import LGBMRegressor
from utilsforecast.preprocessing import fill_gaps

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


HORIZON = 168

files = [f for f in glob.glob('electricity_*.csv') if '_forecast' not in f]

for f in files:
    print(f"\n=== Обработка файла: {f} ===")

    try:
        df = pd.read_csv(f, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(f, encoding='cp1251')

    df.columns = df.columns.str.strip().str.replace('\ufeff', '', regex=False)

    df['ds'] = pd.to_datetime(
        df['date'].astype(str) + ' ' + df['hour'].astype(str) + ':00:00',
        errors='coerce'
    )

    df['y'] = clean_numeric(df['actual_consumption'])
    df['unique_id'] = df['region_name'].astype(str)

    df = df[['ds', 'unique_id', 'y']]
    df = df.dropna(subset=['ds', 'unique_id', 'y'])
    df = df.drop_duplicates(subset=['unique_id', 'ds'])
    df = df.sort_values(['unique_id', 'ds']).reset_index(drop=True)

    print(f"Регион: {df['unique_id'].iloc[0]}, строк: {len(df)}")

    df = fill_gaps(df, freq='h', start='per_serie', end='per_serie')
    df['y'] = df.groupby('unique_id')['y'].transform(lambda s: s.ffill().bfill())
    df = df.dropna(subset=['y'])

    print(f"Строк после fill_gaps: {len(df)}")

    test = df.groupby('unique_id').tail(HORIZON).copy()
    train = df.drop(test.index).copy()

    fcst_eval = MLForecast(
        models=LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            force_col_wise=True,
            verbose=-1
        ),
        freq='h',
        lags=[1, 24, 25, 168, 169],
        date_features=['hour', 'dayofweek', 'month']
    )
    fcst_eval.fit(train)
    preds_eval = fcst_eval.predict(HORIZON)

    merged = preds_eval.merge(test, on=['unique_id', 'ds'], suffixes=('_pred', '_fact'))
    mae = (merged['LGBMRegressor'] - merged['y']).abs().mean()
    mape = ((merged['LGBMRegressor'] - merged['y']).abs() / merged['y']).mean() * 100

    merged['day_ahead'] = merged.groupby('unique_id').cumcount() // 24 + 1
    daily_mape = (
        merged.groupby('day_ahead')
        .apply(lambda g: ((g['LGBMRegressor'] - g['y']).abs() / g['y']).mean() * 100,
               include_groups=False)
    )

    print(f"MAE за неделю: {mae:.2f} МВт*ч")
    print(f"MAPE за неделю: {mape:.2f}%")
    print("MAPE по дням вперёд:")
    for d, m in daily_mape.items():
        print(f"  День {d}: {m:.2f}%")

    fcst = MLForecast(
        models=LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            force_col_wise=True,
            verbose=-1
        ),
        freq='h',
        lags=[1, 24, 25, 168, 169],
        date_features=['hour', 'dayofweek', 'month']
    )
    fcst.fit(df)
    preds = fcst.predict(HORIZON)

    out_name = f.replace('.csv', '_forecast.csv')
    preds.to_csv(out_name, index=False, encoding='utf-8-sig')
    print(f"Прогноз на неделю сохранён в {out_name}")

    region = df['unique_id'].iloc[0]
    hist = df.tail(336)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(hist['ds'], hist['y'], label='Факт (последняя неделя)', color='steelblue')
    ax.plot(preds['ds'], preds['LGBMRegressor'],
            label='Прогноз на неделю', color='orange', linewidth=2)

    ax.set_title(f"Прогноз потребления на неделю: {region}")
    ax.set_xlabel('Дата')
    ax.set_ylabel('МВт*ч')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()

    plot_name = f.replace('.csv', '_plot.png')
    plt.savefig(plot_name, dpi=120)
    plt.close()
    print(f"График сохранён в {plot_name}")

print("\nВсе файлы обработаны")