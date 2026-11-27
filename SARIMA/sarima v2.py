import os
import gc
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pmdarima as pm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

warnings.filterwarnings("ignore")

# ==============================================================================
# НАСТРОЙКИ ПАЙПЛАЙНА (УЛУЧШЕННЫЙ SARIMAX + КАЛЕНДАРЬ + МОНТЕ-КАРЛО)
# ==============================================================================
FILES_TO_PROCESS = [
    'electricity_75.csv',
]

TRAIN_START_DATE = '2022-01-01 00:00:00'
TRAIN_END_DATE = '2025-12-31 23:00:00'
FORECAST_2026_START = '2026-01-01 00:00:00'
FORECAST_2026_END = '2026-12-31 23:00:00'

SEASONALITY_M = 24
FORECAST_STEP = 24
N_SIMULATIONS = 10000
PARAMS_LOOKBACK_DAYS = 28
HISTORY_WINDOW_HOURS = 24 * 60
CACHE_FILE = 'sarimax_calendar_params_cache.json'


# ==============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ==============================================================================
def mean_absolute_scaled_error(y_true, y_pred, y_train, m=24):
    naive_errors = np.abs(y_train.iloc[m:].values - y_train.iloc[:-m].values)
    mae_naive = np.mean(naive_errors)
    if mae_naive == 0:
        return np.nan
    return mean_absolute_error(y_true, y_pred) / mae_naive


def build_calendar_exog(index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Формирует календарные признаки для учета различий между буднями и выходными.
    Используем one-hot для дня недели (кроме базового понедельника) и флаг выходного.
    """
    dow = index.dayofweek
    exog = pd.DataFrame(index=index)
    exog['is_weekend'] = (dow >= 5).astype(float)
    for d in range(1, 7):
        exog[f'dow_{d}'] = (dow == d).astype(float)
    return exog


def preprocess_data(filepath):
    try:
        df = pd.read_csv(filepath, encoding='utf-8-sig')
    except UnicodeDecodeError:
        df = pd.read_csv(filepath, encoding='windows-1251')

    df['datetime'] = pd.to_datetime(df['date']) + pd.to_timedelta(df['hour'], unit='h')
    df.set_index('datetime', inplace=True)
    df.sort_index(inplace=True)
    df = df[~df.index.duplicated(keep='first')]

    region_name = df['region_name'].iloc[0] if 'region_name' in df.columns else os.path.basename(filepath)
    df['actual_consumption'] = (
        df['actual_consumption']
        .astype(str)
        .str.replace(r' МВт\*ч', '', regex=True)
        .str.replace(' ', '', regex=True)
        .astype(float)
    )

    ts = df['actual_consumption'].asfreq('h')
    if ts.isnull().sum() > 0:
        ts = ts.interpolate(method='time')

    return ts, region_name


def get_sarimax_order(train_series, train_exog, cache_key):
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r', encoding='utf-8') as f:
            cache = json.load(f)
    else:
        cache = {}

    if cache_key in cache:
        order = tuple(cache[cache_key]['order'])
        seasonal_order = tuple(cache[cache_key]['seasonal_order'])
        print(f"[{cache_key}] Параметры из кеша: {order} x {seasonal_order}")
        return order, seasonal_order

    y_sub = train_series.iloc[-(24 * PARAMS_LOOKBACK_DAYS):]
    x_sub = train_exog.iloc[-(24 * PARAMS_LOOKBACK_DAYS):]

    auto_model = pm.auto_arima(
        y_sub,
        exogenous=x_sub,
        m=SEASONALITY_M,
        seasonal=True,
        d=None,
        D=1,
        max_d=1,
        max_p=5,
        max_q=3,
        max_P=3,
        start_Q=0,
        max_Q=0,
        stepwise=True,
        trace=True,
        error_action='ignore',
        suppress_warnings=True,
    )

    order = auto_model.order
    seasonal_order = auto_model.seasonal_order
    cache[cache_key] = {
        'order': list(order),
        'seasonal_order': list(seasonal_order),
    }
    with open(CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(cache, f, ensure_ascii=False, indent=4)

    print(f"[{cache_key}] Подобраны параметры: {order} x {seasonal_order}")
    return order, seasonal_order


def simulate_top3_probabilities(forecast_obj, pred_mean_values, n_steps, n_sims=10000):
    """
    Генерирует сценарии с учетом межчасовой ковариации прогноза (если доступна),
    иначе использует стандартную ошибку по каждому часу.
    """
    sims = None
    try:
        cov = forecast_obj.cov_pred_mean
        if cov is not None and cov.shape == (n_steps, n_steps):
            cov_mat = np.asarray(cov, dtype=float)
            cov_mat = 0.5 * (cov_mat + cov_mat.T) + np.eye(n_steps) * 1e-6
            sims = np.random.multivariate_normal(
                mean=pred_mean_values,
                cov=cov_mat,
                size=n_sims,
            )
    except Exception:
        sims = None

    if sims is None:
        pred_se = np.clip(np.asarray(forecast_obj.se_mean.values, dtype=float), 1e-6, None)
        sims = np.random.normal(
            loc=pred_mean_values,
            scale=pred_se,
            size=(n_sims, n_steps),
        )

    top3_sim_idx = np.argsort(sims, axis=1)[:, -3:]
    prob_scores = np.bincount(top3_sim_idx.ravel(), minlength=n_steps) / n_sims
    return prob_scores


# ==============================================================================
# ОСНОВНОЙ ЦИКЛ ПОСУТОЧНОГО ПРОГНОЗА НА 2026 ГОД
# ==============================================================================
def main():
    summary_rows = []
    daily_forecast_records = []

    for file_path in FILES_TO_PROCESS:
        if not os.path.exists(file_path):
            print(f"ОШИБКА: Файл {file_path} не найден.")
            continue

        ts, region_name = preprocess_data(file_path)
        train = ts.loc[TRAIN_START_DATE:TRAIN_END_DATE]
        exog_train = build_calendar_exog(train.index)

        available_2026 = ts.loc[FORECAST_2026_START:FORECAST_2026_END]
        full_2026_index = pd.date_range(
            start=FORECAST_2026_START,
            end=FORECAST_2026_END,
            freq='h'
        )
        actual_2026_aligned = available_2026.reindex(full_2026_index)
        exog_2026_full = build_calendar_exog(full_2026_index)

        cache_key = f"{region_name}_cal"
        order, seasonal_order = get_sarimax_order(train, exog_train, cache_key)

        base_model = SARIMAX(
            train,
            exog=exog_train,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        base_res = base_model.fit(disp=False)
        fitted_params = base_res.params
        del base_model, base_res
        gc.collect()

        history_series = train.iloc[-HISTORY_WINDOW_HOURS:].copy()
        history_exog = exog_train.iloc[-HISTORY_WINDOW_HOURS:].copy()

        preds_list = []
        peak_pred_x, peak_pred_y = [], []
        peak_act_x, peak_act_y = [], []

        total_hits = 0
        total_peaks = 0

        for start_idx in range(0, len(full_2026_index), FORECAST_STEP):
            if start_idx % (24 * 30) == 0:
                print(f"[{region_name}] Обработано {start_idx // 24} / 365 дней...")
            step_index = full_2026_index[start_idx : start_idx + FORECAST_STEP]
            n_steps = len(step_index)
            if n_steps == 0:
                break

            step_exog = exog_2026_full.loc[step_index]

            step_model = SARIMAX(
                history_series,
                exog=history_exog,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            step_res = step_model.filter(fitted_params)

            forecast_obj = step_res.get_forecast(steps=n_steps, exog=step_exog)
            pred_mean = pd.Series(forecast_obj.predicted_mean.values, index=step_index)

            prob_scores = simulate_top3_probabilities(
                forecast_obj=forecast_obj,
                pred_mean_values=pred_mean.values,
                n_steps=n_steps,
                n_sims=N_SIMULATIONS,
            )

            top3_pred_idx = np.argsort(prob_scores)[-3:][::-1]
            for idx in top3_pred_idx:
                peak_pred_x.append(step_index[idx])
                peak_pred_y.append(pred_mean.values[idx])

            actual_chunk = actual_2026_aligned.loc[step_index]
            has_real_fact = not actual_chunk.isnull().any()

            if has_real_fact:
                top3_actual_idx = np.argsort(actual_chunk.values)[-3:][::-1]
                for idx in top3_actual_idx:
                    peak_act_x.append(step_index[idx])
                    peak_act_y.append(actual_chunk.values[idx])

                total_hits += len(set(top3_pred_idx).intersection(set(top3_actual_idx)))
                total_peaks += len(top3_actual_idx)
                assimilate_chunk = actual_chunk
            else:
                assimilate_chunk = pred_mean

            preds_list.extend(pred_mean.values.tolist())

            for h_idx, dt in enumerate(step_index):
                daily_forecast_records.append({
                    'datetime': dt,
                    'date': dt.strftime('%Y-%m-%d'),
                    'hour': dt.hour,
                    'region': region_name,
                    'forecast_mw': pred_mean.values[h_idx],
                    'actual_mw': actual_chunk.values[h_idx] if has_real_fact else np.nan,
                    'prob_top3': prob_scores[h_idx],
                    'is_actual': has_real_fact,
                })

            history_series = pd.concat([history_series, assimilate_chunk]).iloc[-HISTORY_WINDOW_HOURS:]
            history_exog = pd.concat([history_exog, step_exog]).iloc[-HISTORY_WINDOW_HOURS:]

        gc.collect()
        y_pred_all = pd.Series(preds_list, index=full_2026_index)

        valid_mask = ~actual_2026_aligned.isnull()
        if valid_mask.sum() > 0:
            y_true_eval = actual_2026_aligned[valid_mask]
            y_pred_eval = y_pred_all[valid_mask]

            mae = mean_absolute_error(y_true_eval, y_pred_eval)
            mse = mean_squared_error(y_true_eval, y_pred_eval)
            rmse = np.sqrt(mse)
            mase = mean_absolute_scaled_error(y_true_eval, y_pred_eval, train, m=SEASONALITY_M)
            mape = mean_absolute_percentage_error(y_true_eval, y_pred_eval) * 100
            hit_rate = (total_hits / total_peaks) * 100 if total_peaks > 0 else np.nan
        else:
            mae = mse = rmse = mase = mape = hit_rate = np.nan

        summary_rows.append({
            'Регион': region_name,
            'Оценено часов (факт)': int(valid_mask.sum()),
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'MASE': mase,
            'MAPE (%)': mape,
            'Hit-Rate (%)': hit_rate,
        })

        sns.set_theme(style="whitegrid")
        fig, ax = plt.subplots(figsize=(15, 6))
        ax.plot(y_pred_all.index, y_pred_all.values, label='Прогноз SARIMAX + календарь (2026, D+1)', color='#d62728', linewidth=1.4)
        if valid_mask.sum() > 0:
            ax.plot(y_true_eval.index, y_true_eval.values, label='Факт (доступные записи 2026)', color='#2ca02c', linewidth=1.2, alpha=0.85)
            ax.scatter(peak_act_x, peak_act_y, color='blue', edgecolor='white', s=60, marker='o', zorder=4, label='Фактические ТОП-3 пика')
        ax.scatter(peak_pred_x, peak_pred_y, color='gold', edgecolor='black', s=80, marker='*', zorder=5, label='Прогнозируемые ТОП-3 пика')

        title_metrics = f"MAPE: {mape:.2f}% | Hit-Rate: {hit_rate:.1f}%" if not np.isnan(mape) else "Автономный прогноз (факт отсутствует)"
        ax.set_title(f'Посуточный прогноз SARIMAX на 2026 год: {region_name}\n{title_metrics}', fontsize=13, fontweight='bold')
        ax.set_xlabel('Дата', fontsize=11)
        ax.set_ylabel('Потребление (МВт*ч)', fontsize=11)
        ax.legend(loc='upper left', frameon=True)
        plt.tight_layout()

        safe_name = region_name.replace(' ', '_').replace('/', '_')
        plt.savefig(f'forecast_2026_improved_{safe_name}.png', dpi=300)
        plt.close(fig)

    if daily_forecast_records:
        pd.DataFrame(daily_forecast_records).to_csv('forecast_2026_improved_hourly.csv', index=False, encoding='utf-8-sig')

    if summary_rows:
        df_summary = pd.DataFrame(summary_rows)
        print("\n" + "=" * 60)
        print("СВОДКА ПО ПРОГНОЗУ НА 2026 ГОД (УЛУЧШЕННЫЙ ПАЙПЛАЙН)")
        print("=" * 60)
        print(df_summary.to_string(index=False))
        df_summary.to_csv('metrics_2026_improved_summary.csv', index=False, encoding='utf-8-sig')


if __name__ == "__main__":
    main()
