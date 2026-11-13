import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pmdarima as pm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import warnings

# Отключаем предупреждения для чистоты вывода
warnings.filterwarnings("ignore")

# ==============================================================================
# НАСТРОЙКИ
# ==============================================================================
CSV_FILE_PATH = 'electricity_75.csv'
TARGET_COL = 'actual_consumption'

SEASONALITY_M = 24               # Часовая сезонность
TEST_SIZE_HOURS = 24 * 7         # Тестовая выборка (1 неделя)
HISTORY_WEEKS = 12                # Ограничение глубины истории для быстрого подбора параметров

def main():
    # --------------------------------------------------------------------------
    # 1. ЗАГРУЗКА И СПЕЦИФИЧНАЯ ПРЕДОБРАБОТКА ДАННЫХ
    # --------------------------------------------------------------------------
    print("1. Загрузка и предобработка данных...")
    df = pd.read_csv(CSV_FILE_PATH)
    
    # Сборка единого DatetimeIndex из столбцов 'date' и 'hour'
    df['datetime'] = pd.to_datetime(df['date']) + pd.to_timedelta(df['hour'], unit='h')
    df.set_index('datetime', inplace=True)
    df.sort_index(inplace=True)
    
    # Очистка целевой переменной от текста (' МВт*ч') и пробелов (тысячных разделителей)
    df[TARGET_COL] = df[TARGET_COL].astype(str).str.replace(r' МВт\*ч', '', regex=True)
    df[TARGET_COL] = df[TARGET_COL].str.replace(' ', '', regex=True).astype(float)
    
    # Выделяем временной ряд, задаем часовую частоту и интерполируем пропуски
    ts = df[TARGET_COL].asfreq('h')
    if ts.isnull().sum() > 0:
        print(f"Обнаружено {ts.isnull().sum()} пропусков. Выполняется интерполяция.")
        ts = ts.interpolate(method='time')

    # Ограничиваем историю для ускорения работы auto_arima (берем последние HISTORY_WEEKS)
    total_hours_needed = (HISTORY_WEEKS * 7 * 24)
    if len(ts) > total_hours_needed:
        ts = ts.iloc[-total_hours_needed:]

    # --------------------------------------------------------------------------
    # 2. РАЗДЕЛЕНИЕ НА TRAIN И TEST
    # --------------------------------------------------------------------------
    print("\n2. Разделение на обучающую и тестовую выборки...")
    train = ts.iloc[:-TEST_SIZE_HOURS]
    test = ts.iloc[-TEST_SIZE_HOURS:]
    
    print(f"Размер обучающей выборки (train): {len(train)} наблюдений")
    print(f"Размер тестовой выборки (test): {len(test)} наблюдений")

    # --------------------------------------------------------------------------
    # 3. АВТОМАТИЧЕСКИЙ ПОДБОР ПАРАМЕТРОВ ЧЕРЕЗ PMDARIMA
    # --------------------------------------------------------------------------
    print("\n3. Запуск auto_arima для поиска оптимальных параметров (может занять время)...")
    auto_model = pm.auto_arima(
        train,
        m=SEASONALITY_M,
        seasonal=True,
        stepwise=True,
        trace=True,
        error_action='ignore',
        suppress_warnings=True
    )
    
    best_order = auto_model.order
    best_seasonal_order = auto_model.seasonal_order
    print(f"\nОптимальные параметры найдены!")
    print(f"Order: {best_order} | Seasonal Order: {best_seasonal_order}")

    # --------------------------------------------------------------------------
    # 4. ОБУЧЕНИЕ ИТОГОВОЙ МОДЕЛИ SARIMAX
    # --------------------------------------------------------------------------
    print("\n4. Обучение финальной модели statsmodels SARIMAX...")
    sarima_model = SARIMAX(
        train,
        order=best_order,
        seasonal_order=best_seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    sarima_fit = sarima_model.fit(disp=False)
    print(sarima_fit.summary().tables[1])

    # --------------------------------------------------------------------------
    # 5. РАСЧЕТ ПРОГНОЗА И ВЫЧИСЛЕНИЕ МЕТРИК
    # --------------------------------------------------------------------------
    print("\n5. Расчет прогноза на тестовый период...")
    forecast_obj = sarima_fit.get_forecast(steps=len(test))
    y_pred = forecast_obj.predicted_mean
    conf_int = forecast_obj.conf_int(alpha=0.05)
    
    mae = mean_absolute_error(test, y_pred)
    rmse = np.sqrt(mean_squared_error(test, y_pred))
    mape = mean_absolute_percentage_error(test, y_pred)
    
    print("-" * 30)
    print("БИЗНЕС-МЕТРИКИ КАЧЕСТВА ПРОГНОЗА:")
    print(f"MAE  (Mean Absolute Error):      {mae:.2f}")
    print(f"RMSE (Root Mean Squared Error):  {rmse:.2f}")
    print(f"MAPE (Mean Abs Percentage Err):  {mape:.2%}")
    print("-" * 30)

    # --------------------------------------------------------------------------
    # 6. ВИЗУАЛИЗАЦИЯ
    # --------------------------------------------------------------------------
    print("\n6. Построение графика...")
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(14, 7))
    
    # Отрисовываем для наглядности только последнюю неделю из обучающей выборки
    plot_train_start = train.index[-168]
    
    plt.plot(train.loc[plot_train_start:], label='Train (Последние 7 дней истории)', color='#1f77b4', linewidth=1.5)
    plt.plot(test.index, test, label='Test (Фактическое потребление)', color='#2ca02c', linewidth=2)
    plt.plot(y_pred.index, y_pred, label='Forecast (Прогноз SARIMA)', color='#d62728', linestyle='--', linewidth=2)
    
    plt.fill_between(
        conf_int.index,
        conf_int.iloc[:, 0],
        conf_int.iloc[:, 1],
        color='#d62728',
        alpha=0.15,
        label='95% Доверительный интервал'
    )
    
    plt.title(f'Прогноз потребления электроэнергии (MAPE: {mape:.2%} | MAE: {mae:.2f})', fontsize=15, fontweight='bold')
    plt.xlabel('Дата и время', fontsize=12)
    plt.ylabel('Потребление (МВт*ч)', fontsize=12)
    plt.legend(loc='upper left', frameon=True, shadow=True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
