import numpy as np
import pandas as pd

df = pd.read_csv('forecast_2026_improved_hourly.csv')


valid_days = (
    df.groupby('date')['is_actual']
    .agg(lambda s: bool(s.all()) and len(s) == 24)
)
valid_dates = valid_days[valid_days].index
sub = df[df['date'].isin(valid_dates)].copy()

hits_prob = 0
hits_mean = 0
total_peaks = 0

for _, day_df in sub.groupby('date', sort=False):
    actual_top3 = set(np.argsort(day_df['actual_mw'].values)[-3:])
    pred_top3_prob = set(np.argsort(day_df['prob_top3'].values)[-3:])
    pred_top3_mean = set(np.argsort(day_df['forecast_mw'].values)[-3:])

    hits_prob += len(pred_top3_prob & actual_top3)
    hits_mean += len(pred_top3_mean & actual_top3)
    total_peaks += 3

print(f"���� � ������ ������: {len(valid_dates)} (����� �����: {total_peaks})")
print(f"Hit-Rate �� ����������� �����-����� (prob_top3): {hits_prob / total_peaks * 100:.2f}%")
print(f"Hit-Rate �� �������� �������� (forecast_mw):   {hits_mean / total_peaks * 100:.2f}%")
