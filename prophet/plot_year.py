"""
Графики «прогноз vs факт» за весь период (например, 2026 год) по УЖЕ СОХРАНЁННЫМ прогнозам — ничего не пересчитывается.
Отдельная программа: нужны только pandas и matplotlib (Prophet не нужен).

Читает output/bench/forecasts/<метод>.csv (их создаёт main.py) и строит такой же график, как для 30 дней:
прогноз, 80% интервал, факт, предсказанный и фактический топ-3 часа каждых суток, остатки внизу.

    python plot_year.py                                   # регион 75, метод prophet, 2026 год
    python plot_year.py --methods prophet,prophet_nbr     # несколько методов (каждому свои файлы)
    python plot_year.py --region 71 --start 2026-03-01 --end 2026-05-31
    python plot_year.py --rank yhat_upper                 # предсказанный топ-3 по верхней границе интервала

Результат в output/bench/:
    forecast_year_<регион>_<метод>_<год>.png    — один график на весь период
    forecast_months_<регион>_<метод>_<год>.pdf  — по странице на месяц (крупно, как график на 30 дней)
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

TOP_K = 3


def load(path: str, region: str, start: str, end: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["ds"], dtype={"region": str})
    df = df[(df["region"] == region) & (df["ds"] >= pd.Timestamp(start)) &
            (df["ds"] < pd.Timestamp(end) + pd.Timedelta(days=1))]
    df = df.drop_duplicates("ds").sort_values("ds").dropna(subset=["y", "yhat"])
    full = df.groupby(df["ds"].dt.normalize())["ds"].transform("size") == 24      # только полные сутки
    df = df[full].copy()
    for c in ("yhat_lower", "yhat_upper"):                                          # если интервала нет — рисуем линию
        df[c] = df[c].fillna(df["yhat"])
    return df.set_index("ds")


def top_points(df: pd.DataFrame, rank_col: str, k: int = TOP_K):
    """Фактические и предсказанные топ-k часов каждых суток; а также hit rate по периоду."""
    act, pred, hits, days = [], [], 0, 0
    for _, d in df.groupby(df.index.normalize()):
        a, p = d.nlargest(k, "y"), d.nlargest(k, rank_col)
        act.append(a)
        pred.append(p)
        hits += len(set(a.index) & set(p.index))
        days += 1
    if not days:
        return None, None, np.nan, 0
    return pd.concat(act), pd.concat(pred), hits / (k * days) * 100, days


def draw(df: pd.DataFrame, title: str, rank_col: str, big: bool):
    act, pred, hit, days = top_points(df, rank_col)
    s, lw = (14, 0.7) if big else (60, 1.2)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(34 if big else 14, 9 if big else 7), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})
    ax1.fill_between(df.index, df["yhat_lower"], df["yhat_upper"], color="tab:blue", alpha=0.2,
                     label="80% доверительный интервал")
    ax1.plot(df.index, df["yhat"], color="tab:blue", lw=lw, label="Прогноз (yhat)")
    ax1.plot(df.index, df["y"], color="tab:red", lw=lw, label="Факт")
    if act is not None:
        ax1.scatter(pred.index, pred["yhat"], marker="v", color="tab:blue", s=s, zorder=5,
                    label=f"Предсказанный топ-3 (по {rank_col})")
        ax1.scatter(act.index, act["y"], marker="^", color="tab:red", s=s, zorder=5, label="Фактический топ-3")
    ax1.set_title(f"{title}\nhit rate топ-3: {hit:.1f}% ({days} дн.)" if days else title)
    ax1.set_ylabel("МВт*ч")
    ax1.grid(alpha=0.25)
    ax1.legend(loc="upper left", fontsize=9)

    res = df["y"] - df["yhat"]
    ax2.vlines(df.index, 0, res, color="gray", lw=0.6 if big else 1.0)
    ax2.axhline(0, color="black", lw=0.8)
    ax2.set_ylabel("Остаток\n(факт-прогноз)")
    ax2.set_xlabel("Дата")
    ax2.grid(alpha=0.25)
    loc = mdates.AutoDateLocator()
    ax2.xaxis.set_major_locator(loc)
    ax2.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
    fig.tight_layout()
    return fig


def make(forecasts_dir: str, out_dir: str, method: str, region: str, start: str, end: str, rank_col: str):
    path = os.path.join(forecasts_dir, f"{method}.csv")
    if not os.path.exists(path):
        print(f"[!] нет файла {path} — метод {method} пропущен")
        return
    df = load(path, region, start, end)
    if df.empty:
        print(f"[!] {method}: нет полных суток региона {region} за {start}..{end}")
        return
    tag = f"{region}_{method}_{pd.Timestamp(start).year}"
    name = f"регион {region}, {method}"
    fig = draw(df, f"Прогноз vs факт — {name}, {df.index.min():%d.%m.%Y} - {df.index.max():%d.%m.%Y}", rank_col, big=True)
    png = os.path.join(out_dir, f"forecast_year_{tag}.png")
    fig.savefig(png, dpi=120)
    plt.close(fig)
    pdf_path = os.path.join(out_dir, f"forecast_months_{tag}.pdf")
    with PdfPages(pdf_path) as pdf:
        for (yr, mo), g in df.groupby([df.index.year, df.index.month]):
            fig = draw(g, f"Прогноз vs факт — {name}, {yr}-{mo:02d}", rank_col, big=False)
            pdf.savefig(fig, dpi=150)
            plt.close(fig)
    print(f"[OK] {method}: {df.index.normalize().nunique()} дн. -> {png}\n                {pdf_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--forecasts", default=os.path.join("output", "bench", "forecasts"))
    ap.add_argument("--out", default=os.path.join("output", "bench"))
    ap.add_argument("--methods", default="prophet")
    ap.add_argument("--region", default="75")
    ap.add_argument("--start", default="2026-01-01")
    ap.add_argument("--end", default="2026-12-31")
    ap.add_argument("--rank", default="yhat", choices=["yhat", "yhat_upper"],
                    help="по чему выбирать предсказанные топ-3 часа (как в results.txt — yhat)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for m in a.methods.split(","):
        make(a.forecasts, a.out, m.strip(), a.region, a.start, a.end, a.rank)


if __name__ == "__main__":
    main()
