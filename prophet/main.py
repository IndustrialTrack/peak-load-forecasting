"""
Прогноз топ-3 пиковых часов на сутки вперёд — Prophet, единый протокол оценки (задачи 1-2).
Вся логика (признаки, Prophet-варианты, метрики) — в prophet_pipeline.py; здесь только настройки и запуск.

    python main.py --task 1          # один регион (75): Prophet «текущий» (цена того же часа, утечка) vs «честный»
    python main.py --task 2          # несколько регионов: локальный Prophet vs Prophet + регрессор соседей
    python main.py --task all        # обе задачи; по умолчанию на всех ядрах (-j N — число процессов)

Протокол: прогноз на каждый день 2026 года, обучение на ВСЕЙ истории до 00:00 этого дня (2021-2025 + уже прошедшие
дни 2026). Гиперпараметры подбираются на 2025 году (строго до теста).

Результаты в output/bench/:
    results.txt                      — ВСЕ метрики (hit rate и топ-3 часа первыми), сравнение методов, пространственный эффект
    forecast_by_day_<регион>.pdf     — график прогноза против факта на КАЖДЫЙ день 2026 года (по странице на месяц)
    *.csv                            — те же таблицы в csv; forecasts/ — почасовые прогнозы (расчёт возобновляемый)
"""
import argparse
import json
import os

import pandas as pd

import prophet_pipeline as B
from prophet_pipeline import _HAS_REAL_PROPHET

# =============================================================================
# НАСТРОЙКИ ПРОТОКОЛА (всё, что влияет на результат, — здесь)
# =============================================================================
DATA_DIR = "."                       # где лежат electricity_<код>.csv
PRIMARY = "75"                       # регион задачи 1 (Челябинская область)
REGION_NAMES = {"75": "Челябинская обл.", "71": "Тюменская обл."}   # только для подписей; остальные — по коду
TARGET_COL = "actual_consumption"
ENCODING = None

# Тест: каждый день 2026 года — свой прогноз; обучение — вся история до 00:00 этого дня (расширяющееся окно).
# Если в данных нет всего года, тест автоматически обрезается до последних полных суток.
TEST_START, TEST_END = "2026-01-01", "2026-12-31"
# Валидация (подбор гиперпараметров): весь 2025 год с шагом 10 дней — охватывает все сезоны, строго до теста.
VAL_START, VAL_END, VAL_STEP_DAYS = "2025-01-01", "2025-12-31", 10
MIN_TRAIN_DAYS = 180

MAX_TRAIN_DAYS = None                # None = вся история; число = только последние N дней (дешевле по времени)
N_NEIGHBORS = 3                      # сколько самых связанных регионов брать как «соседей»
TOP_K = 3
SEED = 42

TASK_METHODS = {
    "1": ["prophet_current", "prophet"],
    "2": ["prophet", "prophet_nbr"],
}
SPATIAL_PAIRS = [("prophet", "prophet_nbr")]     # (база, новый): локальный Prophet -> Prophet + соседи


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="all", choices=["1", "2", "all"])
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--out", default=os.path.join("output", "bench"))
    ap.add_argument("-j", "--n-jobs", type=int, default=-1, help="число процессов; -1 = все ядра")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--no-tune", action="store_true", help="не подбирать гиперпараметры, взять значения по умолчанию")
    ap.add_argument("--primary", default=PRIMARY)
    ap.add_argument("--test-start", default=TEST_START)
    ap.add_argument("--test-end", default=TEST_END)
    ap.add_argument("--val-step", type=int, default=VAL_STEP_DAYS)
    ap.add_argument("--max-train-days", type=int, default=MAX_TRAIN_DAYS)
    ap.add_argument("--methods", default=None, help="через запятую — ограничить набор методов")
    return ap.parse_args()


def label(code):
    return REGION_NAMES.get(code, code)


def main():
    a = parse()
    os.makedirs(a.out, exist_ok=True)
    tasks = ["1", "2"] if a.task == "all" else [a.task]

    print("=" * 78)
    print("ЕДИНЫЙ ПРОТОКОЛ ОЦЕНКИ PROPHET: ПРОГНОЗ ПИКОВЫХ ЧАСОВ")
    print(f"Prophet: {'настоящий' if _HAS_REAL_PROPHET else 'НЕТ — считается Fourier-GAM фолбэком, это НЕ Prophet!'}")
    print("=" * 78)

    # ---------------- данные ----------------
    files = B.discover_region_files(a.data_dir)
    if a.primary not in files:
        raise SystemExit(f"Не найден electricity_{a.primary}.csv в '{a.data_dir}'. Найдено: {list(files)}")
    if "2" in tasks and len(files) < 2:
        raise SystemExit("Для задачи 2 нужно минимум 2 региона (electricity_<код>.csv). Найден один.")
    all_files = files if "2" in tasks else {a.primary: files[a.primary]}
    cons, price = B.load_panel(all_files, os.path.join(a.out, "cache"), TARGET_COL, ENCODING)
    regions = list(cons.columns)
    print(f"\nРегионы: {[label(r) for r in regions]}; период {cons.index.min()} .. {cons.index.max()}")

    last_full = cons.index.max().normalize() - (pd.Timedelta(days=1) if cons.index.max().hour != 23 else pd.Timedelta(0))
    t0, t1 = pd.Timestamp(a.test_start), min(pd.Timestamp(a.test_end), last_full)
    if t1 < t0:
        raise SystemExit(f"В данных нет фактов за тестовый период (с {a.test_start}); последние полные сутки — {last_full.date()}.")
    if t1 < pd.Timestamp(a.test_end):
        print(f"[!] В данных нет фактов до {a.test_end}; тест обрезан до {t1.date()}")
    if t0 < cons.index.min() + pd.Timedelta(days=MIN_TRAIN_DAYS):
        raise SystemExit("Перед первым тестовым днём нужно >= 180 дней истории.")
    test_days = pd.date_range(t0, t1, freq="D")
    val_end = min(pd.Timestamp(VAL_END), t0 - pd.Timedelta(days=1))
    val_days = pd.date_range(VAL_START, val_end, freq=f"{a.val_step}D")
    val_days = val_days[val_days >= cons.index.min() + pd.Timedelta(days=MIN_TRAIN_DAYS)]
    if len(val_days) == 0 and not a.no_tune:
        raise SystemExit("Нет валидационных дат до теста — проверь VAL_START/VAL_END или используй --no-tune.")
    print(f"Тест: {len(test_days)} дн. {t0.date()}..{t1.date()} | валидация: {len(val_days)} дат "
          f"{val_days.min().date() if len(val_days) else '-'}..{val_days.max().date() if len(val_days) else '-'} "
          f"(шаг {a.val_step} дн.)")

    # ---------------- пространственная структура (по данным ДО теста) ----------------
    corr, neighbors = B.spatial_neighbors(cons, t0 - pd.Timedelta(hours=1), min(N_NEIGHBORS, max(1, len(regions) - 1)))
    if len(regions) > 1:
        corr.to_csv(os.path.join(a.out, "spatial_corr.csv"))
        B.plot_corr(corr, os.path.join(a.out, "spatial_corr.png"))
        print("Соседи (по корреляции остатков до теста):", {label(r): [label(n) for n in v] for r, v in neighbors.items()})

    ctx = B.Context(cons, price, neighbors, a.max_train_days, SEED)

    # ---------------- методы к запуску ----------------
    plan = []                                          # (задача, метод, регионы)
    for t in tasks:
        for m in TASK_METHODS[t]:
            if a.methods and m not in a.methods.split(","):
                continue
            plan.append((t, m, regions if t == "2" else [a.primary]))
    uniq_methods = sorted({m for _, m, _ in plan})

    # ---------------- подбор гиперпараметров на валидации ----------------
    tuned_path = os.path.join(a.out, "tuned_params.json")
    meta = {"val": [VAL_START, str(val_end.date()), a.val_step], "max_train_days": a.max_train_days}
    tuned = {}
    if os.path.exists(tuned_path) and not a.no_resume:
        saved = json.load(open(tuned_path))
        if saved.get("_meta") == meta:
            tuned = saved
        else:
            print("[!] Настройки валидации/окна изменились — подбор параметров выполняется заново")
    tuned["_meta"] = meta
    tune_logs = []
    if not a.no_tune:
        for m in uniq_methods:
            if m in tuned or not B.REGISTRY[m][0].PARAM_GRID:
                continue
            regs_for_tune = regions if m == "prophet_nbr" else [a.primary]
            print(f"\n[TUNE] {m} на {len(val_days)} валидационных датах, регионы {regs_for_tune}")
            best, log = B.tune_method(m, ctx, val_days, regs_for_tune, a.n_jobs)
            tuned[m] = best
            tune_logs.append(log)
            json.dump(tuned, open(tuned_path, "w"), ensure_ascii=False, indent=2)
        if tune_logs:
            old = os.path.join(a.out, "tuning_log.csv")
            pd.concat(([pd.read_csv(old)] if os.path.exists(old) else []) + tune_logs).to_csv(old, index=False)
    json.dump(tuned, open(tuned_path, "w"), ensure_ascii=False, indent=2)

    # ---------------- основной расчёт (тест) ----------------
    print("\n[RUN] прогноз на тестовый период")
    for t, m, regs in plan:
        print(f"\n--- задача {t}: {m}")
        B.run_method(m, ctx, test_days, regs, a.out, tuned.get(m), a.n_jobs, resume=not a.no_resume)

    analyze(a, tasks, plan, regions, corr, neighbors, tuned, t0, t1, val_days)


# =============================================================================
# АНАЛИЗ И ВЫВОД
# =============================================================================
# (ключ колонки, подпись). None в ключе — заголовок блока.
METRIC_ROWS = [
    (None, "ПОПАДАНИЕ В ТОП-3 ЧАСА СУТОК"),
    ("top3_hit", "Hit rate топ-3, % (угадано часов из 3, точное совпадение)"),
    ("top3_tol1", "Hit rate топ-3 с допуском +-1 час, %"),
    ("all3_days", "Дней, где угаданы все 3 часа, %"),
    ("zero_days", "Дней, где не угадан ни один час, %"),
    ("top1_hit", "Угадан час суточного максимума, %"),
    ("top1_tol1", "Угадан час максимума с допуском +-1 час, %"),
    ("capture", "Доля пиковой нагрузки в выбранных часах, % (100 = идеал)"),
    (None, "ОШИБКА В ПИКОВЫХ ЧАСАХ (фактические топ-3 часа суток)"),
    ("peak_MAE", "MAE в пиковые часы, МВт*ч"),
    ("peak_MAPE", "MAPE в пиковые часы, %"),
    ("peak_bias", "Смещение в пиковые часы, МВт*ч (меньше 0: пики занижены)"),
    ("dmax_MAPE", "Ошибка суточного максимума, % (MAPE)"),
    (None, "ОШИБКА ПО ВСЕМ ЧАСАМ"),
    ("MAE", "MAE, МВт*ч"),
    ("RMSE", "RMSE, МВт*ч"),
    ("MAPE", "MAPE, %"),
    ("sMAPE", "sMAPE, %"),
    ("coverage80", "Покрытие 80% интервала, % (идеал около 80)"),
    ("days", "Число дней в оценке"),
]
LBL_W = 66
PAIR_NAMES = {"hits": "hits (угадано часов из 3 за день)", "hits_tol": "hits_tol (то же, допуск +-1 ч)",
              "top1": "top1 (угадан час максимума, доля дней)", "capture": "capture (доля пиковой нагрузки, %)",
              "peak_mae": "peak_MAE (МВт*ч)", "peak_mape": "peak_MAPE (%)", "mae": "MAE (МВт*ч)", "mape": "MAPE (%)"}


def fmt_metrics(df: pd.DataFrame) -> str:
    """df: одна строка на метод, колонки — метрики. Метрики идут строками, методы — столбцами."""
    methods = sorted(df["method"], key=lambda x: ("[" in x, 0))      # варианты «ранг по upper» — в конец
    w = max(15, max(len(m) for m in methods) + 2)
    lines = [" " * LBL_W + "".join(m.rjust(w) for m in methods)]
    for key, lab in METRIC_ROWS:
        if key is None:
            lines += ["", lab]
            continue
        vals = []
        for m in methods:
            v = df.loc[df["method"] == m, key].iloc[0]
            vals.append("-" if pd.isna(v) else (f"{int(round(v))}" if key == "days" else f"{v:.2f}"))
        lines.append("  " + lab.ljust(LBL_W - 2) + "".join(v.rjust(w) for v in vals))
    return "\n".join(lines)


def fmt_pairs(pt: pd.DataFrame) -> str:
    lines = []
    for _, r in pt.iterrows():
        lines.append(f"  {PAIR_NAMES.get(r['метрика'], r['метрика']):<42} {r['база']} -> {r['новый']}  [{label(r['регион']) if r['регион'] != 'ВСЕ' else 'все регионы'}]\n"
                     f"      среднее {r['среднее_база']:.3f} -> {r['среднее_новый']:.3f};  разница {r['diff']:+.3f}  "
                     f"95% ДИ [{r['ci_lo']:+.3f}; {r['ci_hi']:+.3f}];  p(Уилкоксон)={r['p']:.3f};  дней {int(r['n'])}\n"
                     f"      вывод: {r['вывод(95% ДИ)']}")
    return "\n".join(lines)


def bar(title: str) -> str:
    return "\n" + "=" * 100 + f"\n{title}\n" + "=" * 100 + "\n"


def analyze(a, tasks, plan, regions, corr, neighbors, tuned, t0, t1, val_days):
    engine = "Prophet" if _HAS_REAL_PROPHET else "Fourier-GAM (НЕ Prophet!)"
    fc_all = B.load_forecasts(a.out, {m for _, m, _ in plan})
    fc_all = fc_all[(fc_all["ds"] >= t0) & (fc_all["ds"] < t1 + pd.Timedelta(days=1))]       # только тестовый период
    txt = []

    # ---------------- шапка ----------------
    params = {k: v for k, v in tuned.items() if k != "_meta"}
    txt.append("ПРОГНОЗ ТОП-3 ПИКОВЫХ ЧАСОВ ЭНЕРГОПОТРЕБЛЕНИЯ — PROPHET, ЕДИНЫЙ ПРОТОКОЛ ОЦЕНКИ")
    txt.append(f"""
Протокол
  Прогноз: на сутки вперёд, точка прогноза 00:00; на каждый день {t0.date()} .. {t1.date()} ({(t1 - t0).days + 1} дн.) обучение
  на всей истории строго до 00:00 этого дня (расширяющееся окно, переобучение каждый день).
  Окно обучения: {'вся история' if a.max_train_days is None else str(a.max_train_days) + ' дн'}. Предобработка одна для всех вариантов.
  Подбор параметров: {len(val_days)} дат {VAL_START}..{VAL_END} (шаг {a.val_step} дн.), строго до теста; критерий MAPE + MAPE в пиковые часы.
  Движок: {engine}.
  Варианты: prophet_current - как в прежнем коде (цена ТОГО ЖЕ часа: в момент прогноза её нет, утечка, для сравнения);
            prophet         - честный: цена сутки назад;
            prophet_nbr     - prophet + регрессор: потребление соседних регионов сутки назад.
  Метрики считаются только на днях, спрогнозированных всеми сравниваемыми вариантами.
  Выбранные параметры: {json.dumps(params, ensure_ascii=False)}
  Hit rate топ-3 = доля часов из 3 предсказанных пиковых, совпавших с 3 фактическими пиковыми часами суток.""")

    # ---------------- задача 1 ----------------
    one = list(dict.fromkeys(m for t, m, _ in plan if t == "1"))
    if one:
        fc = B.common_days(fc_all[(fc_all["region"] == a.primary) & fc_all["method"].isin(one)])
        if not fc.empty:
            daily = B.evaluate_forecasts(fc, TOP_K)
            summ = B.summarize(daily, TOP_K)
            summ.to_csv(os.path.join(a.out, "task1_summary.csv"), index=False)
            daily.to_csv(os.path.join(a.out, "task1_daily.csv"), index=False)
            B.plot_summary_bars(summ, os.path.join(a.out, "task1_summary.png"),
                                f"{label(a.primary)}: {daily['date'].nunique()} дн.")
            pt = pd.DataFrame()
            if {"prophet_current", "prophet"} <= set(daily["method"]):
                pt = B.pairs_table(daily, [("prophet_current", "prophet")], regions=[a.primary])
                pt = pt[pt["регион"] == a.primary]
                pt.to_csv(os.path.join(a.out, "task1_paired.csv"), index=False)
            txt.append(bar(f"ЗАДАЧА 1. РЕГИОН {label(a.primary)}: {daily['date'].nunique()} общих дней"))
            txt.append(fmt_metrics(summ))
            if len(pt):
                txt.append("\nПарное сравнение по дням: prophet_current (прежний код) -> prophet (честный). "
                           "Разница = новый минус база; ДИ - блочный бутстрэп.\n" + fmt_pairs(pt))
                txt.append("\nПояснение: если честный prophet выглядит хуже прежнего - это не ухудшение модели, а исчезнувшая "
                           "утечка цены того же часа.")

    # ---------------- задача 2 ----------------
    if "2" in tasks and len(regions) > 1:
        fc = B.common_days(fc_all[fc_all["method"].isin(TASK_METHODS["2"])])
        if not fc.empty:
            daily = B.evaluate_forecasts(fc, TOP_K)
            summ = B.summarize(daily, TOP_K)
            s_all = B.summarize_all(summ)
            summ.to_csv(os.path.join(a.out, "task2_summary_by_region.csv"), index=False)
            s_all.to_csv(os.path.join(a.out, "task2_summary_all.csv"), index=False)
            B.plot_summary_bars(s_all, os.path.join(a.out, "task2_summary.png"),
                                f"Prophet: локальный vs с соседями, {len(regions)} регионов")
            pt = B.pairs_table(daily, SPATIAL_PAIRS)
            pt.to_csv(os.path.join(a.out, "task2_spatial_effect.csv"), index=False)
            for met in ("hits", "peak_mape", "mape"):
                B.plot_spatial_delta(pt, met, os.path.join(a.out, f"task2_delta_{met}.png"))

            txt.append(bar("ЗАДАЧА 2. ПРОСТРАНСТВЕННЫЙ ФАКТОР: prophet (локальный) vs prophet_nbr (с соседями)"))
            txt.append("Соседи каждого региона (по корреляции остатков потребления, считалась только по данным до теста):")
            for r, nb in neighbors.items():
                txt.append(f"  {label(r)}: {', '.join(label(n) for n in nb)}")
            txt.append("\nКорреляция остатков:\n" + corr.round(2).to_string())
            txt.append(f"\n--- СРЕДНЕЕ ПО РЕГИОНАМ ({len(regions)} шт.) ---\n" + fmt_metrics(s_all))
            txt.append("\n--- ЭФФЕКТ СОСЕДЕЙ ---\nРазница = prophet_nbr минус prophet. Для hits/hits_tol/top1/capture лучше БОЛЬШЕ, для остальных МЕНЬШЕ.")
            for met in ("hits", "peak_mape", "mape"):
                sub = pt[(pt["метрика"] == met) & (pt["регион"] != "ВСЕ")]
                n = len(sub)
                cnt = sub["вывод(95% ДИ)"].value_counts()
                txt.append(f"  {PAIR_NAMES[met]}: лучше в {cnt.get('лучше', 0)} из {n} регионов, хуже в {cnt.get('хуже', 0)}, "
                           f"без значимой разницы в {cnt.get('без значимой разницы', 0)}"
                           + (f", вывод не делался (мало дней) в {n - cnt.get('лучше', 0) - cnt.get('хуже', 0) - cnt.get('без значимой разницы', 0)}"
                              if n - cnt.get('лучше', 0) - cnt.get('хуже', 0) - cnt.get('без значимой разницы', 0) else ""))
            txt.append("\nПодробно по регионам и метрикам:\n" + fmt_pairs(pt))
            for r in regions:
                part = summ[summ["region"] == r]
                txt.append(f"\n--- РЕГИОН {label(r)} ---\n" + fmt_metrics(part))

    # ---------------- графики по дням ----------------
    pdfs = []
    for r in regions:
        sub = fc_all[(fc_all["region"] == r) & fc_all["method"].isin({m for _, m, rs in plan if r in rs})]
        sub = B.common_days(sub) if not sub.empty else sub
        if sub.empty:
            continue
        pdf = os.path.join(a.out, f"forecast_by_day_{r}.pdf")
        B.plot_daily_forecasts_pdf(sub, pdf, f"{label(r)}: прогноз и факт по часам", main_method="prophet", k=TOP_K)
        pdfs.append(pdf)

    txt.append(bar("ФАЙЛЫ"))
    txt.append("  results.txt (этот файл), графики по дням: " + ", ".join(os.path.basename(p) for p in pdfs))
    txt.append("  таблицы в csv: task1_*.csv, task2_*.csv; почасовые прогнозы: forecasts/")
    path = os.path.join(a.out, "results.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(txt) + "\n")
    print(f"\n[DONE] Метрики и сравнение: {os.path.abspath(path)}")
    for p in pdfs:
        print(f"       Графики по дням:    {os.path.abspath(p)}")



if __name__ == "__main__":      # обязательно для n_jobs > 1 (особенно на Windows)
    main()