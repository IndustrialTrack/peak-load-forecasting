"""Сбор почасовых данных MapPartial для Челябинской области."""

import csv
import logging
import os
import time
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

BASE_URL = "https://br.so-ups.ru/webapi/api/map/MapPartial"
POWER_SYSTEM_ID = 630000
SUBJECT_ID = "71"
REGION_NAME = "Тюменская область"
START_DATE = date(2022, 1, 1)
MOSCOW_TZ = ZoneInfo("Europe/Moscow")
OUTPUT_FILE = f"electricity_{SUBJECT_ID}.csv"
LOG_FILE = f"electricity_{SUBJECT_ID}.log"

REQUEST_TIMEOUT = 30
MAX_RETRIES = 5
REQUEST_DELAY = 0.6

CSV_COLUMNS = [
    "date", "hour", "region_name", "planned_consumption",
    "actual_consumption", "planned_generation", "actual_generation",
    "average_price", "vsvgo_consumption", "vsvgo_avg_price", "fetched_at",
]

FIELDS = {
    "region_name": "Name",
    "planned_consumption": "IBR_PlannedConsumption",
    "actual_consumption": "IBR_ActualConsumption",
    "planned_generation": "IBR_PlannedGeneration",
    "actual_generation": "IBR_ActualGeneration",
    "average_price": "IBR_AveragePrice",
    "vsvgo_consumption": "VSVGO_Consumption",
    "vsvgo_avg_price": "VSVGO_AveragePrice",
}

ENERGY_FIELDS = {
    "planned_consumption", "actual_consumption",
    "planned_generation", "actual_generation", "vsvgo_consumption",
}

session = requests.Session()
session.verify = False
session.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "ru-RU,ru;q=0.9",
    "Referer": "https://br.so-ups.ru/",
    "Origin": "https://br.so-ups.ru",
})


def setup_logger():
    """Настраивает вывод журнала одновременно в терминал и в файл."""
    logger = logging.getLogger("electricity_collector")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    for handler in (
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


logger = setup_logger()


def parse_value(value):
    """Преобразует «4 277 МВт*ч», «1 556 руб./МВт*ч» или «-» в число."""
    if value is None or value == "-":
        return None
    if isinstance(value, (int, float)):
        return float(value)

    value = str(value).strip().replace("\u00a0", "").replace(" ", "")
    value = value.replace(",", ".")
    number = []
    decimal_seen = False
    for char in value:
        if char.isdigit() or (char == "-" and not number):
            number.append(char)
        elif char == "." and not decimal_seen:
            number.append(char)
            decimal_seen = True
        else:
            break
    try:
        return float("".join(number))
    except ValueError:
        return None


def format_value(value, unit):
    """Форматирует значение для итоговой CSV так же, как в отчёте СО ЕЭС."""
    number = parse_value(value)
    if number is None:
        return ""

    # Разделитель тысяч — обычный пробел, чтобы файл был удобен и в Excel.
    if number.is_integer():
        text = f"{int(number):,}".replace(",", " ")
    else:
        text = f"{number:,.2f}".replace(",", " ").rstrip("0").rstrip(".")
    return f"{text} {unit}"


def format_fetched_at(value):
    """Убирает из ISO-времени дробные секунды и смещение часового пояса."""
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    try:
        return datetime.fromisoformat(str(value)).strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return str(value)


def create_csv():
    """Создаёт файл и записывает заголовок только при первом запуске."""
    if os.path.exists(OUTPUT_FILE):
        return
    # BOM помогает Excel под Windows автоматически распознать UTF-8.
    with open(OUTPUT_FILE, "w", encoding="utf-8-sig", newline="") as file:
        csv.DictWriter(file, fieldnames=CSV_COLUMNS).writeheader()


def migrate_existing_csv():
    """Приводит к новому формату записи, созданные предыдущими запусками."""
    if not os.path.exists(OUTPUT_FILE):
        return

    with open(OUTPUT_FILE, "r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        return

    # Признак уже отформатированного файла: часы имеют вид 00, а значения — единицы.
    sample = rows[0]
    if (
        len(sample.get("hour", "")) == 2
        and "МВт*ч" in sample.get("planned_consumption", "")
        and sample.get("fetched_at", "").count("T") == 0
    ):
        return

    for row in rows:
        try:
            row["hour"] = f"{int(row.get('hour', 0)):02d}"
        except (TypeError, ValueError):
            pass
        for field in ENERGY_FIELDS:
            row[field] = format_value(row.get(field), "МВт*ч")
        for field in {"average_price", "vsvgo_avg_price"}:
            row[field] = format_value(row.get(field), "руб./МВт*ч")
        row["fetched_at"] = format_fetched_at(row.get("fetched_at", ""))

    temporary_file = f"{OUTPUT_FILE}.tmp"
    with open(temporary_file, "w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary_file, OUTPUT_FILE)


def load_processed():
    """Возвращает уже сохранённые пары (дата, час)."""
    processed = set()
    if not os.path.exists(OUTPUT_FILE):
        return processed
    with open(OUTPUT_FILE, "r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            try:
                processed.add((row["date"], int(row["hour"])))
            except (KeyError, TypeError, ValueError):
                continue
    return processed


def fetch_hour(target_date, hour):
    """Запрашивает MainArea за указанный час либо возвращает None."""
    params = {
        "MapType": 0,
        "Date": target_date.isoformat(),
        "Hour": hour,
        "PowerSystemId": POWER_SYSTEM_ID,
        "SubjectId": SUBJECT_ID,
        "ServiceMode": "false",
    }
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = session.get(BASE_URL, params=params, timeout=REQUEST_TIMEOUT)
            if response.status_code in {429, 500, 502, 503, 504}:
                raise requests.HTTPError(f"HTTP {response.status_code}")
            response.raise_for_status()

            # Сервер отдаёт кириллицу: задаём верную кодировку до response.json().
            response.encoding = "utf-8"
            main_area = response.json().get("MainArea")
            if isinstance(main_area, dict):
                return main_area
            logger.warning("%s %02d:00: MainArea отсутствует", target_date, hour)
            return None
        except (requests.RequestException, ValueError) as error:
            if attempt == MAX_RETRIES:
                logger.error("%s %02d:00: %s", target_date, hour, error)
                return None
            wait_time = 2 ** (attempt - 1)
            logger.warning(
                "%s %02d:00: %s; повтор через %s с",
                target_date,
                hour,
                error,
                wait_time,
            )
            time.sleep(wait_time)
    return None


def make_row(target_date, hour, main_area):
    row = {
        "date": target_date.isoformat(),
        "hour": f"{hour:02d}",
        "fetched_at": format_fetched_at(datetime.now(MOSCOW_TZ)),
    }
    for csv_field, api_field in FIELDS.items():
        value = main_area.get(api_field)
        if csv_field == "region_name":
            row[csv_field] = value or REGION_NAME
        elif csv_field in ENERGY_FIELDS:
            row[csv_field] = format_value(value, "МВт*ч")
        else:
            row[csv_field] = format_value(value, "руб./МВт*ч")
    return row


def last_hour_for_date(target_date, today):
    return 23 if target_date < today else datetime.now(MOSCOW_TZ).hour - 1


def collect_data():
    create_csv()
    migrate_existing_csv()
    processed = load_processed()
    today = datetime.now(MOSCOW_TZ).date()
    current_date = START_DATE
    logger.info("Сбор данных: %s; записей уже есть: %s", REGION_NAME, len(processed))

    # Для добавления используем обычный UTF-8: BOM создаётся только один раз.
    with open(OUTPUT_FILE, "a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
        while current_date <= today:
            last_hour = last_hour_for_date(current_date, today)
            for hour in range(max(0, last_hour + 1)):
                key = (current_date.isoformat(), hour)
                if key in processed:
                    continue
                main_area = fetch_hour(current_date, hour)
                if main_area is not None:
                    row = make_row(current_date, hour, main_area)
                    writer.writerow(row)
                    file.flush()
                    processed.add(key)
                    logger.info(
                        "Сохранено: %s",
                        ",".join(row[column] for column in CSV_COLUMNS),
                    )
                time.sleep(REQUEST_DELAY)
            current_date += timedelta(days=1)


if __name__ == "__main__":
    try:
        collect_data()
    except KeyboardInterrupt:
        logger.warning("Сбор остановлен пользователем.")
