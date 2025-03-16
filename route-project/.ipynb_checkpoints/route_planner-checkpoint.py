import os
import json
import logging
import requests
import pandas as pd
import networkx as nx

from networkx.algorithms.simple_paths import shortest_simple_paths
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple
from time import sleep

# -----------------------------------------------------------------------------
# Конфигурация и логирование
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("RouteSystem")

# Файлы кэша для справочных данных (Модуль 1)
STATIONS_CACHE_JSON = "stations_list.json"
STATIONS_CACHE_CSV = "stations_list.csv"

# API-ключ для Яндекс.Расписаний (замените на свой реальный ключ)
YANDEX_RASP_API_KEY = "2a22163d-023f-47db-a974-6d683c413a83"

# URL для методов
URL_STATIONS_LIST = "https://api.rasp.yandex.net/v3.0/stations_list"
URL_SEARCH = "https://api.rasp.yandex.net/v3.0/search/"

# Файл с выбранными узлами (при интерактивном выборе)
SELECTED_NODES_FILE = "selected_nodes.json"

# Минимальное время пересадки (штраф)
MIN_TRANSFER_TIME = 1800  # 30 минут (секунд)

# -----------------------------------------------------------------------------
# Модуль 1: Загрузка и кэширование данных станций
# -----------------------------------------------------------------------------
def fetch_stations_list_from_api(api_key: str) -> dict:
    """Запрашивает список станций через API и возвращает JSON-словарь."""
    logger.info("Запрашиваем список станций из API Яндекс.Расписаний...")
    params = {
        "apikey": api_key,
        "lang": "ru_RU",
        "format": "json"
    }
    response = requests.get(URL_STATIONS_LIST, params=params)
    response.raise_for_status()
    data = response.json()
    logger.info("Список станций успешно получен из API.")
    return data

def stations_to_dataframe(stations_json: dict) -> pd.DataFrame:
    """
    Преобразует вложенную структуру (страны → регионы → населённые пункты → станции)
    в плоский DataFrame с дополнительными полями: country_title, region_title, settlement_title.
    Если у станции отсутствует поле 'type', устанавливает его в 'station'.
    """
    stations_list = []
    for country in stations_json.get("countries", []):
        country_title = country.get("title", "")
        for region in country.get("regions", []):
            region_title = region.get("title", "")
            for settlement in region.get("settlements", []):
                settlement_title = settlement.get("title", "")
                for station in settlement.get("stations", []):
                    station["country_title"] = country_title
                    station["region_title"] = region_title
                    station["settlement_title"] = settlement_title
                    if "type" not in station:
                        station["type"] = "station"
                    stations_list.append(station)
    df = pd.json_normalize(stations_list)
    return df

def load_stations_dataframe() -> (pd.DataFrame, dict):
    """
    Если кэш (CSV и JSON) существует, загружает данные. Иначе – запрашивает через API и сохраняет.
    Возвращает DataFrame и исходный JSON.
    """
    if os.path.exists(STATIONS_CACHE_CSV) and os.path.exists(STATIONS_CACHE_JSON):
        logger.info(f"Найден CSV‑кэш: {STATIONS_CACHE_CSV}")
        df = pd.read_csv(STATIONS_CACHE_CSV, encoding="utf-8", low_memory=False)
        with open(STATIONS_CACHE_JSON, "r", encoding="utf-8") as f:
            stations_json = json.load(f)
    else:
        logger.info("Кэш не найден. Запрашиваем данные из API...")
        stations_json = fetch_stations_list_from_api(YANDEX_RASP_API_KEY)
        with open(STATIONS_CACHE_JSON, "w", encoding="utf-8") as f:
            json.dump(stations_json, f, ensure_ascii=False, indent=2)
        logger.info(f"Сохранён JSON‑кэш: {STATIONS_CACHE_JSON}")
        df = stations_to_dataframe(stations_json)
        df.to_csv(STATIONS_CACHE_CSV, index=False, encoding="utf-8")
        logger.info(f"Сохранён CSV‑кэш: {STATIONS_CACHE_CSV}")
    return df, stations_json

# -----------------------------------------------------------------------------
# Модуль 2: Выбор транспортных узлов (фильтрация станций)
# -----------------------------------------------------------------------------
def load_settlements_dataframe() -> pd.DataFrame:
    """Загружает DataFrame для населённых пунктов из JSON‑кэша."""
    if not os.path.exists(STATIONS_CACHE_JSON):
        logger.error(f"Файл {STATIONS_CACHE_JSON} не найден.")
        return pd.DataFrame()
    with open(STATIONS_CACHE_JSON, "r", encoding="utf-8") as f:
        stations_json = json.load(f)
    settlements_list = []
    for country in stations_json.get("countries", []):
        for region in country.get("regions", []):
            for settlement in region.get("settlements", []):
                settlement_copy = settlement.copy()
                settlement_copy["country_title"] = country.get("title", "")
                settlement_copy["region_title"] = region.get("title", "")
                settlement_copy["settlement_title"] = settlement.get("title", "")
                if "type" not in settlement_copy:
                    settlement_copy["type"] = "settlement"
                settlements_list.append(settlement_copy)
    return pd.json_normalize(settlements_list)

def filter_airports_and_train_stations(df: pd.DataFrame, city: str) -> pd.DataFrame:
    """
    Фильтрует DataFrame по заданному городу (по столбцам settlement_title и title)
    и оставляет записи, где station_type равен "airport" или "train_station",
    исключая записи общего уровня (type == "settlement") и содержащие "Тур" в названии.
    """
    mask = (df["settlement_title"].str.contains(city, case=False, na=False)) | \
           (df["title"].str.contains(city, case=False, na=False))
    df_city = df[mask]
    df_city = df_city[df_city["type"].str.lower() != "settlement"]
    valid_types = ["airport", "train_station"]
    df_filtered = df_city[df_city["station_type"].str.lower().isin(valid_types)]
    df_filtered = df_filtered[~df_filtered["title"].str.contains("Тур", case=False, na=False)]
    return df_filtered

def choose_station_interactive(df: pd.DataFrame, city: str) -> str:
    """
    Для Москвы и Санкт-Петербурга выводит интерактивный список узлов и возвращает выбранный код.
    """
    df_filtered = filter_airports_and_train_stations(df, city)
    if df_filtered.empty:
        logger.warning(f"Для города '{city}' не найдено транспортных узлов.")
        return ""
    if len(df_filtered) == 1:
        chosen = df_filtered.iloc[0]
        code = chosen.get("codes.yandex_code") or chosen.get("code", "")
        logger.info(f"Единственный вариант для города '{city}': {code}")
        return code
    print(f"\nВ городе '{city}' найдено несколько вариантов транспортных узлов:")
    df_filtered = df_filtered.reset_index(drop=True)
    for idx, row in df_filtered.iterrows():
        station_name = row.get("title", "Неизвестно")
        station_type = row.get("station_type", "Неизвестно")
        region = row.get("region_title", "")
        print(f"{idx}: {station_name} (Тип: {station_type}, Регион: {region})")
    try:
        choice = int(input("Введите номер выбранного узла: ").strip())
        if 0 <= choice < len(df_filtered):
            chosen = df_filtered.loc[choice]
            code = chosen.get("codes.yandex_code") or chosen.get("code", "")
            logger.info(f"Выбранный узел для города '{city}': {code}")
            return code
        else:
            logger.error("Неверный выбор: номер вне диапазона.")
            return ""
    except Exception as e:
        logger.error(f"Ошибка ввода: {e}")
        return ""

def get_city_code_automatic(df_settlements: pd.DataFrame, city: str, df_stations: pd.DataFrame) -> str:
    """
    Для городов, отличных от Москвы и Санкт-Петербурга, возвращает случайный код транспортного узла,
    где код начинается с "s".
    """
    if "codes.yandex_code" in df_stations.columns:
        col = "codes.yandex_code"
    elif "code" in df_stations.columns:
        col = "code"
    else:
        logger.error("Столбец с кодом не найден.")
        return ""
    df_candidates = df_stations[
        (df_stations["settlement_title"].str.contains(city, case=False, na=False)) &
        (df_stations[col].str.startswith("s"))
    ]
    if not df_candidates.empty:
        candidate = df_candidates.sample(n=1).iloc[0]
        station_code = candidate[col]
        logger.info(f"Случайно выбран код города '{city}': {station_code}")
        return station_code
    else:
        logger.warning(f"Для города '{city}' нет записей с кодом, начинающимся на 's'.")
        return ""

def update_node_code(node: str, city: str) -> str:
    """
    Если node начинается с "c", ищет подходящий код станции (начинающийся с "s") для данного города.
    """
    if not node.startswith("c"):
        return node
    df_stations, _ = load_stations_dataframe()
    if "codes.yandex_code" in df_stations.columns:
        col = "codes.yandex_code"
    elif "code" in df_stations.columns:
        col = "code"
    else:
        logger.error("Столбец с кодом не найден.")
        return node
    df_candidate = df_stations[
        (df_stations["settlement_title"].str.contains(city, case=False, na=False)) &
        (df_stations[col].str.startswith("s"))
    ]
    if not df_candidate.empty:
        candidate_station = df_candidate.iloc[0][col]
        logger.info(f"Обновляем код для города {city}: заменяем {node} на {candidate_station}")
        return candidate_station
    else:
        logger.warning(f"Подходящий код для {city} не найден, остаётся {node}")
        return node

def select_transport_node(city: str, df_stations: pd.DataFrame, df_settlements: pd.DataFrame) -> str:
    """
    Выбирает транспортный узел для заданного города:
      - Для Москвы и СПб – интерактивный выбор.
      - Для остальных – автоматический выбор.
    После выбора, если код начинается с "c", обновляет его через update_node_code.
    """
    if city.lower() in ["москва", "санкт-петербург"]:
        node = choose_station_interactive(df_stations, city)
    else:
        node = get_city_code_automatic(df_settlements, city, df_stations)
    return update_node_code(node, city)

# -----------------------------------------------------------------------------
# Функция интерактивного ввода исходных данных
# -----------------------------------------------------------------------------
def input_route_parameters() -> dict:
    """
    Запрашивает у пользователя:
      - Город/код отправления
      - Город/код прибытия
      - Дату поездки (YYYY-MM-DD)
      - Опционально: промежуточный город и количество суток (1..7)
    Возвращает словарь с данными.
    """
    city_from = input("Введите город/код отправления: ").strip()
    city_to = input("Введите город/код прибытия: ").strip()
    date_str = input("Введите дату (YYYY-MM-DD): ").strip()
    try:
        dt_obj = datetime.strptime(date_str, "%Y-%m-%d")
        if dt_obj.date() < datetime.now().date():
            logger.error("Дата не может быть в прошлом.")
            return {}
    except ValueError:
        logger.error("Неверный формат даты. Ожидается YYYY-MM-DD.")
        return {}
    choice = input("Хотите сделать остановку в промежуточном городе? (Да/Нет): ").strip().lower()
    work_info = None
    if choice in ["да", "yes"]:
        wcity = input("Введите город/код промежуточного узла: ").strip()
        try:
            wdays = int(input("Сколько суток остановка (1..7): ").strip())
            if not (1 <= wdays <= 7):
                logger.error("Число суток должно быть от 1 до 7.")
                return {}
            work_info = (wcity, wdays)
        except ValueError:
            logger.error("Неверный ввод числа суток.")
            return {}
    return {
        "city_from": city_from,
        "city_to": city_to,
        "date": date_str,
        "work_info": work_info
    }

# -----------------------------------------------------------------------------
# Модуль 5: Получение вариантов маршрутов через API
# -----------------------------------------------------------------------------
def get_route_options(from_code: str, to_code: str, date: str,
                      transport_types: str = "plane,train,suburban,bus,water,helicopter",
                      transfers: str = "true", limit: int = 100, offset: int = 0) -> dict:
    """
    Отправляет запрос к API для получения вариантов маршрутов.
    Если статус 404, возвращает пустой словарь.
    """
    params = {
        "apikey": YANDEX_RASP_API_KEY,
        "format": "json",
        "from": from_code,
        "to": to_code,
        "lang": "ru_RU",
        "date": date,
        "transport_types": transport_types,
        "transfers": transfers,
        "limit": limit,
        "offset": offset
    }
    logger.info(f"Отправляем запрос: from={from_code}, to={to_code}, date={date}")
    r = requests.get(URL_SEARCH, params=params)
    try:
        r.raise_for_status()
    except requests.exceptions.HTTPError as e:
        if r.status_code == 404:
            logger.warning(f"Маршруты не найдены для запроса: {params}. Возвращаем пустой словарь.")
            return {}
        else:
            logger.error(f"HTTP ошибка: {e} (Статус: {r.status_code})")
            raise
    logger.info("Маршруты успешно получены.")
    return r.json()

def save_route_options(route_data: dict, filename: str) -> None:
    """
    Сохраняет данные маршрутов в JSON и CSV.
    """
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(route_data, f, ensure_ascii=False, indent=2)
    logger.info(f"JSON сохранён: {filename}")
    if "segments" in route_data:
        df = pd.json_normalize(route_data["segments"])
        csv_name = filename.replace(".json", ".csv")
        df.to_csv(csv_name, index=False, encoding="utf-8")
        logger.info(f"CSV сохранён: {csv_name}")

# -----------------------------------------------------------------------------
# Модуль 6: Построение графа маршрутов и оптимизация
# -----------------------------------------------------------------------------
def combine_route_data(route_data_1: dict, route_data_2: dict, from_node: str, to_node: str) -> dict:
    """
    Объединяет сегменты двух запросов (например, A->X и X->B) в один словарь.
    Сегментам первой части присваивает query=1, второй – query=2.
    """
    seg1 = route_data_1.get("segments", [])
    seg2 = route_data_2.get("segments", [])
    for s in seg1:
        s["query"] = 1
    for s in seg2:
        s["query"] = 2
    combined = seg1 + seg2
    logger.info(f"Объединено сегментов: {len(seg1)} + {len(seg2)} = {len(combined)}")
    return {
        "segments": combined,
        "search": {
            "from": {"code": from_node},
            "to": {"code": to_node}
        }
    }

def build_route_graph(route_data: dict, intermediate_node: str = None) -> nx.DiGraph:
    """
    Извлекает сегменты из route_data и строит ориентированный граф.
    Если задан intermediate_node, то для сегментов query=2 с вылетом из него,
    если интервал меньше MIN_TRANSFER_TIME, добавляется штраф.
    """
    G = nx.DiGraph()
    segments = route_data.get("segments", [])
    interval_segments = route_data.get("interval_segments", [])
    all_segments = segments + interval_segments
    logger.info(f"Начало построения графа: найдено {len(all_segments)} сегментов (с учетом interval_segments)")

    min_arrival = None
    if intermediate_node:
        for s in all_segments:
            if s.get("query") == 1:
                to_st = s.get("to", {})
                if to_st.get("code") == intermediate_node:
                    arr_str = s.get("arrival")
                    if arr_str:
                        try:
                            arr_dt = datetime.fromisoformat(arr_str)
                            if (min_arrival is None) or (arr_dt < min_arrival):
                                min_arrival = arr_dt
                        except ValueError:
                            pass
        if min_arrival:
            logger.info(f"Минимальное время прибытия в {intermediate_node}: {min_arrival.isoformat()}")

    for s in all_segments:
        try:
            from_st = s.get("from") or s.get("departure_from") or {}
            to_st = s.get("to") or s.get("arrival_to") or {}
            from_code = from_st.get("code")
            to_code = to_st.get("code")
            if not from_code or not to_code:
                continue
            duration = s.get("duration")
            if duration is None:
                dep_str = s.get("departure")
                arr_str = s.get("arrival")
                if dep_str and arr_str:
                    try:
                        dep_dt = datetime.fromisoformat(dep_str)
                        arr_dt = datetime.fromisoformat(arr_str)
                        diff_sec = (arr_dt - dep_dt).total_seconds()
                        if diff_sec < 0:
                            continue
                        duration = diff_sec
                    except Exception:
                        continue
                else:
                    continue
            if intermediate_node and s.get("query") == 2 and from_code == intermediate_node and min_arrival:
                dep_str = s.get("departure")
                if dep_str:
                    try:
                        dep_dt = datetime.fromisoformat(dep_str)
                        gap = (dep_dt - min_arrival).total_seconds()
                        if gap < MIN_TRANSFER_TIME:
                            penalty = MIN_TRANSFER_TIME - gap
                            logger.info(f"Добавляем штраф {penalty} сек для сегмента с отправлением {dep_str}")
                            duration += penalty
                    except Exception:
                        pass
            G.add_edge(from_code, to_code, weight=duration, segment=s)
        except Exception as e:
            logger.error(f"Ошибка при обработке сегмента: {e}")
    logger.info(f"Граф построен: узлов={G.number_of_nodes()}, рёбер={G.number_of_edges()}")
    return G

def find_k_shortest_paths(G: nx.DiGraph, source: str, target: str, k: int = 10) -> List[Tuple[List[str], float]]:
    """
    Находит до k кратчайших путей от source до target.
    Использует shortest_simple_paths (пути уже отсортированы по возрастанию веса).
    Возвращает список кортежей (список узлов, суммарное время).
    """
    if source not in G or target not in G:
        logger.warning(f"Узел {source} или {target} отсутствует в графе.")
        return []
    paths_gen = shortest_simple_paths(G, source, target, weight="weight")
    results = []
    for path in paths_gen:
        cost = 0
        for u, v in zip(path, path[1:]):
            cost += G[u][v]["weight"]
        results.append((path, cost))
        if len(results) >= k:
            break
    return results

# -----------------------------------------------------------------------------
# Модуль 7: Формирование DataFrame с детализацией маршрута
# -----------------------------------------------------------------------------
def format_duration(seconds: float) -> str:
    """Форматирует секунды в строку вида 'Xч Yм Zс'."""
    if seconds is None:
        return ""
    hh = int(seconds // 3600)
    mm = int((seconds % 3600) // 60)
    ss = int(seconds % 60)
    return f"{hh}ч {mm}м {ss}с"

def build_optimal_route_dataframe(route: List[str], graph: nx.DiGraph, work_days: int = 0, route_id: int = 1) -> pd.DataFrame:
    """
    Создаёт DataFrame с детализацией маршрута:
      - Маршрут_ID, Номер сегмента, коды отправления/прибытия, названия, времена, количество дней, время в пути, вид транспорта.
    """
    rows = []
    for i in range(len(route) - 1):
        from_node = route[i]
        to_node = route[i + 1]
        edge_data = graph.get_edge_data(from_node, to_node, default={})
        seg = edge_data.get("segment", {}) if edge_data else {}
        from_st = seg.get("from", {})
        to_st = seg.get("to", {})
        from_code = from_st.get("code", from_node)
        to_code = to_st.get("code", to_node)
        from_title = from_st.get("title", "")
        to_title = to_st.get("title", "")
        dep_time = seg.get("departure", "")
        arr_time = seg.get("arrival", "")
        duration_seconds = None
        if dep_time and arr_time:
            try:
                dep_dt = datetime.fromisoformat(dep_time)
                arr_dt = datetime.fromisoformat(arr_time)
                duration_seconds = (arr_dt - dep_dt).total_seconds()
            except Exception:
                pass
        dur_str = format_duration(duration_seconds)
        transport_type = ""
        if "thread" in seg:
            transport_type = seg["thread"].get("transport_type", "")
        else:
            transport_type = seg.get("transport_type", "")
        rows.append({
            "Маршрут_ID": route_id,
            "Номер сегмента маршрута": i + 1,
            "Код отправления": from_code,
            "Город отправления": from_title,
            "Время отправления": dep_time,
            "Код прибытия": to_code,
            "Город прибытия": to_title,
            "Время прибытия": arr_time,
            "Количество дней": work_days,
            "Время в пути": dur_str,
            "Вид транспорта": transport_type
        })
    return pd.DataFrame(rows)

# -----------------------------------------------------------------------------
# Основная функция: объединённый запуск с интерактивным вводом и 10 маршрутами
# -----------------------------------------------------------------------------
def main():
    logger.info("=== Запуск объединённого модуля (10 маршрутов) ===")
    
    # Шаг 1: Ввод исходных данных
    params = input_route_parameters()
    if not params:
        logger.error("Ошибка ввода данных. Завершаем работу.")
        return
    city_from = params["city_from"]
    city_to = params["city_to"]
    date_str = params["date"]
    work_info = params.get("work_info")  # либо (work_city, work_days), либо None

    # Шаг 2: Загрузка справочных данных
    df_stations, _ = load_stations_dataframe()
    df_settlements = load_settlements_dataframe()
    if df_stations.empty or df_settlements.empty:
        logger.error("Нет справочных данных. Завершаем.")
        return
    logger.info(f"Всего станций: {len(df_stations)}")
    
    # Шаг 3: Выбор транспортных узлов (интерактивно)
    from_node = select_transport_node(city_from, df_stations, df_settlements)
    to_node = select_transport_node(city_to, df_stations, df_settlements)
    work_node = ""
    work_days = 0
    if work_info:
        work_city, work_days = work_info
        print(f"\n--- Выбор узла для промежуточного города '{work_city}' ---")
        work_node = select_transport_node(work_city, df_stations, df_settlements)
    
    if not from_node or not to_node:
        logger.error("Не удалось выбрать транспортные узлы для одного из городов.")
        return

    # Сохраняем выбранные узлы в файл
    selected_nodes = {"from_node": from_node, "to_node": to_node}
    if work_info:
        selected_nodes["work_node"] = work_node
    with open(SELECTED_NODES_FILE, "w", encoding="utf-8") as f:
        json.dump(selected_nodes, f, ensure_ascii=False, indent=2)
    logger.info(f"Выбранные узлы сохранены в {SELECTED_NODES_FILE}")
    
    print("\nИсходные данные маршрута:")
    print(f"Отправление: {city_from} -> Узел: {from_node}")
    print(f"Прибытие: {city_to} -> Узел: {to_node}")
    print(f"Дата поездки: {date_str}")
    if work_info:
        print(f"Промежуточный город: {work_city} -> Узел: {work_node}, суток: {work_days}")
    
    # Шаг 4: Получение вариантов маршрутов через API
    if work_node:
        logger.info("Запрашиваем сегмент 1 (A->X)")
        route_data_1 = get_route_options(from_node, work_node, date_str)
        if not route_data_1.get("segments"):
            logger.error("Первый сегмент: отсутствуют сегменты.")
            print("Маршрут не найден.")
            return
        first_seg = route_data_1["segments"][0]
        arrival_str = first_seg.get("arrival")
        if not arrival_str:
            logger.error("Первый сегмент: отсутствует время прибытия.")
            print("Маршрут не найден.")
            return
        arrival_dt = datetime.fromisoformat(arrival_str)
        new_date_str = (arrival_dt + timedelta(days=work_days)).strftime("%Y-%m-%d")
        logger.info("Запрашиваем сегмент 2 (X->B)")
        route_data_2 = get_route_options(work_node, to_node, new_date_str)
        if not route_data_2.get("segments"):
            logger.error("Второй сегмент: отсутствуют сегменты.")
            print("Маршрут не найден.")
            return
        final_data = combine_route_data(route_data_1, route_data_2, from_node, to_node)
    else:
        logger.info("Запрашиваем прямой сегмент (A->B)")
        final_data = get_route_options(from_node, to_node, date_str)
        if not final_data.get("segments"):
            logger.error("Маршрут не найден (пустой ответ).")
            print("Маршрут не найден.")
            return

    # Сохраняем полученные данные маршрута
    fname = f"search_{from_node}_{to_node}_{date_str}.json".replace(" ", "_")
    save_route_options(final_data, fname)
    
    # Шаг 5: Построение графа маршрутов
    intermediate = work_node if work_node else None
    graph = build_route_graph(final_data, intermediate_node=intermediate)
    if graph.number_of_nodes() == 0:
        logger.error("Граф пуст. Нет валидных сегментов.")
        print("Маршрут не найден.")
        return
    if from_node not in graph.nodes or to_node not in graph.nodes:
        logger.error(f"Узел {from_node} или {to_node} отсутствует в графе.")
        print("Маршрут не найден.")
        return

    # Шаг 6: Поиск до 10 кратчайших путей
    k = 10
    routes = find_k_shortest_paths(graph, from_node, to_node, k=k)
    if not routes:
        logger.error("Не найдено ни одного пути.")
        print("Маршрут не найден.")
        return

    # Шаг 7: Формирование общего DataFrame с детализацией каждого маршрута
    all_frames = []
    for idx, (path_nodes, total_time) in enumerate(routes, start=1):
        logger.info(f"[Путь #{idx}] Узлы: {path_nodes}, Время: {total_time} сек")
        df_route = build_optimal_route_dataframe(path_nodes, graph, work_days=work_days if work_info else 0, route_id=idx)
        df_route["Общее_время_в_сек"] = total_time
        df_route["Общее_время_формат"] = format_duration(total_time)
        all_frames.append(df_route)
    df_all = pd.concat(all_frames, ignore_index=True)
    df_all.sort_values(["Маршрут_ID", "Номер сегмента маршрута"], inplace=True)

    # Выводим результат и сохраняем
    print("\n=== Найденные маршруты (до 10) ===")
    print(df_all)
    df_all.to_csv("k_shortest_paths_result.csv", index=False, encoding="utf-8")
    logger.info("Результаты сохранены в k_shortest_paths_result.csv")

if __name__ == "__main__":
    main()