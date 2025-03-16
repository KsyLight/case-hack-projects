import os
import json
import logging
import requests
import pandas as pd
import networkx as nx
from networkx.algorithms.simple_paths import shortest_simple_paths
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple, Union
from time import sleep
from IPython.display import display

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("RouteSystem")

STATIONS_CACHE_JSON = "stations_list.json"
STATIONS_CACHE_CSV = "stations_list.csv"
YANDEX_RASP_API_KEY = "2a22163d-023f-47db-a974-6d683c413a83"
AVIASALES_TOKEN = "dbd789b8fa1dadd43c5376abdcba15a0"
AVIASALES_API_URL = "https://api.travelpayouts.com/v2/prices/latest"
URL_STATIONS_LIST = "https://api.rasp.yandex.net/v3.0/stations_list"
URL_SEARCH = "https://api.rasp.yandex.net/v3.0/search/"
SELECTED_NODES_FILE = "selected_nodes.json"
MIN_TRANSFER_TIME = 1800
YANDEX_TO_IATA_MAP = {
    "s9600370": "SVO",
    "s9600363": "VKO",
}

def get_aviasales_price(origin_iata: str, destination_iata: str, departure_date: str) -> float:
    try:
        params = {
            "origin": origin_iata,
            "destination": destination_iata,
            "departure_at": departure_date,
            "one_way": "true",
            "currency": "RUB",
            "token": AVIASALES_TOKEN,
            "limit": 1,
            "sorting": "price"
        }
        logger.info(f"Aviasales: запрос для {origin_iata} -> {destination_iata} на {departure_date}")
        resp = requests.get(AVIASALES_API_URL, params=params)
        resp.raise_for_status()
        data = resp.json()
        tickets = data.get("data", [])
        if not tickets:
            return None
        first_ticket = tickets[0]
        price_value = first_ticket.get("price")
        return float(price_value) if price_value else None
    except Exception as e:
        logger.error(f"Aviasales: ошибка запроса цены: {e}")
        return None

def fetch_stations_list_from_api(api_key: str) -> dict:
    logger.info("Запрос списка станций из API Яндекс.Расписаний...")
    params = {"apikey": api_key, "lang": "ru_RU", "format": "json"}
    response = requests.get(URL_STATIONS_LIST, params=params)
    response.raise_for_status()
    data = response.json()
    logger.info("Список станций получен.")
    return data

def stations_to_dataframe(stations_json: dict) -> pd.DataFrame:
    stations_list = []
    for country in stations_json.get("countries", []):
        for region in country.get("regions", []):
            for settlement in region.get("settlements", []):
                for station in settlement.get("stations", []):
                    station["settlement_title"] = settlement.get("title", "")
                    station["region_title"] = region.get("title", "")
                    station["country_title"] = country.get("title", "")
                    if "type" not in station:
                        station["type"] = "station"
                    stations_list.append(station)
    df = pd.json_normalize(stations_list)
    df = df[df["country_title"].str.contains("Россия", case=False, na=False)]
    return df

def load_stations_dataframe() -> (pd.DataFrame, dict):
    if os.path.exists(STATIONS_CACHE_CSV) and os.path.exists(STATIONS_CACHE_JSON):
        logger.info(f"Найден CSV-кэш: {STATIONS_CACHE_CSV}")
        df = pd.read_csv(STATIONS_CACHE_CSV, encoding="utf-8", low_memory=False)
        if "settlement_title" not in df.columns:
            logger.info("CSV-кэш не содержит 'settlement_title', пересоздаем DataFrame")
            with open(STATIONS_CACHE_JSON, "r", encoding="utf-8") as f:
                stations_json = json.load(f)
            df = stations_to_dataframe(stations_json)
            df.to_csv(STATIONS_CACHE_CSV, index=False, encoding="utf-8")
        else:
            with open(STATIONS_CACHE_JSON, "r", encoding="utf-8") as f:
                stations_json = json.load(f)
    else:
        logger.info("Кэш не найден, запрашиваем данные через API...")
        stations_json = fetch_stations_list_from_api(YANDEX_RASP_API_KEY)
        with open(STATIONS_CACHE_JSON, "w", encoding="utf-8") as f:
            json.dump(stations_json, f, ensure_ascii=False, indent=2)
        df = stations_to_dataframe(stations_json)
        df.to_csv(STATIONS_CACHE_CSV, index=False, encoding="utf-8")
    return df, stations_json

def load_settlements_dataframe() -> pd.DataFrame:
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
    df = pd.json_normalize(settlements_list)
    df = df[df["country_title"].str.contains("Россия", case=False, na=False)]
    return df

def filter_airports_and_train_stations(df: pd.DataFrame, city: str) -> pd.DataFrame:
    if "station_type" not in df.columns:
        df["station_type"] = ""
    mask = (df["settlement_title"].str.contains(city, case=False, na=False)) | \
           (df["title"].str.contains(city, case=False, na=False))
    df_city = df[mask]
    if "type" in df_city.columns:
        df_city = df_city[df_city["type"].str.lower() != "settlement"]
    valid_types = ["airport", "train_station"]
    df_filtered = df_city[df_city["station_type"].str.lower().isin(valid_types)]
    df_filtered = df_filtered[~df_filtered["title"].str.contains("Тур", case=False, na=False)]
    return df_filtered

def choose_station_interactive(df: pd.DataFrame, city: str) -> str:
    df_filtered = filter_airports_and_train_stations(df, city)
    if df_filtered.empty:
        logger.warning(f"В городе '{city}' не найдено транспортных узлов.")
        return ""
    if len(df_filtered) == 1:
        chosen = df_filtered.iloc[0]
        code = chosen.get("codes.yandex_code") or chosen.get("code", "")
        logger.info(f"Единственный вариант для '{city}': {code}")
        return code
    print(f"\nВ городе '{city}' найдено несколько вариантов:")
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
            logger.info(f"Выбранный узел для '{city}': {code}")
            return code
        else:
            logger.error("Неверный выбор: номер вне диапазона.")
            return ""
    except Exception as e:
        logger.error(f"Ошибка ввода: {e}")
        return ""

def get_all_city_codes_automatic(df_settlements: pd.DataFrame, city: str, df_stations: pd.DataFrame) -> List[str]:
    if "codes.yandex_code" in df_stations.columns:
        col = "codes.yandex_code"
    elif "code" in df_stations.columns:
        col = "code"
    else:
        logger.error("Столбец с кодом не найден.")
        return []
    df_candidates = df_stations[
        (df_stations["settlement_title"].str.contains(city, case=False, na=False)) &
        (df_stations[col].str.startswith("s"))
    ]
    if not df_candidates.empty:
        codes = df_candidates[col].tolist()
        logger.info(f"Найдено {len(codes)} кодов для '{city}': {codes}")
        return codes
    else:
        logger.warning(f"Для '{city}' нет записей с кодом, начинающимся на 's'.")
        return []

def update_node_code(node: str, city: str, df: pd.DataFrame) -> str:
    if not node.startswith("c"):
        return node
    if "settlement_title" not in df.columns:
        logger.error("Нет столбца 'settlement_title'.")
        return node
    if "codes.yandex_code" in df.columns:
        col = "codes.yandex_code"
    elif "code" in df.columns:
        col = "code"
    else:
        logger.error("Столбец с кодом не найден.")
        return node
    df_candidate = df[
        (df["settlement_title"].str.contains(city, case=False, na=False)) &
        (df[col].str.startswith("s"))
    ]
    if not df_candidate.empty:
        candidate_station = df_candidate.iloc[0][col]
        logger.info(f"Обновляем {node} -> {candidate_station} для '{city}'")
        return candidate_station
    else:
        logger.warning(f"Подходящий код для '{city}' не найден, остаётся {node}")
        return node

def select_transport_node(city: str, df_stations: pd.DataFrame, df_settlements: pd.DataFrame) -> Union[str, List[str]]:
    if city.lower() in ["москва", "санкт-петербург"]:
        return choose_station_interactive(df_stations, city)
    else:
        codes = get_all_city_codes_automatic(df_settlements, city, df_stations)
        return [update_node_code(code, city, df_stations) for code in codes]

def input_route_parameters() -> dict:
    city_from = input("Введите город/код отправления: ").strip()
    city_to = input("Введите город/код прибытия: ").strip()
    date_str = input("Введите дату (YYYY-MM-DD): ").strip()
    try:
        dt_obj = datetime.strptime(date_str, "%Y-%m-%d")
        if dt_obj.date() < datetime.now().date():
            logger.error("Дата в прошлом.")
            return {}
    except ValueError:
        logger.error("Неверный формат даты. Ожидается YYYY-MM-DD.")
        return {}
    choice = input("Хотите остановку в промежуточном городе? (Да/Нет): ").strip().lower()
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
    return {"city_from": city_from, "city_to": city_to, "date": date_str, "work_info": work_info}

def get_route_options(from_code: str, to_code: str, date: str,
                      transport_types: str = "plane,train,suburban,bus,water,helicopter",
                      transfers: str = "true", limit: int = 100, offset: int = 0) -> dict:
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
    logger.info(f"Yandex: запрос {from_code} -> {to_code} на {date}")
    r = requests.get(URL_SEARCH, params=params)
    try:
        r.raise_for_status()
    except requests.exceptions.HTTPError as e:
        if r.status_code == 404:
            logger.warning(f"Yandex: маршруты не найдены для {params}")
            return {}
        else:
            logger.error(f"Yandex: HTTP ошибка: {e} (Статус: {r.status_code})")
            raise
    logger.info("Yandex: маршруты получены.")
    return r.json()

def save_route_options(route_data: dict, filename: str) -> None:
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(route_data, f, ensure_ascii=False, indent=2)
    logger.info(f"Сохранён JSON: {filename}")
    if "segments" in route_data:
        df = pd.json_normalize(route_data["segments"])
        csv_name = filename.replace(".json", ".csv")
        df.to_csv(csv_name, index=False, encoding="utf-8")
        logger.info(f"Сохранён CSV: {csv_name}")

def iata_to_yandex(iata: str) -> str:
    for yandex_code, code in YANDEX_TO_IATA_MAP.items():
        if code == iata:
            return yandex_code
    return iata

def yandex_to_iata(yandex_code: str) -> str:
    return YANDEX_TO_IATA_MAP.get(yandex_code, yandex_code)

def get_aviasales_routes(origin_iata: str, destination_iata: str, departure_date: str) -> dict:
    try:
        params = {
            "origin": origin_iata,
            "destination": destination_iata,
            "departure_at": departure_date,
            "one_way": "true",
            "currency": "RUB",
            "token": AVIASALES_TOKEN,
            "limit": 10,
            "sorting": "price"
        }
        logger.info(f"Aviasales: запрос {origin_iata} -> {destination_iata} на {departure_date}")
        resp = requests.get(AVIASALES_API_URL, params=params)
        resp.raise_for_status()
        data = resp.json()
        tickets = data.get("data", [])
        segments = []
        for ticket in tickets:
            dep_str = ticket.get("departure_at")
            if not dep_str:
                continue
            try:
                dep_dt = datetime.fromisoformat(dep_str)
            except Exception:
                continue
            duration_minutes = ticket.get("duration")
            if duration_minutes:
                arr_dt = dep_dt + timedelta(minutes=duration_minutes)
                arr_str = arr_dt.isoformat()
            else:
                arr_str = ""
            segment = {
                "from": {"code": iata_to_yandex(ticket.get("origin", "")), "title": ticket.get("origin", "")},
                "to": {"code": iata_to_yandex(ticket.get("destination", "")), "title": ticket.get("destination", "")},
                "departure": dep_str,
                "arrival": arr_str,
                "transport_type": "plane",
                "thread": {"transport_type": "plane", "carrier": ticket.get("airline", "")},
                "cost": float(ticket.get("price", 0)),
                "query": 3
            }
            segments.append(segment)
        logger.info(f"Aviasales: получено сегментов: {len(segments)}")
        return {"segments": segments, "search": {"from": {"code": origin_iata}, "to": {"code": destination_iata}}}
    except Exception as e:
        logger.error(f"Aviasales: ошибка запроса маршрутов: {e}")
        return {}

def save_aviasales_route_options(route_data: dict, filename: str) -> None:
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(route_data, f, ensure_ascii=False, indent=2)
    logger.info(f"Aviasales: сохранён JSON: {filename}")
    if "segments" in route_data:
        df = pd.json_normalize(route_data["segments"])
        csv_name = filename.replace(".json", ".csv")
        df.to_csv(csv_name, index=False, encoding="utf-8")
        logger.info(f"Aviasales: сохранён CSV: {csv_name}")

def convert_multidigraph_to_digraph(G: nx.MultiDiGraph) -> nx.DiGraph:
    H = nx.DiGraph()
    for u, v, data in G.edges(data=True):
        w = data.get("weight", 0)
        if H.has_edge(u, v):
            if w < H[u][v]["weight"]:
                H[u][v]["weight"] = w
                H[u][v]["time"] = data.get("time")
                H[u][v]["cost"] = data.get("cost")
                H[u][v]["segment"] = data.get("segment")
        else:
            H.add_edge(u, v, weight=w, time=data.get("time"), cost=data.get("cost"), segment=data.get("segment"))
    return H

def combine_route_data(route_datas: List[dict], from_nodes: List[str], to_nodes: List[str]) -> dict:
    combined_segments = []
    for rd in route_datas:
        segs = rd.get("segments", [])
        for s in segs:
            if "query" not in s:
                s["query"] = 1
            combined_segments.append(s)
    logger.info(f"Объединено сегментов: {len(combined_segments)}")
    return {"segments": combined_segments, "search": {"from": {"codes": from_nodes}, "to": {"codes": to_nodes}}}

def build_route_graph(route_data: dict, intermediate_node: Union[str, None] = None, ranking_param: str = "time") -> nx.MultiDiGraph:
    G = nx.MultiDiGraph()
    segments = route_data.get("segments", [])
    interval_segments = route_data.get("interval_segments", [])
    all_segments = segments + interval_segments
    logger.info(f"Построение графа: найдено {len(all_segments)} сегментов")
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
            from_st = s.get("from", {})
            to_st = s.get("to", {})
            from_code = from_st.get("code")
            to_code = to_st.get("code")
            if not from_code or not to_code:
                continue
            dep_str = s.get("departure")
            arr_str = s.get("arrival")
            if not dep_str or not arr_str:
                continue
            try:
                dep_dt = datetime.fromisoformat(dep_str)
                arr_dt = datetime.fromisoformat(arr_str)
                duration = (arr_dt - dep_dt).total_seconds()
                if duration < 0:
                    continue
            except Exception:
                continue
            if intermediate_node and s.get("query") == 2 and from_code == intermediate_node and min_arrival:
                gap = (dep_dt - min_arrival).total_seconds()
                if gap < MIN_TRANSFER_TIME:
                    logger.info(f"Сегмент исключён (пересадка {gap} сек)")
                    continue
            cost_value = None
            transport_type = s.get("thread", {}).get("transport_type", "") or s.get("transport_type", "")
            if transport_type == "plane":
                iata_origin = YANDEX_TO_IATA_MAP.get(from_code)
                iata_destination = YANDEX_TO_IATA_MAP.get(to_code)
                if iata_origin and iata_destination:
                    dep_date_str = dep_dt.strftime("%Y-%m-%d")
                    cost_value = get_aviasales_price(iata_origin, iata_destination, dep_date_str)
            weight = duration if ranking_param == "time" else (float(cost_value) if cost_value is not None else 999999.0)
            G.add_edge(from_code, to_code, weight=weight, time=duration, cost=cost_value, segment=s)
        except Exception as e:
            logger.error(f"Ошибка обработки сегмента: {e}")
    logger.info(f"Граф построен: узлов={G.number_of_nodes()}, рёбер={G.number_of_edges()}")
    return G

def find_k_shortest_paths(G: nx.MultiDiGraph, source: str, target: str, k: int = 10) -> List[Tuple[List[str], float]]:
    H = convert_multidigraph_to_digraph(G)
    if source not in H or target not in H:
        logger.warning(f"Узел {source} или {target} отсутствует в графе.")
        return []
    paths_gen = shortest_simple_paths(H, source, target, weight="weight")
    results = []
    for path in paths_gen:
        total_weight = sum(H[u][v]["weight"] for u, v in zip(path, path[1:]))
        results.append((path, total_weight))
        if len(results) >= k:
            break
    return results

def format_duration(seconds: float) -> str:
    if seconds is None:
        return ""
    hh = int(seconds // 3600)
    mm = int((seconds % 3600) // 60)
    ss = int(seconds % 60)
    return f"{hh}ч {mm}м {ss}с"

def build_optimal_route_dataframe(route: List[str], graph: nx.MultiDiGraph, work_days: int = 0, route_id: int = 1) -> pd.DataFrame:
    rows = []
    for i in range(len(route) - 1):
        from_node = route[i]
        to_node = route[i + 1]
        edge_data = list(graph.get_edge_data(from_node, to_node).values())[0] if graph.has_edge(from_node, to_node) else {}
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
        transport_type = seg.get("thread", {}).get("transport_type", "") or seg.get("transport_type", "")
        rows.append({
            "Маршрут_ID": route_id,
            "Номер сегмента": i + 1,
            "Код отправления": from_code,
            "Город отправления": from_title,
            "Время отправления": dep_time,
            "Код прибытия": to_code,
            "Город прибытия": to_title,
            "Время прибытия": arr_time,
            "Остановок (суток)": work_days,
            "Время в пути": dur_str,
            "Вид транспорта": transport_type
        })
    return pd.DataFrame(rows)

def choose_valid_route_nodes(from_nodes: Union[str, List[str]], to_nodes: Union[str, List[str]], graph: nx.MultiDiGraph) -> Tuple[Union[str, None], Union[str, None]]:
    if isinstance(from_nodes, str):
        from_nodes = [from_nodes]
    if isinstance(to_nodes, str):
        to_nodes = [to_nodes]
    for f in from_nodes:
        for t in to_nodes:
            if f in graph.nodes and t in graph.nodes:
                return f, t
    return None, None

def main():
    params = input_route_parameters()
    if not params:
        return
    city_from = params["city_from"]
    city_to = params["city_to"]
    date_str = params["date"]
    work_info = params.get("work_info")
    ranking_param = input("По какому параметру искать маршрут? (time/cost): ").strip().lower()
    if ranking_param not in ("time", "cost"):
        ranking_param = "time"
    df_stations, _ = load_stations_dataframe()
    df_settlements = load_settlements_dataframe()
    if df_stations.empty or df_settlements.empty:
        print("Нет справочных данных для поиска.")
        return
    if city_from.lower() in ["москва", "санкт-петербург"]:
        from_nodes = [select_transport_node(city_from, df_stations, df_settlements)]
    else:
        from_nodes = select_transport_node(city_from, df_stations, df_settlements)
    if city_to.lower() in ["москва", "санкт-петербург"]:
        to_nodes = [select_transport_node(city_to, df_stations, df_settlements)]
    else:
        to_nodes = select_transport_node(city_to, df_stations, df_settlements)
    if not from_nodes or not to_nodes:
        print("Не удалось выбрать транспортные узлы для одного из городов.")
        return
    work_nodes = []
    work_days = 0
    if work_info:
        work_city, work_days = work_info
        if work_city.lower() in ["москва", "санкт-петербург"]:
            work_nodes = [select_transport_node(work_city, df_stations, df_settlements)]
        else:
            work_nodes = select_transport_node(work_city, df_stations, df_settlements)
    selected_nodes = {"from_nodes": from_nodes, "to_nodes": to_nodes}
    if work_nodes:
        selected_nodes["work_nodes"] = work_nodes
    with open(SELECTED_NODES_FILE, "w", encoding="utf-8") as f:
        json.dump(selected_nodes, f, ensure_ascii=False, indent=2)
    print("\nПоиск маршрутов выполняется по Яндекс.Расписаниям...")
    route_datas = []
    if work_nodes:
        seg1_datas = []
        for f in from_nodes:
            for w in work_nodes:
                rd1 = get_route_options(f, w, date_str)
                if rd1.get("segments"):
                    seg1_datas.append(rd1)
        if not seg1_datas:
            print("Маршруты по Яндекс.Расписаниям не найдены.")
            return
        first_seg = seg1_datas[0]["segments"][0]
        arrival_str = first_seg.get("arrival")
        if not arrival_str:
            print("Маршруты по Яндекс.Расписаниям не найдены (нет времени прибытия).")
            return
        arrival_dt = datetime.fromisoformat(arrival_str)
        new_date_str = (arrival_dt + timedelta(days=work_days)).strftime("%Y-%m-%d")
        seg2_datas = []
        for w in work_nodes:
            for t in to_nodes:
                rd2 = get_route_options(w, t, new_date_str)
                if rd2.get("segments"):
                    seg2_datas.append(rd2)
        if not seg2_datas:
            print("Маршруты по Яндекс.Расписаниям не найдены для второго сегмента.")
            return
        route_datas = seg1_datas + seg2_datas
    else:
        for f in from_nodes:
            for t in to_nodes:
                rd = get_route_options(f, t, date_str)
                if rd.get("segments"):
                    route_datas.append(rd)
        if not route_datas:
            print("Маршруты по Яндекс.Расписаниям не найдены.")
            return
    print("Маршруты по Яндекс.Расписаниям найдены.")
    fname_yandex = f"search_{city_from}_{city_to}_{date_str}.json".replace(" ", "_")
    combined_yandex = combine_route_data(route_datas, from_nodes, to_nodes)
    save_route_options(combined_yandex, fname_yandex)
    print("\nВыполняется поиск маршрутов по Aviasales...")
    aviasales_datas = []
    for f in from_nodes:
        origin_iata = yandex_to_iata(f)
        for t in to_nodes:
            destination_iata = yandex_to_iata(t)
            data = get_aviasales_routes(origin_iata, destination_iata, date_str)
            if data.get("segments"):
                aviasales_datas.append(data)
    if aviasales_datas:
        print("Маршруты по Aviasales найдены.")
        fname_av = f"aviasales_{city_from}_{city_to}_{date_str}.json".replace(" ", "_")
        combined_av = combine_route_data(aviasales_datas, from_nodes, to_nodes)
        save_aviasales_route_options(combined_av, fname_av)
    else:
        print("Маршруты по Aviasales не найдены.")
    combined_segments = combined_yandex.get("segments", [])
    if aviasales_datas:
        combined_segments += combine_route_data(aviasales_datas, from_nodes, to_nodes).get("segments", [])
    combined_data = {"segments": combined_segments, "search": combined_yandex.get("search", {})}
    fname_combined = f"combined_{city_from}_{city_to}_{date_str}.json".replace(" ", "_")
    save_route_options(combined_data, fname_combined)
    graph = build_route_graph(combined_data, intermediate_node=(work_nodes[0] if work_nodes else None), ranking_param=ranking_param)
    if graph.number_of_nodes() == 0:
        print("Маршруты не найдены (граф пуст).")
        return
    start_node, end_node = choose_valid_route_nodes(from_nodes, to_nodes, graph)
    if start_node is None or end_node is None:
        print("Маршруты не найдены (нет валидных узлов в графе).")
        return
    routes = find_k_shortest_paths(graph, start_node, end_node, k=10)
    if not routes:
        print("Маршруты не найдены (нет путей в графе).")
        return
    all_frames = []
    for idx, (path_nodes, total_val) in enumerate(routes, start=1):
        df_route = build_optimal_route_dataframe(
            path_nodes,
            graph,
            work_days=(work_days if work_info else 0),
            route_id=idx
        )
        if ranking_param == "cost":
            df_route["Общая_стоимость"] = total_val
        all_frames.append(df_route)
    df_all = pd.concat(all_frames, ignore_index=True)
    df_all.sort_values(["Маршрут_ID", "Номер сегмента"], inplace=True)
    print("\n=== Найденные маршруты (до 10) ===")
    display(df_all)
    output_filename = "k_shortest_paths_result_time.csv" if ranking_param == "time" else "k_shortest_paths_result.csv"
    df_all.to_csv(output_filename, index=False, encoding="utf-8")

if __name__ == "__main__":
    main()