from flask import Flask, request, jsonify
import os
import json
import logging
from datetime import datetime, timedelta

import pandas as pd
import networkx as nx
from networkx.algorithms.simple_paths import shortest_simple_paths

# Импортируем функции из route_planner.py (предположим, он лежит в том же каталоге)
from route_planner import (
    load_stations_dataframe,
    load_settlements_dataframe,
    select_transport_node,
    get_route_options,
    combine_route_data,
    build_route_graph,
    find_k_shortest_paths,
    build_optimal_route_dataframe,
    format_duration
)

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RailwayApp")

@app.route("/")
def index():
    return "Сервис для поиска маршрутов запущен. Обратитесь к /api/route для поиска."

@app.route("/api/route", methods=["GET"])
def api_route():
    """
    GET-параметры:
      - from (обязательный): Город/код отправления (пример: "Москва" или "s9600370")
      - to (обязательный): Город/код прибытия (пример: "Санкт-Петербург" или "s9600366")
      - date (обязательный): Дата поездки (формат YYYY-MM-DD)
      - work_city (необязательный): Город/код промежуточного узла
      - work_days (необязательный): Количество суток остановки (целое число 1..7)

    Пример запроса:
      /api/route?from=Москва&to=Санкт-Петербург&date=2025-03-17
    """
    from_param = request.args.get("from", "").strip()
    to_param = request.args.get("to", "").strip()
    date_param = request.args.get("date", "").strip()
    work_city = request.args.get("work_city", "").strip()
    work_days_str = request.args.get("work_days", "").strip()

    # Проверяем обязательные параметры
    if not from_param or not to_param or not date_param:
        return jsonify({"error": "Параметры 'from', 'to' и 'date' обязательны"}), 400
    
    # Проверяем дату
    try:
        datetime.strptime(date_param, "%Y-%m-%d")
    except ValueError:
        return jsonify({"error": "Неверный формат даты. Ожидается YYYY-MM-DD"}), 400

    # work_days — необязательный, но если задан, проверим число
    work_days = 0
    if work_days_str:
        try:
            work_days = int(work_days_str)
            if not (1 <= work_days <= 7):
                return jsonify({"error": "work_days должен быть от 1 до 7"}), 400
        except ValueError:
            return jsonify({"error": "work_days должен быть числом"}), 400

    # Загружаем справочные данные (CSV + JSON)
    df_stations, _ = load_stations_dataframe()
    df_settlements = load_settlements_dataframe()
    if df_stations.empty or df_settlements.empty:
        return jsonify({"error": "Не удалось загрузить справочные данные (stations_list.csv/json)"}), 500

    # Выбор узлов (отправление, прибытие, промежуточный)
    from_node = select_transport_node(from_param, df_stations, df_settlements)
    to_node = select_transport_node(to_param, df_stations, df_settlements)
    if not from_node or not to_node:
        return jsonify({"error": "Не удалось определить коды станций 'from' или 'to'"}), 400

    work_node = ""
    if work_city:
        work_node = select_transport_node(work_city, df_stations, df_settlements)

    # Запрашиваем маршруты
    if work_node:
        route_data_1 = get_route_options(from_node, work_node, date_param)
        if not route_data_1.get("segments"):
            return jsonify({"error": "Нет сегментов для первого участка"}), 404
        # Определяем дату второго участка (прибытие + work_days)
        first_seg = route_data_1["segments"][0]
        arrival_str = first_seg.get("arrival")
        if not arrival_str:
            return jsonify({"error": "Первый участок без arrival"}), 404
        arrival_dt = datetime.fromisoformat(arrival_str)
        date2 = (arrival_dt + timedelta(days=work_days)).strftime("%Y-%m-%d")
        route_data_2 = get_route_options(work_node, to_node, date2)
        if not route_data_2.get("segments"):
            return jsonify({"error": "Нет сегментов для второго участка"}), 404
        final_data = combine_route_data(route_data_1, route_data_2, from_node, to_node)
    else:
        final_data = get_route_options(from_node, to_node, date_param)
        if not final_data.get("segments"):
            return jsonify({"error": "Нет маршрутов для прямого участка"}), 404

    # Строим граф
    graph = build_route_graph(final_data, intermediate_node=work_node if work_node else None)
    if graph.number_of_nodes() == 0 or from_node not in graph.nodes or to_node not in graph.nodes:
        return jsonify({"error": "Граф пуст или не содержит нужные узлы"}), 404

    # Ищем до 10 кратчайших путей
    routes = find_k_shortest_paths(graph, from_node, to_node, k=10)
    if not routes:
        return jsonify({"error": "Пути не найдены"}), 404

    # Формируем результат
    results = []
    for idx, (path_nodes, total_time) in enumerate(routes, start=1):
        df_route = build_optimal_route_dataframe(path_nodes, graph, work_days=work_days, route_id=idx)
        df_route["Общее_время_в_сек"] = total_time
        df_route["Общее_время_формат"] = format_duration(total_time)
        route_dict = {
            "route_id": idx,
            "total_time_sec": total_time,
            "total_time_str": format_duration(total_time),
            "details": df_route.to_dict(orient="records")
        }
        results.append(route_dict)

    return jsonify(results), 200

if __name__ == "__main__":
    # Railway обычно подставляет PORT через переменную окружения
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)