from flask import Flask, request, jsonify
import logging
import os

# Если у вас все функции находятся в route_planner.py, импортируем их:
# (Важно: следите, чтобы в route_planner.py не было конфликта при import, 
# и чтобы функции были видны извне)
from route_planner import (
    get_route_options,
    combine_route_data,
    build_route_graph,
    k_shortest_paths,
    build_route_details_df,
    format_duration
)

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

@app.route("/")
def index():
    return "Сервис для поиска маршрутов. Используйте /api/route."

@app.route("/api/route", methods=["GET"])
def route_api():
    """
    Простой GET-эндпоинт:
    Пример запроса: /api/route?from=s9600370&to=s9600366&date=2025-06-01&k=3
    """
    from_code = request.args.get("from", "")
    to_code   = request.args.get("to", "")
    date_str  = request.args.get("date", "")
    k         = int(request.args.get("k", 3))  # по умолчанию 3 пути

    if not from_code or not to_code or not date_str:
        return jsonify({"error": "Необходимо указать параметры from, to, date"}), 400

    # 1) Получаем данные маршрутов из API
    route_data = get_route_options(from_code, to_code, date_str)
    if not route_data.get("segments"):
        return jsonify({"error": "Маршруты не найдены"}), 404

    # 2) Формируем граф
    G = build_route_graph(route_data, intermediate_node=None)

    # 3) Ищем K кратчайших путей
    routes = k_shortest_paths(G, from_code, to_code, k=k)
    if not routes:
        return jsonify({"error": "Пути не найдены"}), 404

    # 4) Формируем результат
    results = []
    for idx, (path_nodes, total_time) in enumerate(routes, start=1):
        # Собираем DataFrame с детализацией
        df_details = build_route_details_df(G, path_nodes, route_id=idx)
        route_dict = {
            "route_id": idx,
            "nodes": path_nodes,
            "total_time_sec": int(total_time),
            "total_time_str": format_duration(total_time),
            "segments_detail": df_details.to_dict(orient="records")
        }
        results.append(route_dict)

    return jsonify(results), 200


if __name__ == "__main__":
    # Запуск локально: python app.py
    # По умолчанию сервер будет доступен на http://127.0.0.1:5000
    app.run(debug=True)