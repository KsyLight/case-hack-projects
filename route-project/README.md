# Route Project

## Структура области без кода
- <a href="https://github.com/KsyLight/case-hack-projects/blob/main/route-project/UI-%D0%B4%D0%B8%D0%B7%D0%B0%D0%B9%D0%BD.png">UI-дизайн
- <a href="">Презентация

## Предварительные действия - клонирование папки репозитория

**Важно иметь установленный Git. Если он отсутствует, необходимо установить.**

Проект использует следующие библиотеки:  
- `requests` – для работы с API  
- `pandas` – для обработки данных  
- `networkx` – для построения графов маршрутов  
- `IPython` – для работы с Jupyter Notebook  

Можно испольлзовать просто `notebook.ipynb` без `.py` файла (то есть без скрипта), установив необходимые зависимости из пункта 7.

1. Открыть терминал или командную строку.
2. Перейти в папку, куда будет клонирован проект:
   ```sh
   cd path/to/your/folder
   ```
   *(Замените `path/to/your/folder` на ваш путь)*
3. Клонировать только папку `route-project` из репозитория:
   ```sh
   git clone --depth 1 --filter=blob:none --sparse https://github.com/KsyLight/case-hack-projects.git
   cd case-hack-projects
   git sparse-checkout set route-project
   ```
4. Перейти в папку проекта:
   ```sh
   cd route-project
   ```

## Основные действия

5. Создать виртуальное окружение (рекомендуется):
   ```sh
   python -m venv route_venv
   ```
6. Активировать виртуальное окружение:
   - Для Windows:
     ```sh
     route_venv\Scripts\activate
     ```
   - Для macOS/Linux:
     ```sh
     source route_venv/bin/activate
     ```
7. Установить зависимости:
   ```sh
   pip install -r requirements.txt
   ```
8. Запустить скрипт:
   ```sh
   python route_planner.py
   ```
9. Ввести запрашиваемые параметры и дождаться результата.

## Дополнительная информация

- Если возникли проблемы с зависимостями, попробуйте обновить `pip`:
  ```sh
  python -m pip install --upgrade pip
  ```
- Для выхода из виртуального окружения выполните команду:
  ```sh
  deactivate
  ```

<div align="center">
    <img src="cat.jpg" alt="Описание изображения" width="1000" height="auto">
</div>

