# Route Project

## Предварительные действия - клонирование папки репозитория

**Важно иметь установленный Git. Если он отсутствует, необходимо установить.**

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

![Описание изображения](cat.jpg)
