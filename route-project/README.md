# Route Project

## Структура области без кода
- <a href="https://github.com/KsyLight/case-hack-projects/blob/main/route-project/UI-%D0%B4%D0%B8%D0%B7%D0%B0%D0%B9%D0%BD.png">UI-дизайн
- <a href="https://github.com/KsyLight/case-hack-projects/blob/main/route-project/CupIT2025-%D0%90%D0%94-%D0%9F%D0%B0%D0%BD%D0%B4%D1%83%D1%80%D0%B8_%D1%87%D0%BE%D0%BD%D0%B3%D1%83%D1%80%D0%B8_%D0%B8_%D0%B4%D1%83%D0%B4%D1%83%D0%BA.pdf">Презентация

## Предварительные действия - клонирование папки репозитория

**Важно иметь установленный Git. Если он отсутствует, необходимо  <a href="[https://github.com/KsyLight/case-hack-projects/blob/main/route-project/CupIT2025-%D0%90%D0%94-%D0%9F%D0%B0%D0%BD%D0%B4%D1%83%D1%80%D0%B8_%D1%87%D0%BE%D0%BD%D0%B3%D1%83%D1%80%D0%B8_%D0%B8_%D0%B4%D1%83%D0%B4%D1%83%D0%BA.pdf](https://git-scm.com/downloads)">Презентация.**
**Важно иметь установленный Git. Если он отсутствует, необходимо  <a href="https://git-scm.com/downloads">установить.**

Проект использует следующие библиотеки:  
- `requests` – для работы с API  
- `pandas` – для обработки данных  
- `networkx` – для построения графов маршрутов  
- `IPython` – для работы с Jupyter Notebook  

1. Открыть терминал или командную строку.
2. Перейти в папку, куда будет клонирован проект:

   ```sh
   cd path/to/your/folder
   ```
   *(Замените `path/to/your/folder` на ваш путь)*
4. Клонировать только папку `route-project` из репозитория:

   ```sh
   git clone --depth 1 --filter=blob:none --sparse https://github.com/KsyLight/case-hack-projects.git
   ```
   ```sh
   cd case-hack-projects
   ```
   ```sh
   git sparse-checkout set route-project
   ```
5. Перейти в папку проекта:

   ```sh
   cd route-project
   ```

## Основные действия

5. Создать виртуальное окружение (рекомендуется):

   ```sh
   python -m venv route_venv
   ```
7. Активировать виртуальное окружение:
   - Для Windows:

     ```sh
     route_venv\Scripts\activate
     ```
   - Для macOS/Linux:

     ```sh
     source route_venv/bin/activate
     ```
8. Установить зависимости:

   ```sh
   pip install -r requirements.txt
   ```
10. Далее есть два варианта: запустить скрипт или использовать ноутбук:
- Запуск скрипта:

  ```sh
   python route_planner.py
   ```
- Открытие ноутбука в Jupyter Notebook (можно использовать любой другой IDE, открыв через него)
   - Запускаем Jupyter Notebook:

     ```sh
     jupyter notebook notebook.ipynb
     ```
   - Запускаем выполненние ноутбука
10. Ввести запрашиваемые параметры и дождаться результата.

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

