#!/bin/bash
# Двойной клик по этому файлу запускает весь конвейер.
# Если macOS ругается «неизвестный разработчик» — правый клик -> Открыть.

cd "$(dirname "$0")" || exit 1

# Ищем рабочий python3: сначала системный, потом типичные места установки
PYTHON=""
for candidate in \
    "$(command -v python3)" \
    /usr/local/bin/python3 \
    /opt/homebrew/bin/python3 \
    /Library/Frameworks/Python.framework/Versions/Current/bin/python3
do
    if [ -x "$candidate" ]; then
        PYTHON="$candidate"
        break
    fi
done

if [ -z "$PYTHON" ]; then
    echo "Не найден Python 3. Установите его с https://www.python.org/downloads/"
    echo
    read -n 1 -s -r -p "Нажмите любую клавишу, чтобы закрыть окно..."
    exit 1
fi

"$PYTHON" "СТАРТ.py" "$@"
STATUS=$?

echo
if [ $STATUS -eq 0 ]; then
    echo "Работа завершена."
else
    echo "Скрипт завершился с ошибкой (код $STATUS). Сообщения смотрите выше."
fi
echo
read -n 1 -s -r -p "Нажмите любую клавишу, чтобы закрыть окно..."
echo
