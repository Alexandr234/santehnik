#!/bin/bash
# Двойной клик по этому файлу запускает дубляж на macOS.
# Скрипт сам создаст виртуальное окружение и поставит всё необходимое.

# Переходим в папку, где лежит этот файл.
cd "$(dirname "$0")" || exit 1

# Ищем подходящий Python.
if command -v python3 >/dev/null 2>&1; then
    PY=python3
elif command -v python >/dev/null 2>&1; then
    PY=python
else
    echo "[ОШИБКА] Python не найден."
    echo "Установи Python 3 с https://www.python.org/downloads/ и попробуй снова."
    read -r -p "Нажми Enter, чтобы закрыть..."
    exit 1
fi

"$PY" "srt_gpt_voicer_dubber_v2.py" "$@"
