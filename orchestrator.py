#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Оркестратор пайплайна автоматизации истории.

Последовательно запускает три скрипта:
  1. Промпты  -> prompt_pipeline.py
  2. Визуал   -> history_fastgen_image_generator_RU_EN_20workers.py
  3. Монтаж   -> history_video_creator_ONE_CLICK_SAFE.py

Каждый скрипт запускается тем же интерпретатором Python, что и оркестратор,
в своей рабочей директории. Если какой-то этап завершается с ошибкой,
пайплайн останавливается и последующие скрипты не запускаются.
"""

import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Порядок запуска скриптов пайплайна.
SCRIPTS = [
    "/Users/aleksandrtomilov/Desktop/ИСТОРИЯ АВТОМАТИЗАЦИЯ/ПРОМПТЫ/prompt_pipeline.py",
    "/Users/aleksandrtomilov/Desktop/ИСТОРИЯ АВТОМАТИЗАЦИЯ/ВИЗУАЛ/history_fastgen_image_generator_RU_EN_20workers.py",
    "/Users/aleksandrtomilov/Desktop/ИСТОРИЯ АВТОМАТИЗАЦИЯ/МОНТАЖ/history_video_creator_ONE_CLICK_SAFE.py",
]


def log(message: str) -> None:
    """Печатает сообщение с отметкой времени."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def run_script(script_path: str, index: int, total: int) -> int:
    """
    Запускает один скрипт и возвращает его код завершения.

    Скрипт запускается тем же интерпретатором Python и с рабочей директорией,
    равной папке скрипта, чтобы относительные пути внутри работали корректно.
    """
    path = Path(script_path)

    log("=" * 70)
    log(f"ЭТАП {index}/{total}: {path.name}")
    log(f"Путь: {script_path}")
    log("=" * 70)

    if not path.is_file():
        log(f"ОШИБКА: файл не найден -> {script_path}")
        return 1

    start = time.monotonic()
    try:
        result = subprocess.run(
            [sys.executable, str(path)],
            cwd=str(path.parent),
        )
        code = result.returncode
    except KeyboardInterrupt:
        log("Прервано пользователем (Ctrl+C).")
        raise
    except Exception as exc:  # noqa: BLE001 - хотим показать любую ошибку запуска
        log(f"ОШИБКА запуска: {exc}")
        return 1

    elapsed = time.monotonic() - start
    if code == 0:
        log(f"ЭТАП {index}/{total} завершён успешно за {elapsed:.1f} c.")
    else:
        log(f"ЭТАП {index}/{total} завершился с кодом {code} за {elapsed:.1f} c.")
    return code


def main() -> int:
    total = len(SCRIPTS)
    log(f"Старт оркестратора. Скриптов в пайплайне: {total}")
    pipeline_start = time.monotonic()

    for index, script_path in enumerate(SCRIPTS, start=1):
        code = run_script(script_path, index, total)
        if code != 0:
            log("-" * 70)
            log(f"Пайплайн остановлен на этапе {index}/{total} (код {code}).")
            log("Последующие скрипты запущены не будут.")
            return code

    total_elapsed = time.monotonic() - pipeline_start
    log("-" * 70)
    log(f"Пайплайн полностью завершён за {total_elapsed:.1f} c. Все этапы выполнены.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
