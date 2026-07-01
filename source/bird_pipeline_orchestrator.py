#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Оркестратор пайплайна «ПТИЦЫ РЕАЛИСТИЧНЫЕ».

Запускает по очереди три скрипта:
    1. Генерация промптов   (ПРОМПТЫ)
    2. Генерация визуала     (ВИЗУАЛ)
    3. Монтаж видео          (МОНТАЖ)

Каждый следующий скрипт стартует только после успешного завершения
предыдущего. Если какой-то шаг падает — пайплайн останавливается,
и остальные шаги не запускаются (это поведение можно изменить
флагом --keep-going).

Запуск:
    python3 bird_pipeline_orchestrator.py
    python3 bird_pipeline_orchestrator.py --keep-going   # не останавливаться на ошибке
    python3 bird_pipeline_orchestrator.py --python /path/to/python
"""

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# --- Шаги пайплайна: (человеко-понятное имя, путь к скрипту) ---------------
STEPS = [
    (
        "1/3 · ПРОМПТЫ",
        "/Users/aleksandrtomilov/Desktop/ПТИЦЫ РЕАЛИСТИЧНЫЕ/ПРОМПТЫ/"
        "bird_prompt_pipeline_realistic_ru_de_es_pl.py",
    ),
    (
        "2/3 · ВИЗУАЛ",
        "/Users/aleksandrtomilov/Desktop/ПТИЦЫ РЕАЛИСТИЧНЫЕ/ВИЗУАЛ/"
        "flower_veo31_visual_batch_generator_realistic.py",
    ),
    (
        "3/3 · МОНТАЖ",
        "/Users/aleksandrtomilov/Desktop/ПТИЦЫ РЕАЛИСТИЧНЫЕ/МОНТАЖ/"
        "video_creator_times_autovenv_no_subs_realistic.py",
    ),
]


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def fmt_duration(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}ч {m}м {s}с"
    if m:
        return f"{m}м {s}с"
    return f"{s}с"


def run_step(name: str, script: str, python_exe: str) -> int:
    path = Path(script)
    log(f"▶️  ШАГ {name}")
    log(f"    Скрипт: {script}")

    if not path.exists():
        log(f"❌ Файл не найден: {script}")
        return 127

    started = time.time()
    # Запускаем скрипт из его собственной папки, чтобы относительные
    # пути внутри скрипта работали корректно.
    result = subprocess.run(
        [python_exe, str(path)],
        cwd=str(path.parent),
    )
    elapsed = time.time() - started

    if result.returncode == 0:
        log(f"✅ ШАГ {name} завершён за {fmt_duration(elapsed)}")
    else:
        log(
            f"❌ ШАГ {name} упал (код возврата {result.returncode}) "
            f"после {fmt_duration(elapsed)}"
        )
    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Оркестратор пайплайна «ПТИЦЫ РЕАЛИСТИЧНЫЕ»."
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Интерпретатор Python для запуска скриптов (по умолчанию — текущий).",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Не останавливать пайплайн при падении шага, а идти дальше.",
    )
    args = parser.parse_args()

    log("🐦 Старт пайплайна «ПТИЦЫ РЕАЛИСТИЧНЫЕ»")
    log(f"    Python: {args.python}")
    log(f"    Шагов: {len(STEPS)}")
    print("-" * 60, flush=True)

    pipeline_started = time.time()
    failed = []

    for name, script in STEPS:
        code = run_step(name, script, args.python)
        print("-" * 60, flush=True)
        if code != 0:
            failed.append(name)
            if not args.keep_going:
                log("🛑 Пайплайн остановлен из-за ошибки. "
                    "Оставшиеся шаги пропущены.")
                break

    total = fmt_duration(time.time() - pipeline_started)
    if failed:
        log(f"⚠️  Пайплайн завершён с ошибками за {total}. "
            f"Проблемные шаги: {', '.join(failed)}")
        return 1

    log(f"🎉 Пайплайн полностью завершён за {total}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
