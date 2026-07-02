#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Оркестратор пайплайна «ПСИХОЛОГИЯ ГЕРМАНИЯ ПОЛЬША».

Запускает пять скриптов строго друг за другом и показывает ВЕСЬ их вывод
в реальном времени (stdout и stderr — построчно, как будто запустил вручную):

  1. СКРИПТ ОЗВУЧКИ/voicer_batch_tts_RU_PL_DE.py      — озвучка RU/PL/DE
  2. ПРОМПТЫ/psych_prompt_pipeline_v2карлюнг.py        — генерация промптов
  3. ПРОМПТЫ/КАРЛ ЮНГ.py                                — обработка промптов
  4. ВИЗУАЛ/flower_image_generator2юнг.py               — генерация картинок
  5. МОНТАЖ/video_creator_zoom15.py                     — сборка видео

По умолчанию: если какой-то скрипт падает (ненулевой код возврата),
пайплайн останавливается и остальные шаги НЕ запускаются. Это можно
изменить флагом --keep-going.

Запуск:
   cd "/Users/aleksandrtomilov/Desktop/ПСИХОЛОГИЯ ГЕРМАНИЯ ПОЛЬША"
   python3 run_pipeline.py

Полезные флаги:
   python3 run_pipeline.py --keep-going     # не останавливаться на ошибке
   python3 run_pipeline.py --from 3         # начать с 3-го шага
   python3 run_pipeline.py --only 2 4       # запустить только шаги 2 и 4
   python3 run_pipeline.py --list           # показать список шагов и выйти
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Корневая папка проекта. Все пути к скриптам строятся от неё.
BASE_DIR = Path("/Users/aleksandrtomilov/Desktop/ПСИХОЛОГИЯ ГЕРМАНИЯ ПОЛЬША")

# Шаги пайплайна в порядке выполнения: (человекочитаемое имя, путь к скрипту).
STEPS = [
    ("Озвучка RU/PL/DE", BASE_DIR / "СКРИПТ ОЗВУЧКИ" / "voicer_batch_tts_RU_PL_DE.py"),
    ("Генерация промптов", BASE_DIR / "ПРОМПТЫ" / "psych_prompt_pipeline_v2карлюнг.py"),
    ("Карл Юнг", BASE_DIR / "ПРОМПТЫ" / "КАРЛ ЮНГ.py"),
    ("Генерация картинок", BASE_DIR / "ВИЗУАЛ" / "flower_image_generator2юнг.py"),
    ("Сборка видео", BASE_DIR / "МОНТАЖ" / "video_creator_zoom15.py"),
]


def log(message: str = "") -> None:
    print(message, flush=True)


def stamp() -> str:
    return datetime.now().strftime("%H:%M:%S")


def fmt_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}ч {m}м {s}с"
    if m:
        return f"{m}м {s}с"
    return f"{s}с"


def run_step(index: int, total: int, name: str, script: Path) -> int:
    """
    Запускает один скрипт и стримит его вывод построчно.
    Возвращает код возврата процесса (0 = успех).
    """
    log("")
    log("=" * 72)
    log(f"[ШАГ {index}/{total}] {name}")
    log(f"[ФАЙЛ] {script}")
    log(f"[СТАРТ] {stamp()}")
    log("=" * 72)

    if not script.exists():
        log(f"[ОШИБКА] Скрипт не найден: {script}")
        return 127

    started = time.time()

    # -u = небуферизованный вывод у дочернего Python, чтобы print'ы шли сразу.
    # stderr сливаем в stdout, чтобы видеть всё в одном потоке по порядку.
    try:
        process = subprocess.Popen(
            [sys.executable, "-u", str(script)],
            cwd=str(script.parent),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
    except Exception as exc:
        log(f"[ОШИБКА] Не удалось запустить {script}: {exc}")
        return 1

    try:
        assert process.stdout is not None
        for line in process.stdout:
            # Вывод скрипта как есть, без лишних префиксов.
            print(line, end="", flush=True)
        process.wait()
    except KeyboardInterrupt:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
        raise

    duration = time.time() - started
    code = process.returncode

    log("")
    if code == 0:
        log(f"[ОК] {name} — успешно за {fmt_duration(duration)} (завершено в {stamp()})")
    else:
        log(f"[ПРОВАЛ] {name} — код возврата {code}, время {fmt_duration(duration)} (в {stamp()})")

    return code


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Оркестратор пайплайна: запускает скрипты друг за другом и стримит их вывод.",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Не останавливаться на ошибке — выполнить все шаги, ошибки собрать в конце.",
    )
    parser.add_argument(
        "--from",
        dest="from_step",
        type=int,
        default=1,
        metavar="N",
        help="Начать с шага N (1..%d)." % len(STEPS),
    )
    parser.add_argument(
        "--only",
        type=int,
        nargs="+",
        metavar="N",
        help="Запустить только указанные шаги (например: --only 2 4).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Показать список шагов и выйти.",
    )
    return parser.parse_args()


def select_steps(args: argparse.Namespace):
    """Возвращает список кортежей (номер, имя, путь) с учётом --from / --only."""
    numbered = [(i + 1, name, script) for i, (name, script) in enumerate(STEPS)]

    if args.only:
        chosen = set(args.only)
        return [item for item in numbered if item[0] in chosen]

    return [item for item in numbered if item[0] >= args.from_step]


def main() -> None:
    args = parse_args()

    if args.list:
        log("Шаги пайплайна:")
        for i, (name, script) in enumerate(STEPS, start=1):
            mark = "✓" if script.exists() else "✗ (нет файла)"
            log(f"  {i}. {name}  {mark}")
            log(f"     {script}")
        return

    steps = select_steps(args)
    if not steps:
        log("[ВНИМАНИЕ] Нет шагов для запуска (проверь --from / --only).")
        sys.exit(1)

    total = len(STEPS)
    pipeline_started = time.time()

    log("#" * 72)
    log("ОРКЕСТРАТОР ПАЙПЛАЙНА")
    log(f"Старт: {stamp()}")
    log(f"К запуску: {', '.join(str(n) for n, _, _ in steps)} из {total}")
    log(f"Режим ошибок: {'продолжать' if args.keep_going else 'останавливаться на первой'}")
    log("#" * 72)

    failures = []

    for number, name, script in steps:
        code = run_step(number, total, name, script)
        if code != 0:
            failures.append((number, name, code))
            if not args.keep_going:
                log("")
                log(f"[СТОП] Шаг {number} «{name}» упал. Останавливаю пайплайн (--keep-going чтобы продолжать).")
                break

    total_time = time.time() - pipeline_started

    log("")
    log("#" * 72)
    log(f"ИТОГ. Общее время: {fmt_duration(total_time)}. Финиш: {stamp()}")
    if failures:
        log("Ошибки:")
        for number, name, code in failures:
            log(f"  - Шаг {number} «{name}»: код {code}")
        log("#" * 72)
        sys.exit(1)

    log("Все шаги выполнены успешно. ✅")
    log("#" * 72)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nОстановлено пользователем.", flush=True)
        sys.exit(130)
