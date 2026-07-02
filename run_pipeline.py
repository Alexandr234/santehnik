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

ПРОВЕРКА ГОТОВНОСТИ (главное новое):
  Перед каждым шагом оркестратор проверяет, есть ли уже все нужные файлы-результаты
  этого шага. Если результаты готовы — шаг ПРОПУСКАЕТСЯ (не тратим время и лимиты API):

    - Озвучка готова, если в папке ОЗВУЧКА лежат scenario_ru/pl/de.mp3.
    - Промпты готовы, если в ПРОМПТЫ лежат prompts_{ru,pl,de}.txt и image_times_{ru,pl,de}.json.
    - Картинки готовы, если в ВИЗУАЛ число PNG совпадает с числом кадров из image_times_*.json.
    - Видео готово, если в МОНТАЖ лежит готовый .mp4.
    - Шаг «КАРЛ ЮНГ» проверить нечем (выход неизвестен) — он запускается всегда,
      пока не пропишешь его файлы в KARL_JUNG_OUTPUTS ниже.

  Что именно считать «готовым» — настраивается в блоке НАСТРОЙКИ ПУТЕЙ ниже.
  Принудительно перезапустить: --no-skip (всё) или --force N (конкретные шаги).

По умолчанию: если какой-то шаг ПАДАЕТ (ненулевой код), пайплайн останавливается.
Пропуск готового шага падением НЕ считается. Флаг --keep-going не останавливаться на ошибке.

Запуск:
   cd "/Users/aleksandrtomilov/Desktop/ПСИХОЛОГИЯ ГЕРМАНИЯ ПОЛЬША"
   python3 run_pipeline.py

Полезные флаги:
   python3 run_pipeline.py --list           # показать шаги и их готовность, выйти
   python3 run_pipeline.py --no-skip         # прогнать все шаги, даже если готовы
   python3 run_pipeline.py --force 4 5       # заставить перезапустить шаги 4 и 5
   python3 run_pipeline.py --from 3          # начать с 3-го шага
   python3 run_pipeline.py --only 2 4        # запустить только шаги 2 и 4 (они форсируются)
   python3 run_pipeline.py --keep-going      # не останавливаться на ошибке
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

# ============================================================
# НАСТРОЙКИ ПУТЕЙ
# ------------------------------------------------------------
# Корневая папка проекта. Все пути строятся от неё.
BASE_DIR = Path("/Users/aleksandrtomilov/Desktop/ПСИХОЛОГИЯ ГЕРМАНИЯ ПОЛЬША")

# Папки-результаты (где искать готовые файлы для проверки готовности).
VOICE_DIR   = BASE_DIR / "ОЗВУЧКА"     # сюда voicer кладёт mp3
PROMPTS_DIR = BASE_DIR / "ПРОМПТЫ"     # сюда pipeline кладёт промпты и тайминги
VISUAL_DIR  = BASE_DIR / "ВИЗУАЛ"      # сюда генератор кладёт png
MONTAGE_DIR = BASE_DIR / "МОНТАЖ"      # сюда монтажёр кладёт итоговое видео

# Языки и имена языковых подпапок.
LANGS = ["ru", "pl", "de"]
LANG_SUBFOLDER = {"ru": "RU", "pl": "PL", "de": "DE"}

# Ожидаемые имена готовых озвучек в VOICE_DIR.
VOICEOVER_FILES = ["scenario_ru.mp3", "scenario_pl.mp3", "scenario_de.mp3"]

# Минимум готовых видеофайлов в MONTAGE_DIR, чтобы считать шаг 5 выполненным.
# Если делаешь по одному видео на язык — поставь 3.
MIN_FINAL_VIDEOS = 1
VIDEO_EXTS = ["*.mp4", "*.mov", "*.mkv"]

# Выходные файлы шага «КАРЛ ЮНГ» (шаг 3). Пусто = проверить нельзя, шаг запускается ВСЕГДА.
# Если знаешь, что он создаёт (например, ПРОМПТЫ/jung_ru.txt) — впиши пути сюда,
# и тогда шаг будет пропускаться, когда эти файлы уже есть. Пути можно абсолютные
# или с шаблонами (glob), напр. PROMPTS_DIR / "jung_*.txt".
KARL_JUNG_OUTPUTS: list[Path] = [
    # PROMPTS_DIR / "jung_ru.txt",
    # PROMPTS_DIR / "jung_pl.txt",
    # PROMPTS_DIR / "jung_de.txt",
]


# ============================================================
# ХЕЛПЕРЫ ПРОВЕРКИ ФАЙЛОВ
# ============================================================
def _nonempty(path: Path, min_bytes: int = 1) -> bool:
    try:
        return path.is_file() and path.stat().st_size >= min_bytes
    except OSError:
        return False


def _glob_nonempty(folder: Path, pattern: str, min_bytes: int = 1) -> list[Path]:
    if not folder.exists():
        return []
    return [p for p in folder.glob(pattern) if _nonempty(p, min_bytes)]


def _expected_image_count(lang: str) -> Optional[int]:
    """Сколько картинок должно быть для языка — по числу кадров в image_times_{lang}.json."""
    j = PROMPTS_DIR / f"image_times_{lang}.json"
    if not _nonempty(j, 2):
        # запасной вариант: файл в языковой подпапке (*_image_times.json)
        sub = _glob_nonempty(PROMPTS_DIR / LANG_SUBFOLDER[lang], "*_image_times.json", 2)
        if not sub:
            return None
        j = sub[0]
    try:
        data = json.loads(j.read_text(encoding="utf-8"))
        return len(data) if isinstance(data, list) else None
    except Exception:
        return None


def _count_images(lang: str) -> int:
    """Число готовых PNG для языка. Ищем в ВИЗУАЛ/<LANG>/, потом мягкий фолбэк."""
    sub = VISUAL_DIR / LANG_SUBFOLDER[lang]
    pngs = _glob_nonempty(sub, "*.png")
    if not pngs:
        # мягкий фолбэк: любые png в подпапке, чьё имя содержит код языка
        for cand in VISUAL_DIR.glob(f"*{lang}*"):
            if cand.is_dir():
                pngs += _glob_nonempty(cand, "*.png")
    return len(pngs)


# ============================================================
# ФУНКЦИИ ГОТОВНОСТИ ШАГОВ
# Каждая возвращает (готово: bool, пояснение: str).
# ============================================================
def ready_voiceover() -> tuple[bool, str]:
    missing = [name for name in VOICEOVER_FILES if not _nonempty(VOICE_DIR / name, 1024)]
    if missing:
        return False, f"нет: {', '.join(missing)}"
    return True, f"все {len(VOICEOVER_FILES)} озвучки на месте"


def ready_prompts() -> tuple[bool, str]:
    oks, details = [], []
    for lang in LANGS:
        txt = _nonempty(PROMPTS_DIR / f"prompts_{lang}.txt", 100)
        times = _nonempty(PROMPTS_DIR / f"image_times_{lang}.json", 2)
        if not (txt and times):
            sub = PROMPTS_DIR / LANG_SUBFOLDER[lang]
            txt = txt or bool(_glob_nonempty(sub, "*_prompts.txt", 100))
            times = times or bool(_glob_nonempty(sub, "*_image_times.json", 2))
        ok = txt and times
        oks.append(ok)
        details.append(f"{lang.upper()}:{'ok' if ok else '—'}")
    return all(oks), ", ".join(details)


def ready_karl_jung() -> Optional[tuple[bool, str]]:
    """Вернёт None, если проверять нечем (KARL_JUNG_OUTPUTS пуст) — тогда шаг запускается всегда."""
    if not KARL_JUNG_OUTPUTS:
        return None
    missing = []
    for spec in KARL_JUNG_OUTPUTS:
        # spec может быть шаблоном (есть * ? [) или точным путём
        if any(ch in spec.name for ch in "*?["):
            if not _glob_nonempty(spec.parent, spec.name, 1):
                missing.append(spec.name)
        elif not _nonempty(spec, 1):
            missing.append(str(spec))
    if missing:
        return False, f"нет: {', '.join(missing)}"
    return True, "выходные файлы КАРЛ ЮНГ на месте"


def ready_images() -> tuple[bool, str]:
    oks, details = [], []
    for lang in LANGS:
        have = _count_images(lang)
        want = _expected_image_count(lang)
        if want is None:
            ok = have > 0
            details.append(f"{lang.upper()}:{have}(?)")
        else:
            ok = want > 0 and have >= want
            details.append(f"{lang.upper()}:{have}/{want}")
        oks.append(ok)
    return (all(oks) and len(oks) > 0), ", ".join(details)


def ready_video() -> tuple[bool, str]:
    vids: list[Path] = []
    for ext in VIDEO_EXTS:
        vids += _glob_nonempty(MONTAGE_DIR, ext, 1024)
    return len(vids) >= MIN_FINAL_VIDEOS, f"{len(vids)} видео (нужно ≥{MIN_FINAL_VIDEOS})"


# ============================================================
# ОПИСАНИЕ ШАГОВ
# ============================================================
@dataclass
class Step:
    name: str
    script: Path
    # Функция готовности: (готово, пояснение) или None если проверить нельзя (всегда запускать).
    ready: Optional[Callable[[], Optional[tuple[bool, str]]]]


STEPS: list[Step] = [
    Step("Озвучка RU/PL/DE",   BASE_DIR / "СКРИПТ ОЗВУЧКИ" / "voicer_batch_tts_RU_PL_DE.py", ready_voiceover),
    Step("Генерация промптов", BASE_DIR / "ПРОМПТЫ" / "psych_prompt_pipeline_v2карлюнг.py",  ready_prompts),
    Step("Карл Юнг",           BASE_DIR / "ПРОМПТЫ" / "КАРЛ ЮНГ.py",                          ready_karl_jung),
    Step("Генерация картинок", BASE_DIR / "ВИЗУАЛ" / "flower_image_generator2юнг.py",         ready_images),
    Step("Сборка видео",       BASE_DIR / "МОНТАЖ" / "video_creator_zoom15.py",               ready_video),
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


def check_ready(step: Step) -> Optional[tuple[bool, str]]:
    """Безопасно вызывает функцию готовности. None = проверить нельзя."""
    if step.ready is None:
        return None
    try:
        return step.ready()
    except Exception as exc:
        # Любая ошибка проверки трактуется как «не готово» — лучше перезапустить, чем пропустить нужное.
        return False, f"ошибка проверки: {exc}"


def run_step(index: int, total: int, name: str, script: Path) -> int:
    """Запускает один скрипт и стримит его вывод построчно. Возвращает код возврата (0 = успех)."""
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
        description="Оркестратор пайплайна: запускает скрипты друг за другом, пропускает готовые, стримит вывод.",
    )
    parser.add_argument("--keep-going", action="store_true",
                        help="Не останавливаться на ошибке — выполнить все шаги, ошибки собрать в конце.")
    parser.add_argument("--no-skip", action="store_true",
                        help="Не пропускать готовые шаги — прогнать всё заново.")
    parser.add_argument("--force", type=int, nargs="+", metavar="N", default=None,
                        help="Принудительно перезапустить указанные шаги, даже если они готовы.")
    parser.add_argument("--from", dest="from_step", type=int, default=1, metavar="N",
                        help="Начать с шага N (1..%d)." % len(STEPS))
    parser.add_argument("--only", type=int, nargs="+", metavar="N", default=None,
                        help="Запустить только указанные шаги (они форсируются).")
    parser.add_argument("--list", action="store_true",
                        help="Показать список шагов и их готовность, затем выйти.")
    return parser.parse_args()


def select_steps(args: argparse.Namespace) -> list[tuple[int, Step]]:
    """Возвращает список (номер, Step) с учётом --from / --only."""
    numbered = list(enumerate(STEPS, start=1))
    if args.only:
        chosen = set(args.only)
        return [(n, s) for n, s in numbered if n in chosen]
    return [(n, s) for n, s in numbered if n >= args.from_step]


def main() -> None:
    args = parse_args()

    if args.list:
        log("Шаги пайплайна и готовность результатов:")
        for i, step in enumerate(STEPS, start=1):
            exists = "✓" if step.script.exists() else "✗ нет файла"
            r = check_ready(step)
            if r is None:
                status = "проверить нельзя → запуск всегда"
            else:
                ready, detail = r
                status = ("ГОТОВО → пропуск" if ready else "не готово → запуск") + f" ({detail})"
            log(f"  {i}. {step.name}  [{exists}]  {status}")
            log(f"     {step.script}")
        return

    steps = select_steps(args)
    if not steps:
        log("[ВНИМАНИЕ] Нет шагов для запуска (проверь --from / --only).")
        sys.exit(1)

    # Какие шаги форсируем (запуск даже если готовы).
    forced: set[int] = set()
    if args.force:
        forced |= set(args.force)
    if args.only:
        forced |= set(args.only)  # явно выбрал --only → значит хочешь их выполнить

    total = len(STEPS)
    pipeline_started = time.time()

    log("#" * 72)
    log("ОРКЕСТРАТОР ПАЙПЛАЙНА")
    log(f"Старт: {stamp()}")
    log(f"К запуску: {', '.join(str(n) for n, _ in steps)} из {total}")
    log(f"Пропуск готовых: {'ВЫКЛ' if args.no_skip else 'ВКЛ'}"
        + (f" | форс: {sorted(forced)}" if forced else ""))
    log(f"Режим ошибок: {'продолжать' if args.keep_going else 'останавливаться на первой'}")
    log("#" * 72)

    failures: list[tuple[int, str, int]] = []
    skipped: list[tuple[int, str, str]] = []
    ran = 0

    for number, step in steps:
        # Решаем, можно ли пропустить.
        if not args.no_skip and number not in forced:
            r = check_ready(step)
            if r is not None and r[0]:
                log("")
                log("=" * 72)
                log(f"[ПРОПУСК {number}/{total}] {step.name} — результаты уже готовы ({r[1]})")
                log("=" * 72)
                skipped.append((number, step.name, r[1]))
                continue

        code = run_step(number, total, step.name, step.script)
        ran += 1
        if code != 0:
            failures.append((number, step.name, code))
            if not args.keep_going:
                log("")
                log(f"[СТОП] Шаг {number} «{step.name}» упал. Останавливаю пайплайн (--keep-going чтобы продолжать).")
                break

    total_time = time.time() - pipeline_started

    log("")
    log("#" * 72)
    log(f"ИТОГ. Запущено: {ran} | пропущено: {len(skipped)} | ошибок: {len(failures)}. "
        f"Общее время: {fmt_duration(total_time)}. Финиш: {stamp()}")
    if skipped:
        log("Пропущены (уже готовы):")
        for number, name, detail in skipped:
            log(f"  - Шаг {number} «{name}»: {detail}")
    if failures:
        log("Ошибки:")
        for number, name, code in failures:
            log(f"  - Шаг {number} «{name}»: код {code}")
        log("#" * 72)
        sys.exit(1)

    log("Все нужные шаги выполнены успешно. ✅")
    log("#" * 72)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nОстановлено пользователем.", flush=True)
        sys.exit(130)
