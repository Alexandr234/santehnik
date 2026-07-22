#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ОРКЕСТРАТОР «документального завода»: запускает все скрипты по порядку.

Порядок шагов:
  1. prompts   — doc_prompt_pipeline_universal_ru_de_es_pl.py
                 (транскрипция озвучек -> промпты + тайминги с паузами-перебивками)
  2. visuals   — flow_visual_batch_generator_realistic.py
                 (генерация ВСЕХ видео по промптам, включая b-roll перебивки)
  3. repair    — flow_repair_missing_videos_100percent.py
                 (догенерация пропусков; повторяется циклом, пока покрытие
                  не станет 100% или не кончатся раунды --repair-rounds)
  4. assemble  — video_creator_times_autovenv_no_subs_realistic.py
                 (финальная сборка: видеоряд + голос с паузами + звук природы)

Оркестратор ожидает, что все 5 скриптов лежат В ОДНОЙ ПАПКЕ с ним самим.

Запуск (обычный полный прогон):
   export OPENAI_API_KEY="sk-..."
   export FAST_GEN_API_KEY="ТВОЙ_КЛЮЧ"
   python3 run_documentary_factory.py

Полезное:
   python3 run_documentary_factory.py --start-from visuals    # начать с шага 2
   python3 run_documentary_factory.py --only assemble         # только финальная сборка
   python3 run_documentary_factory.py --skip prompts          # пропустить шаг(и)
   python3 run_documentary_factory.py --repair-rounds 5       # до 5 раундов ремонта
   python3 run_documentary_factory.py --dry-run               # показать план, ничего не запускать
   python3 run_documentary_factory.py --no-pauses             # без документальных пауз
   python3 run_documentary_factory.py --pause-seconds 2.8 --pause-every 2

Замечания:
  - Если существует <BASE_FOLDER>/venv/bin/python, все шаги запускаются через него
    (как это делает сам сборщик видео).
  - Если какой-то шаг упал, оркестратор останавливается и печатает, с какого шага
    продолжить: --start-from <шаг>.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

# =========================
# КОНФИГ
# =========================

SCRIPTS_DIR = Path(__file__).resolve().parent

DEFAULT_BASE_FOLDER = os.getenv("BASE_FOLDER", "/Users/aleksandrtomilov/Desktop/ПТИЦЫ РЕАЛИСТИЧНЫЕ")

SCRIPT_PROMPTS = "doc_prompt_pipeline_universal_ru_de_es_pl.py"
SCRIPT_VISUALS = "flow_visual_batch_generator_realistic.py"
SCRIPT_REPAIR = "flow_repair_missing_videos_100percent.py"
SCRIPT_ASSEMBLE = "video_creator_times_autovenv_no_subs_realistic.py"

STEP_ORDER = ["prompts", "visuals", "repair", "assemble"]

STEP_SCRIPTS = {
    "prompts": SCRIPT_PROMPTS,
    "visuals": SCRIPT_VISUALS,
    "repair": SCRIPT_REPAIR,
    "assemble": SCRIPT_ASSEMBLE,
}

STEP_TITLES = {
    "prompts": "ПРОМПТЫ И ТАЙМИНГИ (транскрипция, паузы-перебивки)",
    "visuals": "ГЕНЕРАЦИЯ ВИДЕО (все позиции, включая b-roll)",
    "repair": "РЕМОНТ: догенерация пропусков до 100%",
    "assemble": "ФИНАЛЬНАЯ СБОРКА (видеоряд + голос + природа)",
}

# Ключи, которые нужны шагам. Проверяются заранее, чтобы не падать на середине.
STEP_REQUIRED_ENV = {
    "prompts": ["OPENAI_API_KEY"],
    "visuals": ["FAST_GEN_API_KEY|FASTGEN_API_KEY|MEDIA_GEN_API_KEY"],
    "repair": ["FAST_GEN_API_KEY|FASTGEN_API_KEY|MEDIA_GEN_API_KEY"],
    "assemble": [],
}

LOCALES = ["RU", "GE", "PL", "ES"]
LOCALE_PROMPT_FOLDERS = {
    "RU": "RU_русский",
    "GE": "DE_немецкий",
    "PL": "PL_польский",
    "ES": "ES_испанский",
}

TIMECODE_RE = re.compile(
    r"^(?:(\d+)\s*\|\s*)?"
    r"(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*"
    r"(\d{2}:\d{2}:\d{2},\d{3})"
    r"(?:\s*\|.*)?$"
)

MIN_VIDEO_BYTES = 1024


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def banner(title: str) -> None:
    line = "=" * 64
    print(f"\n{line}\n  {title}\n{line}", flush=True)


# =========================
# ОКРУЖЕНИЕ
# =========================

def pick_python(base_folder: Path) -> str:
    """Проектный venv, если он есть; иначе текущий интерпретатор."""
    venv_python = base_folder / "venv" / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


def env_present(spec: str) -> bool:
    """spec вида 'A|B|C' — достаточно любой из переменных."""
    return any(os.getenv(name) for name in spec.split("|"))


def check_env_for_steps(steps: list[str]) -> list[str]:
    problems: list[str] = []
    for step in steps:
        for spec in STEP_REQUIRED_ENV.get(step, []):
            if not env_present(spec):
                pretty = spec.replace("|", " или ")
                problems.append(f"шаг '{step}': не задан ключ {pretty}")
    return problems


def check_scripts_exist(steps: list[str]) -> list[str]:
    problems: list[str] = []
    for step in steps:
        script = SCRIPTS_DIR / STEP_SCRIPTS[step]
        if not script.exists():
            problems.append(f"шаг '{step}': не найден скрипт {script}")
    return problems


# =========================
# ПОКРЫТИЕ ВИДЕО (для цикла ремонта)
# =========================

def newest_prompts_file(language_dir: Path) -> Path | None:
    if not language_dir.exists():
        return None
    candidates = sorted(
        language_dir.glob("*_prompts.txt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def count_prompt_positions(prompts_file: Path) -> int:
    count = 0
    for line in prompts_file.read_text(encoding="utf-8").splitlines():
        if TIMECODE_RE.match(line.strip()):
            count += 1
    return count


def count_valid_videos(visual_dir: Path) -> int:
    if not visual_dir.exists():
        return 0
    count = 0
    for p in visual_dir.glob("*.mp4"):
        if re.match(r"^\d+", p.name):
            try:
                if p.stat().st_size >= MIN_VIDEO_BYTES:
                    count += 1
            except OSError:
                continue
    return count


def coverage_report(base_folder: Path) -> tuple[bool, list[str]]:
    """Возвращает (всё ли 100%, строки отчёта). Языки без промптов пропускаются."""
    prompts_root = base_folder / "ПРОМПТЫ"
    visual_root = base_folder / "ВИЗУАЛ"

    lines: list[str] = []
    all_full = True
    seen_any = False

    for locale in LOCALES:
        language_dir = prompts_root / LOCALE_PROMPT_FOLDERS[locale]
        prompts_file = newest_prompts_file(language_dir)
        if prompts_file is None:
            lines.append(f"  {locale}: нет файла промптов — пропускаю")
            continue
        seen_any = True
        need = count_prompt_positions(prompts_file)
        have = count_valid_videos(visual_root / locale)
        mark = "✅ 100%" if have >= need and need > 0 else f"⚠️ {have}/{need}"
        if need == 0 or have < need:
            all_full = False
        lines.append(f"  {locale}: {have}/{need} {mark}  ({prompts_file.name})")

    if not seen_any:
        all_full = False
        lines.append("  Ни одного файла промптов не найдено.")
    return all_full, lines


# =========================
# ЗАПУСК ШАГОВ
# =========================

def run_step(python_bin: str, step: str, extra_args: list[str], child_env: dict, dry_run: bool) -> None:
    script = SCRIPTS_DIR / STEP_SCRIPTS[step]
    cmd = [python_bin, str(script), *extra_args]
    banner(f"ШАГ {STEP_ORDER.index(step) + 1}/4: {STEP_TITLES[step]}")
    log("Команда: " + " ".join(cmd))
    if dry_run:
        log("[DRY-RUN] запуск пропущен")
        return
    # Вывод шага идёт напрямую в терминал — видно весь прогресс дочернего скрипта.
    result = subprocess.run(cmd, env=child_env)
    if result.returncode != 0:
        raise RuntimeError(
            f"Шаг '{step}' завершился с ошибкой (код {result.returncode}). "
            f"После исправления продолжай с него: --start-from {step}"
        )
    log(f"Шаг '{step}' завершён успешно ✅")


def run_repair_until_full(
    python_bin: str,
    base_folder: Path,
    repair_rounds: int,
    child_env: dict,
    dry_run: bool,
) -> None:
    banner(f"ШАГ 3/4: {STEP_TITLES['repair']} (до {repair_rounds} раундов)")

    if dry_run:
        log(f"[DRY-RUN] {repair_rounds} раунд(ов) ремонта пропущено")
        return

    full, lines = coverage_report(base_folder)
    log("Покрытие перед ремонтом:")
    for line in lines:
        print(line, flush=True)

    if full:
        log("Все видео уже на месте — ремонт не нужен ✅")
        return

    for round_no in range(1, repair_rounds + 1):
        log(f"— Раунд ремонта {round_no}/{repair_rounds} —")
        script = SCRIPTS_DIR / STEP_SCRIPTS["repair"]
        result = subprocess.run([python_bin, str(script)], env=child_env)
        if result.returncode != 0:
            raise RuntimeError(
                f"Ремонтный скрипт упал (код {result.returncode}). "
                f"Продолжить: --start-from repair"
            )

        full, lines = coverage_report(base_folder)
        log(f"Покрытие после раунда {round_no}:")
        for line in lines:
            print(line, flush=True)
        if full:
            log("Покрытие 100% во всех языках ✅")
            return

    raise RuntimeError(
        f"После {repair_rounds} раунд(ов) ремонта покрытие всё ещё не 100% "
        f"(обычно это сеть/лимиты/ключ). Запусти ещё раз: --start-from repair"
    )


# =========================
# MAIN
# =========================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Оркестратор: промпты -> генерация видео -> ремонт до 100% -> финальная сборка."
    )
    parser.add_argument("--base-folder", default=DEFAULT_BASE_FOLDER,
                        help="Корневая папка проекта (в ней ОЗВУЧКА/ПРОМПТЫ/ВИЗУАЛ/ГОТОВЫЕ ВИДЕО).")
    parser.add_argument("--start-from", choices=STEP_ORDER, default="prompts",
                        help="Начать с этого шага (предыдущие пропускаются).")
    parser.add_argument("--only", choices=STEP_ORDER, default=None,
                        help="Выполнить только один шаг.")
    parser.add_argument("--skip", nargs="*", choices=STEP_ORDER, default=[],
                        help="Пропустить перечисленные шаги.")
    parser.add_argument("--repair-rounds", type=int, default=3,
                        help="Максимум раундов ремонтного скрипта (default: 3).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Показать план и команды, ничего не запуская.")
    # Пробрасывается в шаг prompts:
    parser.add_argument("--no-pauses", action="store_true",
                        help="Отключить документальные паузы (уходит в шаг prompts).")
    parser.add_argument("--pause-seconds", type=float, default=None,
                        help="Средняя длина паузы-перебивки, сек (уходит в шаг prompts).")
    parser.add_argument("--pause-every", type=int, default=None,
                        help="Пауза после каждой N-й фразы (уходит в шаг prompts).")
    args = parser.parse_args()

    base_folder = Path(args.base_folder).expanduser()

    # Какие шаги выполняем.
    if args.only:
        steps = [args.only]
    else:
        steps = STEP_ORDER[STEP_ORDER.index(args.start_from):]
    steps = [s for s in steps if s not in set(args.skip)]

    if not steps:
        print("Нечего запускать: все шаги пропущены.", file=sys.stderr)
        sys.exit(1)

    banner("ДОКУМЕНТАЛЬНЫЙ ЗАВОД — ПЛАН ЗАПУСКА")
    log(f"Папка проекта: {base_folder}")
    log(f"Папка скриптов: {SCRIPTS_DIR}")
    for s in steps:
        log(f"  -> {s}: {STEP_TITLES[s]}")

    # Предварительные проверки: скрипты на месте, ключи заданы, папка существует.
    problems = check_scripts_exist(steps) + check_env_for_steps(steps)
    if not base_folder.exists():
        problems.append(f"папка проекта не найдена: {base_folder}")
    if problems and not args.dry_run:
        print("\n❌ Нельзя стартовать:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        sys.exit(1)
    if problems and args.dry_run:
        log("⚠️ Проблемы (в dry-run не блокируют):")
        for p in problems:
            log(f"  - {p}")

    python_bin = pick_python(base_folder)
    log(f"Python для шагов: {python_bin}")

    # Дочерние скрипты читают BASE_FOLDER из окружения (где поддерживают).
    child_env = os.environ.copy()
    child_env["BASE_FOLDER"] = str(base_folder)

    prompts_args: list[str] = []
    if args.no_pauses:
        prompts_args.append("--no-pauses")
    if args.pause_seconds is not None:
        prompts_args.extend(["--pause-seconds", str(args.pause_seconds)])
    if args.pause_every is not None:
        prompts_args.extend(["--pause-every", str(args.pause_every)])

    started = time.time()
    try:
        for step in steps:
            if step == "prompts":
                run_step(python_bin, step, prompts_args, child_env, args.dry_run)
            elif step == "repair":
                run_repair_until_full(python_bin, base_folder, args.repair_rounds, child_env, args.dry_run)
            else:
                run_step(python_bin, step, [], child_env, args.dry_run)
    except KeyboardInterrupt:
        print("\n⏹ Остановлено пользователем.", file=sys.stderr)
        sys.exit(130)
    except RuntimeError as exc:
        print(f"\n❌ {exc}", file=sys.stderr)
        sys.exit(1)

    elapsed_min = (time.time() - started) / 60
    banner("ГОТОВО")
    log(f"Все шаги выполнены за {elapsed_min:.1f} мин.")
    log(f"Готовые ролики: {base_folder / 'ГОТОВЫЕ ВИДЕО'}")


if __name__ == "__main__":
    main()
