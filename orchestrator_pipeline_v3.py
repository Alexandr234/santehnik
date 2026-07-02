#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Оркестратор для последовательного запуска пайплайна из трёх скриптов:

  1) master_prompt_pipeline_children_enabled_no_famous_APIKEY_FIXED_v4ДЕТИ.py
  2) image_prompt_converter_api_character_aliases.py
  3) video_from_local_images_base64.py

Что делает:
- запускает скрипты строго по очереди;
- в реальном времени показывает stdout/stderr каждого скрипта в окне вывода;
- пишет общий лог в файл logs/orchestrator_YYYYmmdd_HHMMSS.log;
- ПРОПУСКАЕТ этап, если результат его работы уже лежит в папках
  (не запускает генерацию заново);
- останавливает пайплайн, если любой этап завершился с ошибкой;
- умеет находить файл, даже если в названии «похожие» пробелы/тире/Unicode-символы.

Запуск:
    python3 orchestrator_pipeline_v3.py

Полезные ключи запуска:
    --force            запустить все этапы заново, игнорируя уже готовые результаты
    --force VIDEO_GEN  запустить заново только конкретный этап (по его name)
    --list             показать, что оркестратор считает «готовым», и выйти

При необходимости поменяйте PYTHON_BIN на путь к вашему venv-интерпретатору,
а output_globs в PIPELINE — под реальные выходные папки ваших скриптов.
"""

from __future__ import annotations

import os
import sys
import glob
import time
import queue
import signal
import threading
import subprocess
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional


# ============================================================
# НАСТРОЙКИ
# ============================================================

PYTHON_BIN = sys.executable  # или укажите руками, например: "/Users/.../venv/bin/python3"
STOP_ON_ERROR = True
CREATE_LOG_FILE = True
SHOW_RESOLVED_PATHS = True

BASE = "/Users/aleksandrtomilov/Desktop/ПРОМПТЫ"

# ------------------------------------------------------------
# Описание пайплайна.
#
# Для каждого этапа:
#   name         — читаемое имя этапа (используется в логах и в --force NAME).
#   path         — путь к скрипту.
#   output_globs — список glob-шаблонов выходных файлов/папок.
#                  Если хотя бы по одному шаблону найдётся непустой файл —
#                  этап считается УЖЕ ВЫПОЛНЕННЫМ и оркестратор его ПРОПУСТИТ.
#                  Пути можно задавать абсолютными или относительно папки
#                  самого скрипта (см. cwd этапа). Шаблоны поддерживают ** .
#
# ВАЖНО: подправьте output_globs под реальные выходные папки ваших скриптов.
# Значения ниже — разумные значения по умолчанию по смыслу каждого этапа.
# ------------------------------------------------------------

PIPELINE = [
    {
        # 1) Генерация текстовых промптов (сценарии/промпты).
        "name": "MASTER_PROMPTS",
        "path": f"{BASE}/master_prompt_pipeline_children_enabled_no_famous_APIKEY_FIXED_v4ДЕТИ.py",
        "output_globs": [
            # Готовые промпты обычно складываются рядом со скриптом.
            f"{BASE}/**/*prompt*.txt",
            f"{BASE}/**/*prompt*.json",
            f"{BASE}/output/**/*.txt",
            f"{BASE}/prompts/**/*.txt",
        ],
    },
    {
        # 2) Конвертация промптов в промпты для картинок + алиасы персонажей.
        "name": "IMAGE_PROMPTS",
        "path": f"{BASE}/СКРИПТ КАРТИНКИ/image_prompt_converter_api_character_aliases.py",
        "output_globs": [
            f"{BASE}/СКРИПТ КАРТИНКИ/**/*image*prompt*.txt",
            f"{BASE}/СКРИПТ КАРТИНКИ/**/*image*prompt*.json",
            f"{BASE}/СКРИПТ КАРТИНКИ/output/**/*.txt",
            f"{BASE}/СКРИПТ КАРТИНКИ/output/**/*.json",
        ],
    },
    {
        # 3) Генерация видео из локальных картинок (base64).
        "name": "VIDEO_GENERATION",
        "path": f"{BASE}/ГЕНЕРАЦИЯ ВИДЕО СКРИПТ/video_from_local_images_base64.py",
        "output_globs": [
            f"{BASE}/ГЕНЕРАЦИЯ ВИДЕО СКРИПТ/**/*.mp4",
            f"{BASE}/ГЕНЕРАЦИЯ ВИДЕО СКРИПТ/output/**/*.mp4",
            f"{BASE}/ГЕНЕРАЦИЯ ВИДЕО СКРИПТ/videos/**/*.mp4",
        ],
    },
]


# ============================================================
# СЛУЖЕБНОЕ
# ============================================================

@dataclass
class Step:
    name: str
    raw_path: str
    output_globs: list[str] = field(default_factory=list)
    resolved_path: Optional[Path] = None


class Logger:
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._lock = threading.Lock()
        self.log_file = None

        if enabled:
            logs_dir = Path.cwd() / "logs"
            logs_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_path = logs_dir / f"orchestrator_{stamp}.log"
            self.log_file = self.log_path.open("a", encoding="utf-8")
        else:
            self.log_path = None

    def write(self, text: str) -> None:
        with self._lock:
            print(text, flush=True)
            if self.log_file:
                self.log_file.write(text + "\n")
                self.log_file.flush()

    def close(self) -> None:
        if self.log_file:
            self.log_file.close()


def now_ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def print_banner(logger: Logger, title: str, char: str = "=") -> None:
    line = char * max(20, len(title) + 8)
    logger.write("")
    logger.write(line)
    logger.write(f"=== {title} ===")
    logger.write(line)


def normalize_for_match(value: str) -> str:
    """
    Нормализует строку так, чтобы можно было найти файл,
    даже если там неразрывные пробелы, разные тире и т.п.
    """
    value = unicodedata.normalize("NFKC", value)

    replacements = {
        "\u00A0": " ",  # no-break space
        "\u2007": " ",
        "\u202F": " ",
        "\u2009": " ",
        "\u2002": " ",
        "\u2003": " ",
        "\u2013": "-",  # en dash
        "\u2014": "-",  # em dash
        "\u2212": "-",  # minus sign
        "ё": "е",  # ё -> е
        "Ё": "Е",  # Ё -> Е
    }
    for src, dst in replacements.items():
        value = value.replace(src, dst)

    value = " ".join(value.split())
    return value.casefold().strip()


def resolve_script_path(raw_path: str) -> Path:
    """
    1) Пробует exact path.
    2) Если не найден, ищет в той же папке файл с "похожим" именем.
    """
    candidate = Path(raw_path).expanduser()
    if candidate.exists():
        return candidate

    parent = candidate.parent
    target_name_norm = normalize_for_match(candidate.name)

    if not parent.exists():
        raise FileNotFoundError(
            f"Папка не найдена: {parent}\n"
            f"Проверьте путь в PIPELINE."
        )

    matches: list[Path] = []
    for child in parent.iterdir():
        if not child.is_file():
            continue
        child_name_norm = normalize_for_match(child.name)
        if child_name_norm == target_name_norm:
            matches.append(child)

    if len(matches) == 1:
        return matches[0]

    if len(matches) > 1:
        raise FileNotFoundError(
            "Найдено несколько похожих файлов, нельзя однозначно выбрать:\n"
            + "\n".join(f"- {m}" for m in matches)
        )

    # более мягкий поиск: по частичному совпадению основы имени
    stem_norm = normalize_for_match(candidate.stem)
    soft_matches: list[Path] = []
    for child in parent.iterdir():
        if not child.is_file():
            continue
        child_stem_norm = normalize_for_match(child.stem)
        if stem_norm in child_stem_norm or child_stem_norm in stem_norm:
            soft_matches.append(child)

    if len(soft_matches) == 1:
        return soft_matches[0]

    debug_list = "\n".join(f"- {p.name}" for p in sorted(parent.iterdir()) if p.is_file())
    raise FileNotFoundError(
        f"Не удалось найти файл:\n{candidate}\n\n"
        f"Что пробовал:\n"
        f"- exact path\n"
        f"- Unicode-normalized name match\n"
        f"- soft stem match\n\n"
        f"Файлы в папке {parent}:\n{debug_list}"
    )


def _iter_existing_outputs(step: Step) -> list[Path]:
    """
    Возвращает список непустых файлов, найденных по output_globs этапа.
    Относительные шаблоны раскрываются относительно папки скрипта.
    """
    found: list[Path] = []
    base_dir = step.resolved_path.parent if step.resolved_path else Path.cwd()

    for pattern in step.output_globs:
        pat = pattern
        if not os.path.isabs(pat):
            pat = str(base_dir / pat)
        for match in glob.glob(pat, recursive=True):
            p = Path(match)
            try:
                if p.is_file() and p.stat().st_size > 0:
                    found.append(p)
            except OSError:
                continue
    return found


def is_already_done(step: Step) -> tuple[bool, list[Path]]:
    """
    Этап считается выполненным, если задан хотя бы один output_glob
    и по нему найден непустой файл.
    Если output_globs пуст — этап никогда не пропускается.
    """
    if not step.output_globs:
        return False, []
    found = _iter_existing_outputs(step)
    return (len(found) > 0), found


def stream_reader(pipe, source_name: str, step_name: str, q: queue.Queue) -> None:
    try:
        for line in iter(pipe.readline, ""):
            q.put((source_name, step_name, line.rstrip("\n")))
    finally:
        try:
            pipe.close()
        except Exception:
            pass


def terminate_process_tree(proc: subprocess.Popen, logger: Logger) -> None:
    if proc.poll() is not None:
        return

    logger.write(f"[{now_ts()}] [ORCHESTRATOR] Останавливаю текущий процесс PID={proc.pid}...")

    try:
        if os.name == "nt":
            proc.terminate()
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except Exception as e:
        logger.write(f"[{now_ts()}] [ORCHESTRATOR] Не удалось отправить SIGTERM: {e}")

    try:
        proc.wait(timeout=10)
        return
    except subprocess.TimeoutExpired:
        pass

    logger.write(f"[{now_ts()}] [ORCHESTRATOR] Процесс не завершился после SIGTERM, отправляю принудительное завершение...")
    try:
        if os.name == "nt":
            proc.kill()
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception as e:
        logger.write(f"[{now_ts()}] [ORCHESTRATOR] Не удалось отправить SIGKILL: {e}")


def run_step(step: Step, logger: Logger) -> int:
    assert step.resolved_path is not None

    print_banner(logger, f"СТАРТ ЭТАПА: {step.name}")
    logger.write(f"[{now_ts()}] [ORCHESTRATOR] Скрипт: {step.resolved_path}")
    logger.write(f"[{now_ts()}] [ORCHESTRATOR] Интерпретатор: {PYTHON_BIN}")

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    # На macOS/Linux создаём отдельную process group, чтобы корректно убивать дочерние процессы.
    kwargs = {}
    if os.name != "nt":
        kwargs["preexec_fn"] = os.setsid
    else:
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]

    started = time.time()
    proc = subprocess.Popen(
        [PYTHON_BIN, "-u", str(step.resolved_path)],
        cwd=str(step.resolved_path.parent),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        encoding="utf-8",
        errors="replace",
        env=env,
        **kwargs,
    )

    q: queue.Queue = queue.Queue()
    stdout_thread = threading.Thread(target=stream_reader, args=(proc.stdout, "STDOUT", step.name, q), daemon=True)
    stderr_thread = threading.Thread(target=stream_reader, args=(proc.stderr, "STDERR", step.name, q), daemon=True)
    stdout_thread.start()
    stderr_thread.start()

    try:
        while True:
            try:
                source, step_name, line = q.get(timeout=0.15)
                logger.write(f"[{now_ts()}] [{step_name}] [{source}] {line}")
            except queue.Empty:
                if proc.poll() is not None:
                    break
    except KeyboardInterrupt:
        logger.write(f"[{now_ts()}] [ORCHESTRATOR] Получен Ctrl+C.")
        terminate_process_tree(proc, logger)
        raise

    # дочитываем хвост
    time.sleep(0.2)
    while not q.empty():
        source, step_name, line = q.get_nowait()
        logger.write(f"[{now_ts()}] [{step_name}] [{source}] {line}")

    return_code = proc.wait()
    elapsed = time.time() - started

    status = "УСПЕШНО" if return_code == 0 else "ОШИБКА"
    logger.write(f"[{now_ts()}] [ORCHESTRATOR] Этап {step.name} завершён: {status} | code={return_code} | {elapsed:.1f}s")
    return return_code


def build_steps() -> list[Step]:
    steps: list[Step] = []
    for item in PIPELINE:
        steps.append(
            Step(
                name=item["name"],
                raw_path=item["path"],
                output_globs=list(item.get("output_globs", [])),
            )
        )
    return steps


def parse_args(argv: list[str]) -> tuple[set[str], bool, bool]:
    """
    Возвращает (force_names, force_all, list_only).
    --force без имени  -> force_all = True
    --force NAME ...   -> в force_names попадают перечисленные имена
    --list             -> list_only = True
    """
    force_names: set[str] = set()
    force_all = False
    list_only = False

    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--list":
            list_only = True
        elif arg == "--force":
            # собираем идущие следом имена (не начинающиеся с --)
            names = []
            j = i + 1
            while j < len(argv) and not argv[j].startswith("--"):
                names.append(argv[j])
                j += 1
            if names:
                force_names.update(names)
            else:
                force_all = True
            i = j - 1
        i += 1

    return force_names, force_all, list_only


def main() -> int:
    force_names, force_all, list_only = parse_args(sys.argv[1:])

    logger = Logger(enabled=CREATE_LOG_FILE)

    try:
        print_banner(logger, "ПОДГОТОВКА ПАЙПЛАЙНА")
        logger.write(f"[{now_ts()}] [ORCHESTRATOR] Рабочая папка: {Path.cwd()}")
        logger.write(f"[{now_ts()}] [ORCHESTRATOR] Python: {PYTHON_BIN}")
        if force_all:
            logger.write(f"[{now_ts()}] [ORCHESTRATOR] Режим --force: все этапы будут запущены заново.")
        elif force_names:
            logger.write(f"[{now_ts()}] [ORCHESTRATOR] Режим --force для этапов: {', '.join(sorted(force_names))}")

        steps = build_steps()

        for step in steps:
            step.resolved_path = resolve_script_path(step.raw_path)
            if SHOW_RESOLVED_PATHS:
                logger.write(f"[{now_ts()}] [ORCHESTRATOR] {step.name} -> {step.resolved_path}")

        # Режим --list: просто показать статус готовности и выйти.
        if list_only:
            print_banner(logger, "СТАТУС ГОТОВНОСТИ ЭТАПОВ")
            for step in steps:
                done, found = is_already_done(step)
                mark = "ГОТОВО" if done else "нужно запускать"
                logger.write(f"[{now_ts()}] [ORCHESTRATOR] {step.name}: {mark}")
                for p in found[:10]:
                    logger.write(f"    - {p}")
                if len(found) > 10:
                    logger.write(f"    ... и ещё {len(found) - 10} файл(ов)")
            return 0

        print_banner(logger, "ЗАПУСК ПАЙПЛАЙНА")
        total_started = time.time()

        skipped = 0
        for index, step in enumerate(steps, start=1):
            logger.write(f"[{now_ts()}] [ORCHESTRATOR] Этап {index}/{len(steps)}: {step.name}")

            force_this = force_all or (step.name in force_names)

            if not force_this:
                done, found = is_already_done(step)
                if done:
                    print_banner(logger, f"ПРОПУСК ЭТАПА: {step.name} (результат уже есть)", char="-")
                    logger.write(f"[{now_ts()}] [ORCHESTRATOR] Найдены готовые результаты ({len(found)} файл(ов)), генерация не запускается:")
                    for p in found[:10]:
                        logger.write(f"    - {p}")
                    if len(found) > 10:
                        logger.write(f"    ... и ещё {len(found) - 10} файл(ов)")
                    skipped += 1
                    continue

            code = run_step(step, logger)

            if code != 0 and STOP_ON_ERROR:
                print_banner(logger, "ПАЙПЛАЙН ОСТАНОВЛЕН ИЗ-ЗА ОШИБКИ", char="!")
                logger.write(f"[{now_ts()}] [ORCHESTRATOR] Неуспешный этап: {step.name}")
                return code

        total_elapsed = time.time() - total_started
        print_banner(logger, "ПАЙПЛАЙН ЗАВЕРШЁН УСПЕШНО", char="#")
        logger.write(f"[{now_ts()}] [ORCHESTRATOR] Пропущено этапов (уже готовы): {skipped}/{len(steps)}")
        logger.write(f"[{now_ts()}] [ORCHESTRATOR] Общее время: {total_elapsed:.1f}s")
        if logger.log_path:
            logger.write(f"[{now_ts()}] [ORCHESTRATOR] Лог: {logger.log_path}")
        return 0

    except FileNotFoundError as e:
        print_banner(logger, "ОШИБКА ПОИСКА ФАЙЛА", char="!")
        logger.write(str(e))
        return 2
    except KeyboardInterrupt:
        print_banner(logger, "ОСТАНОВЛЕНО ПОЛЬЗОВАТЕЛЕМ", char="!")
        return 130
    except Exception as e:
        print_banner(logger, "НЕПРЕДВИДЕННАЯ ОШИБКА", char="!")
        logger.write(repr(e))
        return 1
    finally:
        logger.close()


if __name__ == "__main__":
    raise SystemExit(main())
