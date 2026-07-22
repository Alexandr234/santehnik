#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ПЕРЕМОНТАЖНИК: чинит позиции пауз БЕЗ перегенерации визуала.

Ситуация: визуал уже полностью сгенерирован по старым таймингам, где паузы-перебивки
(b-roll) стояли после КАЖДОГО блока — в том числе посреди незаконченного предложения
(«...я расскажу об одном из самых редких» [ПАУЗА] «союзов...»). Звучит обрывисто.

Что делает этот скрипт (для каждого языка RU/GE/PL/ES):
  1. Читает текущий файл таймингов ПРОМПТЫ/image_times_<xx>.txt
     (строки с type: speech / broll и src: ... для речи).
  2. Находит ТЕКСТ каждой речевой позиции (из *_prompts.json, либо
     *_visual_blocks.json, либо из *.srt в языковой папке ПРОМПТОВ).
  3. Определяет, какие фразы заканчиваются ПОСРЕДИ предложения (нет . ! ? … в конце).
  4. ПЕРЕДВИГАЕТ паузы: перебивка, стоявшая после оборванной фразы, уезжает вперёд —
     за конец ближайшего ЗАКОНЧЕННОГО предложения. Сами видео не трогаются и не
     перегенерируются: меняется только их ПОРЯДОК и тайминги.
  5. Пересчитывает финальную шкалу, ПЕРЕИМЕНОВЫВАЕТ mp4-файлы в папке ВИЗУАЛ/<ЯЗЫК>
     под новый порядок (двухфазно, с бэкапом манифеста переименований) и
     перезаписывает файл таймингов (старый сохраняется в *.bak_...).
  6. По желанию сразу запускает финальную сборку (--assemble): сборщик порежет
     озвучку по фразам, вставит тишину в новые (правильные) паузы и соберёт звук
     природы из клипов.

Ничего не генерируется заново: используются ровно те же mp4.

Запуск:
   python3 remontage_fix_pauses.py --dry-run      # показать, что изменится
   python3 remontage_fix_pauses.py                # применить
   python3 remontage_fix_pauses.py --assemble     # применить и сразу собрать ролики
   python3 remontage_fix_pauses.py --locales RU GE
   python3 remontage_fix_pauses.py --base-folder "/путь/к/проекту"

Скрипт ожидает, что лежит в одной папке с video_creator_times_autovenv_no_subs_realistic.py
(нужно только для --assemble).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# =========================
# КОНФИГ
# =========================

SCRIPTS_DIR = Path(__file__).resolve().parent

BASE_DIR = Path(os.getenv("BASE_FOLDER", "/Users/aleksandrtomilov/Desktop/ПТИЦЫ РЕАЛИСТИЧНЫЕ"))

LOCALES = ["RU", "GE", "PL", "ES"]

# Файлы таймингов, которые читает финальный сборщик.
TIMING_FILE_NAMES = {
    "RU": "image_times_ru.txt",
    "GE": "image_times_de.txt",
    "PL": "image_times_pl.txt",
    "ES": "image_times_es.txt",
}

# Языковые папки внутри ПРОМПТЫ (там лежат *_prompts.json / *_visual_blocks.json / *.srt).
LOCALE_PROMPT_FOLDERS = {
    "RU": "RU_русский",
    "GE": "DE_немецкий",
    "PL": "PL_польский",
    "ES": "ES_испанский",
}

ASSEMBLER_SCRIPT = "video_creator_times_autovenv_no_subs_realistic.py"

MIN_VIDEO_BYTES = 1024

TC = r"\d{1,2}:\d{2}:\d{2}[,.]\d{1,3}"
TIMING_LINE_RE = re.compile(
    rf"^\s*(?P<idx>\d+)\s*\|\s*(?P<start>{TC})\s*-->\s*(?P<end>{TC})"
    r"(?:\s*\|\s*duration\s*:\s*(?P<duration>\d+(?:[,.]\d+)?)\s*s?)?"
    r"(?:\s*\|\s*type\s*:\s*(?P<kind>[a-zа-я_\-]+))?"
    rf"(?:\s*\|\s*src\s*:\s*(?P<src_start>{TC})\s*-->\s*(?P<src_end>{TC}))?",
    re.IGNORECASE,
)

# Фраза завершена, если оканчивается . ! ? … (допустимы кавычки/скобки после знака).
SENTENCE_END_RE = re.compile(r"[.!?…]+[»\"'\)\]]*\s*$")


@dataclass
class Entry:
    index: int              # индекс в СТАРОМ файле таймингов (совпадает с префиксом mp4)
    kind: str               # "speech" | "broll"
    duration: float
    src_start: Optional[float]
    src_end: Optional[float]
    text: str = ""          # текст фразы (только speech)
    # заполняется при пересборке:
    new_index: int = 0
    new_start: float = 0.0
    new_end: float = 0.0


def log(msg: str) -> None:
    print(msg, flush=True)


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def phrase_is_complete(text: str) -> bool:
    return bool(SENTENCE_END_RE.search(clean_text(text)))


# =========================
# ПАРСИНГ
# =========================

def tc_to_seconds(value: str) -> float:
    text = value.strip().replace(",", ".")
    parts = text.split(":")
    if len(parts) == 3:
        return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
    if len(parts) == 2:
        return float(parts[0]) * 60 + float(parts[1])
    return float(text)


def seconds_to_tc(seconds: float) -> str:
    if seconds < 0:
        seconds = 0
    ms = int(round((seconds - int(seconds)) * 1000))
    whole = int(seconds)
    if ms == 1000:
        whole += 1
        ms = 0
    h, rem = divmod(whole, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def safe_tc(value: str) -> str:
    return value.replace(":", "-").replace(",", "-").replace(" ", "")


def parse_timing_file(path: Path) -> list[Entry]:
    entries: list[Entry] = []
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line:
            continue
        m = TIMING_LINE_RE.match(line)
        if not m:
            continue
        start = tc_to_seconds(m.group("start"))
        end = tc_to_seconds(m.group("end"))
        dur_text = m.group("duration")
        duration = float(dur_text.replace(",", ".")) if dur_text else max(0.0, end - start)
        kind_text = (m.group("kind") or "speech").strip().lower()
        kind = "broll" if kind_text in ("broll", "b-roll", "pause", "пауза") else "speech"
        if kind == "speech":
            src_start = tc_to_seconds(m.group("src_start")) if m.group("src_start") else start
            src_end = tc_to_seconds(m.group("src_end")) if m.group("src_end") else end
        else:
            src_start = src_end = None
        entries.append(Entry(
            index=int(m.group("idx")), kind=kind, duration=duration,
            src_start=src_start, src_end=src_end,
        ))
    return entries


# =========================
# ТЕКСТЫ ФРАЗ (три источника, по убыванию надёжности)
# =========================

def newest(paths: list[Path]) -> Optional[Path]:
    paths = [p for p in paths if p.exists()]
    return max(paths, key=lambda p: p.stat().st_mtime) if paths else None


def load_texts_from_prompts_json(language_dir: Path, entries: list[Entry]) -> bool:
    """Новый формат *_prompts.json: список TimelineEntry с index/kind/text."""
    f = newest(list(language_dir.glob("*_prompts.json")))
    if f is None:
        return False
    try:
        data = json.loads(f.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not isinstance(data, list) or not data:
        return False
    by_index = {}
    for item in data:
        if isinstance(item, dict) and "index" in item and "kind" in item:
            by_index[int(item["index"])] = item
    if not by_index:
        return False
    filled = 0
    for e in entries:
        item = by_index.get(e.index)
        if item and e.kind == "speech":
            e.text = clean_text(str(item.get("text", "")))
            if e.text:
                filled += 1
    if filled:
        log(f"    тексты фраз: {f.name} ({filled} шт.)")
        return filled > 0
    return False


def load_texts_from_visual_blocks(language_dir: Path, entries: list[Entry]) -> bool:
    """*_visual_blocks.json: только речевые блоки, по порядку."""
    f = newest(list(language_dir.glob("*_visual_blocks.json")))
    if f is None:
        return False
    try:
        blocks = json.loads(f.read_text(encoding="utf-8"))
    except Exception:
        return False
    speech_entries = [e for e in entries if e.kind == "speech"]
    if not isinstance(blocks, list) or len(blocks) != len(speech_entries):
        return False
    for e, b in zip(speech_entries, blocks):
        e.text = clean_text(str(b.get("text", "")))
    log(f"    тексты фраз: {f.name} ({len(speech_entries)} шт., по порядку)")
    return True


def load_texts_from_srt(language_dir: Path, entries: list[Entry]) -> bool:
    """*.srt: собираем текст фразы из сегментов, чья середина попадает в src-диапазон."""
    f = newest(list(language_dir.glob("*.srt")))
    if f is None:
        return False
    seg_re = re.compile(rf"({TC})\s*-->\s*({TC})")
    segments: list[tuple[float, float, str]] = []
    lines = f.read_text(encoding="utf-8-sig").splitlines()
    i = 0
    while i < len(lines):
        m = seg_re.search(lines[i])
        if m:
            start, end = tc_to_seconds(m.group(1)), tc_to_seconds(m.group(2))
            i += 1
            texts = []
            while i < len(lines) and lines[i].strip() and not seg_re.search(lines[i]):
                if not lines[i].strip().isdigit():
                    texts.append(lines[i].strip())
                i += 1
            segments.append((start, end, " ".join(texts)))
        else:
            i += 1
    if not segments:
        return False
    filled = 0
    for e in entries:
        if e.kind != "speech" or e.src_start is None:
            continue
        parts = [t for s, en, t in segments if e.src_start - 0.2 <= (s + en) / 2 <= (e.src_end or 0) + 0.2]
        e.text = clean_text(" ".join(parts))
        if e.text:
            filled += 1
    if filled:
        log(f"    тексты фраз: {f.name} ({filled} шт., по SRT)")
    return filled > 0


def load_texts(language_dir: Path, entries: list[Entry]) -> bool:
    return (
        load_texts_from_prompts_json(language_dir, entries)
        or load_texts_from_visual_blocks(language_dir, entries)
        or load_texts_from_srt(language_dir, entries)
    )


# =========================
# ПЕРЕМОНТАЖ: перестановка пауз к границам предложений
# =========================

def reorder_entries(entries: list[Entry]) -> tuple[list[Entry], list[str]]:
    """Возвращает (новый порядок, отчёт о перемещениях).

    Правило: перебивка, стоявшая после ОБОРВАННОЙ фразы, откладывается и вставляется
    после ближайшей следующей ЗАВЕРШЁННОЙ фразы. Перебивки после завершённых фраз
    остаются на месте. Отложенный хвост (если сценарий кончился оборванной фразой)
    уходит в самый конец — финальный кадр природы.
    """
    new_order: list[Entry] = []
    deferred: list[Entry] = []
    report: list[str] = []
    last_speech: Optional[Entry] = None

    for e in entries:
        if e.kind == "speech":
            new_order.append(e)
            last_speech = e
            if phrase_is_complete(e.text) and deferred:
                for d in deferred:
                    report.append(
                        f"пауза #{d.index:04d} передвинута: теперь после фразы #{e.index:04d} "
                        f"(«...{clean_text(e.text)[-40:]}»)"
                    )
                new_order.extend(deferred)
                deferred = []
        else:
            if last_speech is None or phrase_is_complete(last_speech.text):
                new_order.append(e)
            else:
                report.append(
                    f"пауза #{e.index:04d} снята с оборванной фразы #{last_speech.index:04d} "
                    f"(«...{clean_text(last_speech.text)[-40:]}») — будет перенесена вперёд"
                )
                deferred.append(e)

    if deferred:
        for d in deferred:
            report.append(f"пауза #{d.index:04d} ушла в самый конец ролика (финальный кадр)")
        new_order.extend(deferred)

    return new_order, report


def recompute_timeline(new_order: list[Entry]) -> None:
    cursor = 0.0
    for n, e in enumerate(new_order, 1):
        if e.kind == "speech" and e.src_start is not None and e.src_end is not None:
            dur = e.src_end - e.src_start
        else:
            dur = e.duration
        e.new_index = n
        e.new_start = cursor
        e.new_end = cursor + dur
        e.duration = dur
        cursor += dur


def timing_line(e: Entry) -> str:
    base = (
        f"{e.new_index:03d} | {seconds_to_tc(e.new_start)} --> {seconds_to_tc(e.new_end)} "
        f"| duration: {e.duration:.2f}s | type: {e.kind}"
    )
    if e.kind == "speech" and e.src_start is not None:
        base += f" | src: {seconds_to_tc(e.src_start)} --> {seconds_to_tc(e.src_end)}"
    return base


# =========================
# ВИДЕОФАЙЛЫ: поиск и переименование под новый порядок
# =========================

def find_video_by_index(visual_dir: Path, index: int) -> Optional[Path]:
    candidates = sorted(visual_dir.glob(f"{index:04d}_*.mp4")) or sorted(visual_dir.glob(f"{index:04d}*.mp4"))
    for p in candidates:
        try:
            if p.stat().st_size >= MIN_VIDEO_BYTES:
                return p
        except OSError:
            continue
    return None


def new_video_name(e: Entry) -> str:
    return f"{e.new_index:04d}_{safe_tc(seconds_to_tc(e.new_start))}__{safe_tc(seconds_to_tc(e.new_end))}.mp4"


def rename_videos(visual_dir: Path, new_order: list[Entry], dry_run: bool) -> None:
    """Двухфазное переименование: сначала все во временные имена, потом в финальные.
    Манифест переименований сохраняется рядом (на случай отката)."""
    plan: list[tuple[Path, str]] = []
    missing: list[int] = []
    for e in new_order:
        old = find_video_by_index(visual_dir, e.index)
        if old is None:
            missing.append(e.index)
            continue
        plan.append((old, new_video_name(e)))

    if missing:
        raise RuntimeError(
            f"В {visual_dir} не найдены видео для позиций: "
            + ", ".join(f"{i:04d}" for i in missing)
            + ". Сначала добей пропуски ремонтником, потом запусти перемонтаж."
        )

    if dry_run:
        for old, new in plan:
            if old.name != new:
                log(f"    [DRY-RUN] {old.name} -> {new}")
        return

    manifest = {old.name: new for old, new in plan}
    stamp = time.strftime("%Y%m%d_%H%M%S")
    (visual_dir / f"_remontage_manifest_{stamp}.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    tmp_pairs: list[tuple[Path, str]] = []
    for k, (old, new) in enumerate(plan):
        tmp = visual_dir / f"__remontage_tmp_{k:04d}.mp4"
        old.rename(tmp)
        tmp_pairs.append((tmp, new))
    renamed = 0
    for tmp, new in tmp_pairs:
        target = visual_dir / new
        tmp.rename(target)
        renamed += 1
    log(f"    переименовано видео: {renamed}")


# =========================
# ОБРАБОТКА ЯЗЫКА
# =========================

def process_locale(locale: str, prompts_root: Path, visual_root: Path, dry_run: bool) -> bool:
    """Возвращает True, если перемонтаж применён (или в dry-run показан)."""
    log(f"\n===== {locale} =====")
    timing_path = prompts_root / TIMING_FILE_NAMES[locale]
    if not timing_path.exists():
        log(f"  [SKIP] нет файла таймингов: {timing_path}")
        return False

    entries = parse_timing_file(timing_path)
    if not entries:
        log(f"  [SKIP] файл таймингов пустой/не распознан: {timing_path.name}")
        return False

    broll_n = sum(1 for e in entries if e.kind == "broll")
    if broll_n == 0:
        log("  [SKIP] в таймингах нет пауз-перебивок — чинить нечего")
        return False

    language_dir = prompts_root / LOCALE_PROMPT_FOLDERS[locale]
    if not load_texts(language_dir, entries):
        log(f"  [ERROR] не нашёл тексты фраз ни в *_prompts.json, ни в *_visual_blocks.json, "
            f"ни в *.srt внутри {language_dir}. Без текста не определить границы предложений.")
        return False

    no_text = [e.index for e in entries if e.kind == "speech" and not e.text]
    if no_text:
        log(f"  ⚠️ без текста {len(no_text)} фраз(ы) — считаю их завершёнными (паузы после них не двигаю): "
            + ", ".join(f"{i:04d}" for i in no_text[:10]))
        for e in entries:
            if e.kind == "speech" and not e.text:
                e.text = "…"  # трактуем как завершённую

    new_order, report = reorder_entries(entries)
    if not report:
        log("  ✅ все паузы уже стоят после завершённых предложений — менять нечего")
        return False

    log(f"  Перемещений: {len(report)}")
    for line in report:
        log(f"    - {line}")

    recompute_timeline(new_order)

    # Переименовываем видео под новый порядок.
    visual_dir = visual_root / locale
    rename_videos(visual_dir, new_order, dry_run)

    # Перезаписываем файл таймингов (с бэкапом).
    new_content = "\n".join(timing_line(e) for e in new_order) + "\n"
    if dry_run:
        log(f"  [DRY-RUN] файл таймингов {timing_path.name} не изменён")
        return True

    backup = timing_path.with_suffix(timing_path.suffix + f".bak_{time.strftime('%Y%m%d_%H%M%S')}")
    backup.write_text(timing_path.read_text(encoding="utf-8"), encoding="utf-8")
    timing_path.write_text(new_content, encoding="utf-8")
    log(f"  Тайминги обновлены: {timing_path.name} (бэкап: {backup.name})")
    return True


def run_assembler(base_folder: Path) -> None:
    script = SCRIPTS_DIR / ASSEMBLER_SCRIPT
    if not script.exists():
        raise RuntimeError(f"Не найден сборщик рядом с перемонтажником: {script}")
    venv_python = base_folder / "venv" / "bin" / "python"
    python_bin = str(venv_python) if venv_python.exists() else sys.executable
    env = os.environ.copy()
    env["BASE_FOLDER"] = str(base_folder)
    log("\n===== ФИНАЛЬНАЯ СБОРКА =====")
    result = subprocess.run([python_bin, str(script)], env=env)
    if result.returncode != 0:
        raise RuntimeError(f"Сборщик завершился с ошибкой (код {result.returncode})")


# =========================
# MAIN
# =========================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Перемонтаж БЕЗ перегенерации: двигает паузы-перебивки к границам предложений "
                    "по уже существующему визуалу и правит тайминги/порядок видео."
    )
    parser.add_argument("--base-folder", default=str(BASE_DIR),
                        help="Корневая папка проекта (ПРОМПТЫ/ВИЗУАЛ/ОЗВУЧКА внутри).")
    parser.add_argument("--locales", nargs="*", default=LOCALES,
                        help="Какие языки обрабатывать (default: все).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Показать перемещения и переименования, ничего не меняя.")
    parser.add_argument("--assemble", action="store_true",
                        help="После перемонтажа сразу запустить финальную сборку роликов.")
    args = parser.parse_args()

    base_folder = Path(args.base_folder).expanduser()
    prompts_root = base_folder / "ПРОМПТЫ"
    visual_root = base_folder / "ВИЗУАЛ"

    if not prompts_root.exists() or not visual_root.exists():
        print(f"❌ Не найдены папки ПРОМПТЫ/ВИЗУАЛ в {base_folder}", file=sys.stderr)
        sys.exit(1)

    log("ПЕРЕМОНТАЖНИК: паузы -> к границам предложений (визуал не перегенерируется)")
    log(f"Проект: {base_folder}")
    if args.dry_run:
        log("Режим: DRY-RUN (ничего не меняется)")

    changed_any = False
    try:
        for locale in [loc.upper() for loc in args.locales]:
            if locale not in TIMING_FILE_NAMES:
                log(f"[SKIP] неизвестный язык: {locale}")
                continue
            if process_locale(locale, prompts_root, visual_root, args.dry_run):
                changed_any = True
    except RuntimeError as exc:
        print(f"\n❌ {exc}", file=sys.stderr)
        sys.exit(1)

    if args.dry_run:
        log("\nDRY-RUN завершён. Запусти без --dry-run, чтобы применить.")
        return

    if not changed_any:
        log("\nИзменений нет — всё уже смонтировано правильно.")
        return

    if args.assemble:
        try:
            run_assembler(base_folder)
        except RuntimeError as exc:
            print(f"\n❌ {exc}", file=sys.stderr)
            sys.exit(1)
        log("\nГОТОВО: перемонтаж применён и ролики пересобраны.")
    else:
        log("\nГОТОВО: тайминги и порядок видео исправлены.")
        log("Теперь запусти финальную сборку:")
        log(f"  python3 {ASSEMBLER_SCRIPT}")
        log("(или этот же скрипт с флагом --assemble)")


if __name__ == "__main__":
    main()
