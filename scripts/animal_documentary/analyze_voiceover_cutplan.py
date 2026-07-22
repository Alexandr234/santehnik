#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
АНАЛИЗАТОР ОЗВУЧКИ -> ЧЁТКИЙ ПЛАН МОНТАЖА (cutplan_<xx>.json).

Зачем: таймкоды Whisper неточные — если резать озвучку прямо по ним, разрез может
попасть в середину слова («обезь…яна»), а естественные паузы дыхания между фразами
теряются (озвучка «съёживается» на десятки секунд).

Этот скрипт анализирует РЕАЛЬНЫЙ звук каждой озвучки (ОЗВУЧКА/RU.mp3, PL.mp3, ...):
  1. Находит все интервалы настоящей тишины (ffmpeg silencedetect).
  2. Читает тайминги ПРОМПТЫ/image_times_<xx>.txt (type: speech/broll, src: ...).
  3. Планирует разрезы озвучки ТОЛЬКО в местах пауз-перебивок и ТОЛЬКО по реальной
     тишине рядом с границей фраз — слово физически не может быть разрезано.
  4. Между фразами без паузы озвучка не режется вовсе: все микропаузы дыхания
     сохраняются, ни одна секунда звука не теряется (первая фраза начинается с 0.0,
     последняя заканчивается концом файла).
  5. Подгоняет плановую длительность каждого видеосегмента под фактический кусок
     звука и пишет ЧЁТКИЙ ПЛАН в ПРОМПТЫ/cutplan_<xx>.json.

Монтажный скрипт (video_creator_times_autovenv_no_subs_realistic.py) сам находит
cutplan_<xx>.json и собирает ролик строго по нему. Если плана нет — сборщик делает
такой же анализ на лету; но отдельный анализатор удобен, чтобы ЗАРАНЕЕ увидеть отчёт
и проверить все разрезы.

Запуск:
   python3 analyze_voiceover_cutplan.py                  # все языки
   python3 analyze_voiceover_cutplan.py --locales RU PL
   python3 analyze_voiceover_cutplan.py --base-folder "/путь/к/проекту"
   python3 analyze_voiceover_cutplan.py --noise-db -30   # если озвучка с шумным фоном

После него просто запусти сборщик — он напишет
«Использую точный план монтажа: cutplan_ru.json».
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

BASE_DIR = Path(os.getenv("BASE_FOLDER", "/Users/aleksandrtomilov/Desktop/ПТИЦЫ РЕАЛИСТИЧНЫЕ"))

LOCALES = ["RU", "GE", "PL", "ES"]

TIMING_FILE_NAMES = {
    "RU": "image_times_ru.txt",
    "GE": "image_times_de.txt",
    "PL": "image_times_pl.txt",
    "ES": "image_times_es.txt",
}

CUTPLAN_FILE_NAMES = {
    "RU": "cutplan_ru.json",
    "GE": "cutplan_de.json",
    "PL": "cutplan_pl.json",
    "ES": "cutplan_es.json",
}

AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg"}

# Параметры поиска тишины (те же, что в сборщике).
SILENCE_NOISE_DB = -35.0
SILENCE_MIN_DUR = 0.12
CUT_SEARCH_WINDOW = 0.7

TC = r"\d{1,2}:\d{2}:\d{2}[,.]\d{1,3}"
TIMING_LINE_RE = re.compile(
    rf"^\s*(?P<idx>\d+)\s*\|\s*(?P<start>{TC})\s*-->\s*(?P<end>{TC})"
    r"(?:\s*\|\s*duration\s*:\s*(?P<duration>\d+(?:[,.]\d+)?)\s*s?)?"
    r"(?:\s*\|\s*type\s*:\s*(?P<kind>[a-zа-я_\-]+))?"
    rf"(?:\s*\|\s*src\s*:\s*(?P<src_start>{TC})\s*-->\s*(?P<src_end>{TC}))?",
    re.IGNORECASE,
)


@dataclass
class TimingEntry:
    index: int
    kind: str
    duration: float
    src_start: Optional[float]
    src_end: Optional[float]


def log(msg: str) -> None:
    print(msg, flush=True)


# =========================
# БАЗОВЫЕ УТИЛИТЫ
# =========================

def tc_to_seconds(value: str) -> float:
    text = value.strip().replace(",", ".")
    parts = text.split(":")
    if len(parts) == 3:
        return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
    if len(parts) == 2:
        return float(parts[0]) * 60 + float(parts[1])
    return float(text)


def ffprobe_duration(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe не смог прочитать {path}:\n{result.stderr}")
    return float(result.stdout.strip())


def natural_key(path: Path):
    name = path.name.lower()
    return [int(p) if p.isdigit() else p for p in re.split(r"(\d+)", name)]


def find_audio_for_language(audio_dir: Path, lang: str) -> Optional[Path]:
    if not audio_dir.exists():
        return None
    files = sorted(
        [p for p in audio_dir.iterdir() if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS],
        key=natural_key,
    )
    for p in files:
        if p.stem.upper() == lang:
            return p
    pattern = re.compile(rf"(^|[_\-\s]){re.escape(lang)}($|[_\-\s])", re.IGNORECASE)
    for p in files:
        if pattern.search(p.stem):
            return p
    return None


def parse_timing_file(path: Path) -> list[TimingEntry]:
    entries: list[TimingEntry] = []
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
        entries.append(TimingEntry(
            index=int(m.group("idx")), kind=kind, duration=duration,
            src_start=src_start, src_end=src_end,
        ))
    return entries


# =========================
# АНАЛИЗ ЗВУКА (та же логика, что в сборщике)
# =========================

def detect_silences(audio_path: Path, audio_duration: float, noise_db: float, min_dur: float) -> list[tuple[float, float]]:
    cmd = [
        "ffmpeg", "-hide_banner", "-nostats",
        "-i", str(audio_path),
        "-af", f"silencedetect=noise={noise_db}dB:d={min_dur}",
        "-f", "null", "-",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    text = result.stderr or ""
    silences: list[tuple[float, float]] = []
    pending: Optional[float] = None
    for m in re.finditer(r"silence_(start|end):\s*([0-9.]+)", text):
        kind, value = m.group(1), float(m.group(2))
        if kind == "start":
            pending = value
        elif pending is not None:
            silences.append((pending, value))
            pending = None
    if pending is not None:
        silences.append((pending, audio_duration))
    return silences


def choose_cut_point(gap_lo: float, gap_hi: float, silences: list[tuple[float, float]], prev_cut: float) -> tuple[float, bool]:
    nominal = (gap_lo + gap_hi) / 2.0
    window_lo = gap_lo - CUT_SEARCH_WINDOW
    window_hi = gap_hi + CUT_SEARCH_WINDOW

    best: Optional[float] = None
    best_dist = 0.0
    for s, e in silences:
        if e < window_lo or s > window_hi:
            continue
        lo = max(s, window_lo)
        hi = min(e, window_hi)
        if hi <= lo:
            continue
        point = (lo + hi) / 2.0
        dist = abs(point - nominal)
        if best is None or dist < best_dist:
            best, best_dist = point, dist

    if best is not None:
        return max(best, prev_cut + 0.05), True
    return max(nominal, prev_cut + 0.05), False


def plan_alignment(timings: list[TimingEntry], audio_duration: float, silences: list[tuple[float, float]]):
    """Возвращает (entry_durations, sequence, cuts_report). Логика идентична сборщику."""
    groups: list[tuple[str, list[TimingEntry]]] = []
    for t in timings:
        if groups and groups[-1][0] == t.kind:
            groups[-1][1].append(t)
        else:
            groups.append((t.kind, [t]))

    speech_groups = [g[1] for g in groups if g[0] == "speech"]
    if not speech_groups:
        raise RuntimeError("В таймингах нет ни одной речевой позиции.")

    cuts: list[dict] = []
    cut_points: list[float] = []
    prev_cut = 0.0
    for gi in range(len(speech_groups) - 1):
        a = speech_groups[gi][-1]
        b = speech_groups[gi + 1][0]
        gap_lo = a.src_end if a.src_end is not None else 0.0
        gap_hi = b.src_start if b.src_start is not None else gap_lo
        if gap_hi < gap_lo:
            gap_lo, gap_hi = gap_hi, gap_lo
        cut, snapped = choose_cut_point(gap_lo, gap_hi, silences, prev_cut)
        cut = min(cut, max(prev_cut + 0.05, audio_duration - 0.05))
        cut_points.append(cut)
        prev_cut = cut
        cuts.append({
            "after_phrase_index": a.index,
            "at": round(cut, 3),
            "nominal_gap": [round(gap_lo, 3), round(gap_hi, 3)],
            "snapped_to_silence": snapped,
        })

    ranges: list[tuple[float, float]] = []
    for gi in range(len(speech_groups)):
        start = 0.0 if gi == 0 else cut_points[gi - 1]
        end = audio_duration if gi == len(speech_groups) - 1 else cut_points[gi]
        ranges.append((start, end))

    entry_durations: list[float] = [0.0] * len(timings)
    pos_of = {id(t): i for i, t in enumerate(timings)}
    audio_bounds: dict[int, tuple[float, float]] = {}

    for ents, (run_start, run_end) in zip(speech_groups, ranges):
        bounds = [run_start]
        for e in ents[1:]:
            bounds.append(e.src_start if e.src_start is not None else run_start)
        bounds.append(run_end)
        for i in range(1, len(bounds)):
            bounds[i] = max(bounds[i], bounds[i - 1] + 0.05)
        for e, seg_start, seg_end in zip(ents, bounds, bounds[1:]):
            entry_durations[pos_of[id(e)]] = seg_end - seg_start
            audio_bounds[pos_of[id(e)]] = (seg_start, seg_end)

    for t in timings:
        if t.kind == "broll":
            entry_durations[pos_of[id(t)]] = t.duration

    sequence: list[list] = []
    run_index = 0
    for kind, ents in groups:
        if kind == "speech":
            sequence.append(["audio", round(ranges[run_index][0], 3), round(ranges[run_index][1], 3)])
            run_index += 1
        else:
            sequence.append(["silence", round(sum(e.duration for e in ents), 3)])

    return entry_durations, sequence, cuts, audio_bounds


# =========================
# ОБРАБОТКА ЯЗЫКА
# =========================

def process_locale(locale: str, base_folder: Path, noise_db: float, min_dur: float) -> bool:
    prompts_root = base_folder / "ПРОМПТЫ"
    audio_dir = base_folder / "ОЗВУЧКА"

    log(f"\n===== {locale} =====")

    timing_path = prompts_root / TIMING_FILE_NAMES[locale]
    if not timing_path.exists():
        log(f"  [SKIP] нет файла таймингов: {timing_path}")
        return False

    audio_path = find_audio_for_language(audio_dir, locale)
    if audio_path is None:
        log(f"  [SKIP] не найдена озвучка для {locale} в {audio_dir}")
        return False

    timings = parse_timing_file(timing_path)
    if not timings:
        log(f"  [SKIP] тайминги пустые/не распознаны: {timing_path.name}")
        return False

    audio_duration = ffprobe_duration(audio_path)
    speech_n = sum(1 for t in timings if t.kind == "speech")
    broll_n = len(timings) - speech_n
    log(f"  Озвучка: {audio_path.name} ({audio_duration:.2f}s)")
    log(f"  Позиции: {len(timings)} (фраз: {speech_n}, пауз: {broll_n})")

    silences = detect_silences(audio_path, audio_duration, noise_db, min_dur)
    log(f"  Интервалов тишины найдено: {len(silences)} (порог {noise_db}dB, мин. {min_dur}s)")

    entry_durations, sequence, cuts, audio_bounds = plan_alignment(timings, audio_duration, silences)

    snapped = sum(1 for c in cuts if c["snapped_to_silence"])
    log(f"  Разрезов озвучки: {len(cuts)}, по реальной тишине: {snapped}")
    for c in cuts:
        if not c["snapped_to_silence"]:
            log(f"    ⚠️ разрез после фразы #{c['after_phrase_index']:04d} на {c['at']}s: "
                f"тишина рядом не найдена — режу в середине зазора. "
                f"Если слышен обрыв, попробуй --noise-db -30")

    src_total = sum(
        (t.src_end - t.src_start)
        for t in timings if t.kind == "speech" and t.src_start is not None
    )
    gaps = audio_duration - src_total
    if gaps > 0.3:
        log(f"  Естественные паузы дыхания: ~{gaps:.1f}s — сохраняются полностью")

    total_timeline = sum(entry_durations)
    log(f"  Итоговая шкала ролика: {total_timeline:.2f}s "
        f"(озвучка {audio_duration:.2f}s + паузы {total_timeline - audio_duration:.2f}s)")

    plan = {
        "locale": locale,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "audio_file": audio_path.name,
        "audio_duration": round(audio_duration, 3),
        "timing_file": timing_path.name,
        "silence_noise_db": noise_db,
        "silence_min_dur": min_dur,
        "cuts": cuts,
        "sequence": sequence,
        "entries": [
            {
                "index": t.index,
                "kind": t.kind,
                "duration": round(entry_durations[i], 3),
                **(
                    {"audio_from": round(audio_bounds[i][0], 3), "audio_to": round(audio_bounds[i][1], 3)}
                    if i in audio_bounds else {}
                ),
            }
            for i, t in enumerate(timings)
        ],
    }

    out_path = prompts_root / CUTPLAN_FILE_NAMES[locale]
    out_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"  ✅ План монтажа записан: {out_path}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Анализ озвучек по реальной тишине -> точный план монтажа cutplan_<xx>.json для сборщика."
    )
    parser.add_argument("--base-folder", default=str(BASE_DIR),
                        help="Корневая папка проекта (ОЗВУЧКА и ПРОМПТЫ внутри).")
    parser.add_argument("--locales", nargs="*", default=LOCALES,
                        help="Какие языки анализировать (default: все).")
    parser.add_argument("--noise-db", type=float, default=SILENCE_NOISE_DB,
                        help=f"Порог тишины в дБ (default: {SILENCE_NOISE_DB}; для шумного фона попробуй -30).")
    parser.add_argument("--min-silence", type=float, default=SILENCE_MIN_DUR,
                        help=f"Минимальная длительность тишины, сек (default: {SILENCE_MIN_DUR}).")
    args = parser.parse_args()

    base_folder = Path(args.base_folder).expanduser()
    if not (base_folder / "ПРОМПТЫ").exists() or not (base_folder / "ОЗВУЧКА").exists():
        print(f"❌ Не найдены папки ОЗВУЧКА/ПРОМПТЫ в {base_folder}", file=sys.stderr)
        sys.exit(1)

    log("АНАЛИЗАТОР ОЗВУЧКИ: точный план монтажа по реальной тишине")
    log(f"Проект: {base_folder}")

    done = 0
    for locale in [loc.upper() for loc in args.locales]:
        if locale not in TIMING_FILE_NAMES:
            log(f"[SKIP] неизвестный язык: {locale}")
            continue
        try:
            if process_locale(locale, base_folder, args.noise_db, args.min_silence):
                done += 1
        except RuntimeError as exc:
            print(f"  ❌ {locale}: {exc}", file=sys.stderr)

    log(f"\nГотово планов: {done}. Теперь запусти сборщик — он подхватит cutplan_*.json автоматически.")


if __name__ == "__main__":
    main()
