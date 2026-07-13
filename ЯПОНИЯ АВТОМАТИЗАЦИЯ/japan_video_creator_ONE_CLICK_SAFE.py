#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ONE-CLICK SAFE semantic montage for ЯПОНИЯ АВТОМАТИЗАЦИЯ.

Версия для запуска одним нажатием:
- по умолчанию всегда пересобирает видео заново;
- не использует старый/битый mp4 как готовый результат;
- финальный файл появляется только после полной успешной сборки;
- старый mp4 перед сборкой уходит в backup, чтобы случайно не открыть битый файл;
- папка визуала выбирается автоматически по самым свежим картинкам среди RU/EN папок;
- клики остаются, но добавляются безопасно через единую WAV-дорожку, без amix=120.

Главная цель:
- не сдвигать картинки относительно озвучки;
- не использовать старую логику "первые 2 видео, потом картинки";
- не брать картинки просто по сортировке папки;
- строго сопоставлять блок image_times/prompts с картинкой того же индекса;
- если нужной картинки нет, держать предыдущую доступную картинку, чтобы монтаж не падал.

Что делает:
1) RU/EN озвучка из:
   /Users/aleksandrtomilov/Desktop/ЯПОНИЯ АВТОМАТИЗАЦИЯ/ОЗВУЧКА

2) Тайминги:
   RU -> /Users/aleksandrtomilov/Desktop/ЯПОНИЯ АВТОМАТИЗАЦИЯ/ПРОМПТЫ/ru_RU_image_times.txt
   EN -> /Users/aleksandrtomilov/Desktop/ЯПОНИЯ АВТОМАТИЗАЦИЯ/ПРОМПТЫ/en_EN_image_times.txt

3) Промпты для аудита:
   RU -> /Users/aleksandrtomilov/Desktop/ЯПОНИЯ АВТОМАТИЗАЦИЯ/ПРОМПТЫ/ru_RU_prompts.txt
   EN -> /Users/aleksandrtomilov/Desktop/ЯПОНИЯ АВТОМАТИЗАЦИЯ/ПРОМПТЫ/en_EN_prompts.txt

4) Визуал ищется в:
   RU_STRICT_SYNC
   RU_SCENE_LOCK
   RU_SCENE_LOCK_TEST
   RU_FIXED
   RU
   и аналогично для EN.

5) Видеоряд:
   - только картинки;
   - без первых двух видео;
   - без зума;
   - без переходов;
   - по абсолютным таймкодам;
   - паузы между фразами закрываются предыдущей картинкой.

Запуск:
   python3 history_video_creator_STRICT_SEMANTIC_SYNC_HOLD_PREVIOUS_RU_EN_SKIP_EXISTING.py
   python3 history_video_creator_STRICT_SEMANTIC_SYNC_HOLD_PREVIOUS_RU_EN_SKIP_EXISTING.py --lang ALL
   python3 history_video_creator_STRICT_SEMANTIC_SYNC_HOLD_PREVIOUS_RU_EN_SKIP_EXISTING.py --lang RU
   python3 history_video_creator_STRICT_SEMANTIC_SYNC_HOLD_PREVIOUS_RU_EN_SKIP_EXISTING.py --lang EN

Проверка без рендера:
   python3 history_video_creator_STRICT_SEMANTIC_SYNC_HOLD_PREVIOUS_RU_EN_SKIP_EXISTING.py --lang ALL --dry-run

Если папка картинок другая:
   python3 history_video_creator_STRICT_SEMANTIC_SYNC_HOLD_PREVIOUS_RU_EN_SKIP_EXISTING.py --lang RU --visual-dir "/path/to/RU_STRICT_SYNC"
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import re
import subprocess
import shlex
import sys
import tempfile
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Автозапуск через venv проекта, если он есть.
PROJECT_ROOT = Path("/Users/aleksandrtomilov/Desktop/ЯПОНИЯ АВТОМАТИЗАЦИЯ")
VENV_PYTHON = PROJECT_ROOT / "venv" / "bin" / "python"
if VENV_PYTHON.exists() and Path(sys.executable).resolve() != VENV_PYTHON.resolve():
    env = os.environ.copy()
    env["VIRTUAL_ENV"] = str(PROJECT_ROOT / "venv")
    env["PATH"] = str(PROJECT_ROOT / "venv" / "bin") + os.pathsep + env.get("PATH", "")
    os.execve(str(VENV_PYTHON), [str(VENV_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]], env)

try:
    import cv2
    import numpy as np
except ImportError:
    print("Ошибка: нужны numpy и opencv-python. Установи: pip install numpy opencv-python", file=sys.stderr)
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    print("Ошибка: нужна tqdm. Установи: pip install tqdm", file=sys.stderr)
    sys.exit(1)


AUDIO_DIR = PROJECT_ROOT / "ОЗВУЧКА"
CLICK_SOUND_PATH = PROJECT_ROOT / "МОНТАЖ" / "МЫШЬ.MP3"
CLICK_EVERY_SEGMENTS = 3
CLICK_VOLUME = 1.0  # оригинальная громкость клика; голос не трогаем
VISUAL_ROOT_DIR = PROJECT_ROOT / "ВИЗУАЛ"
PROMPTS_DIR = PROJECT_ROOT / "ПРОМПТЫ"
OUTPUT_VIDEO_DIR = PROJECT_ROOT / "ГОТОВЫЕ ВИДЕО"
OUTPUT_VIDEO_DIR.mkdir(parents=True, exist_ok=True)

# ONE-CLICK SAFE DEFAULTS
# Скрипт рассчитан на запуск без аргументов: просто нажал — он пересобрал заново.
ALWAYS_REBUILD_BY_DEFAULT = True
AUTO_PICK_NEWEST_VISUAL_DIR = True
BACKUP_OLD_OUTPUT_BEFORE_REBUILD = True
MIN_VALID_OUTPUT_SIZE_BYTES = 5 * 1024 * 1024
OUTPUT_BACKUP_DIR = OUTPUT_VIDEO_DIR / "_OLD_OR_BROKEN_BACKUPS"
OUTPUT_BACKUP_DIR.mkdir(parents=True, exist_ok=True)

FPS = 25
TARGET_WIDTH = 1920
TARGET_HEIGHT = 1080
VIDEO_CODEC = "h264_videotoolbox"
VIDEO_CODEC_FALLBACK = "libx264"
VIDEO_BITRATE = "5M"
PIXEL_FORMAT = "yuv420p"
AUDIO_CODEC = "aac"
AUDIO_BITRATE = "192k"

AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

VISUAL_FOLDERS_BY_LANG = {
    "RU": ["RU_STRICT_SYNC", "RU_SCENE_LOCK", "RU_SCENE_LOCK_TEST", "RU_FIXED", "RU"],
    "EN": ["EN_STRICT_SYNC", "EN_SCENE_LOCK", "EN_SCENE_LOCK_TEST", "EN_FIXED", "EN"],
}

TIMING_FILE_BY_LANG = {
    "RU": PROMPTS_DIR / "ru_RU_image_times.txt",
    "EN": PROMPTS_DIR / "en_EN_image_times.txt",
}

PROMPTS_FILE_BY_LANG = {
    "RU": PROMPTS_DIR / "ru_RU_prompts.txt",
    "EN": PROMPTS_DIR / "en_EN_prompts.txt",
}


@dataclass
class TimingEntry:
    index: int
    start: float
    end: float
    duration: float
    image_name: str
    raw_line: str


@dataclass
class Segment:
    index: int
    start: float
    end: float
    path: Path
    voice_text: str
    image_name_from_times: str
    resolved_by: str


@dataclass
class ClickEvent:
    group_number: int
    after_segment_index: int
    time: float


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def quote_cmd(cmd: list[str | Path]) -> str:
    """Красиво печатает команду, чтобы её можно было скопировать в Terminal."""
    return shlex.join(str(x) for x in cmd)


def run_subprocess_verbose(cmd: list[str | Path], title: str) -> None:
    """
    Запускает процесс и печатает весь вывод сразу в терминал.
    Это особенно важно для FFmpeg: раньше subprocess.run(..., capture_output=True)
    прятал весь вывод, и казалось, что скрипт завис.
    """
    log("=" * 70)
    log(f"▶️ {title}")
    log("Команда, которая запускается:")
    print(quote_cmd(cmd), flush=True)

    started = time.time()
    proc = subprocess.Popen(
        [str(x) for x in cmd],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    try:
        if proc.stdout is not None:
            for raw_line in proc.stdout:
                line = raw_line.rstrip("\n")
                if line:
                    print(f"[process] {line}", flush=True)
        code = proc.wait()
    except KeyboardInterrupt:
        log(f"⛔ {title}: остановлено пользователем, убиваю процесс...")
        proc.kill()
        raise

    elapsed = time.time() - started
    log(f"⏹️ {title}: exit code={code}, время={elapsed:.1f}s")

    if code != 0:
        raise RuntimeError(f"{title} завершился с ошибкой: exit code {code}")


def natural_key(path: Path | str):
    name = Path(path).name.lower()
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", name)]


def check_dependencies() -> None:
    for cmd_name in ["ffmpeg", "ffprobe"]:
        try:
            subprocess.run([cmd_name, "-version"], capture_output=True, check=True, timeout=5)
        except Exception:
            print(f"Критическая ошибка: {cmd_name} не найден.", file=sys.stderr)
            sys.exit(1)


def ffprobe_duration(file_path: Path) -> float:
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(file_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr)
    return float(result.stdout.strip())


def ffprobe_stream_duration(file_path: Path, stream_selector: str) -> Optional[float]:
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", stream_selector,
        "-show_entries", "stream=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(file_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return None
    text = result.stdout.strip()
    if not text or text.upper() == "N/A":
        return None
    try:
        return float(text.splitlines()[0])
    except Exception:
        return None


def validate_final_mp4(path: Path, expected_duration: float, lang: str) -> None:
    """
    Проверяет, что финальный mp4 не битый: есть файл, нормальный размер,
    длительность контейнера близка к ожидаемой, аудио и видео не обрываются на первой минуте.
    """
    if not path.exists():
        raise RuntimeError(f"{lang}: финальный mp4 не создан: {path}")
    size = path.stat().st_size
    if size < MIN_VALID_OUTPUT_SIZE_BYTES:
        raise RuntimeError(f"{lang}: финальный mp4 подозрительно маленький: {size} bytes")

    container_duration = ffprobe_duration(path)
    video_duration = ffprobe_stream_duration(path, "v:0")
    audio_duration = ffprobe_stream_duration(path, "a:0")

    log(
        f"{lang}: проверка финального mp4: size={size} bytes, "
        f"container={container_duration:.3f}s, "
        f"video={video_duration if video_duration is not None else 'N/A'}s, "
        f"audio={audio_duration if audio_duration is not None else 'N/A'}s"
    )

    # Контейнер обязан быть почти полной длины.
    if container_duration < expected_duration - 3:
        raise RuntimeError(
            f"{lang}: финальный mp4 короче ожидаемого: {container_duration:.3f}s вместо {expected_duration:.3f}s"
        )

    # Если stream duration доступен, проверяем отдельно аудио/видео.
    if video_duration is not None and video_duration < expected_duration - 3:
        raise RuntimeError(
            f"{lang}: видеопоток обрывается: {video_duration:.3f}s вместо {expected_duration:.3f}s"
        )
    if audio_duration is not None and audio_duration < expected_duration - 3:
        raise RuntimeError(
            f"{lang}: аудиопоток обрывается: {audio_duration:.3f}s вместо {expected_duration:.3f}s"
        )

    log(f"✅ {lang}: финальный mp4 прошёл проверку, файл не выглядит битым")


def backup_existing_output(output_path: Path, lang: str) -> Optional[Path]:
    """
    Перед новой сборкой убираем старый mp4 из финального имени.
    Так ты случайно не откроешь битый файл от прошлого зависшего запуска.
    """
    if not output_path.exists():
        return None
    if output_path.stat().st_size <= 0:
        try:
            output_path.unlink()
            log(f"{lang}: удалён пустой старый файл: {output_path.name}")
        except Exception as exc:
            log(f"⚠️ {lang}: не смог удалить пустой старый файл {output_path.name}: {exc}")
        return None

    ts = time.strftime("%Y%m%d_%H%M%S")
    backup_path = OUTPUT_BACKUP_DIR / f"{output_path.stem}__OLD_{ts}{output_path.suffix}"
    try:
        os.replace(output_path, backup_path)
        log(f"{lang}: старый финальный mp4 убран в backup: {backup_path}")
        return backup_path
    except Exception as exc:
        log(f"⚠️ {lang}: не смог перенести старый файл в backup: {exc}")
        return None


def cleanup_partial_outputs(output_path: Path, lang: str) -> None:
    """Удаляет хвосты от прошлых оборванных сборок."""
    patterns = [
        output_path.with_name(output_path.stem + "__BUILDING.mp4"),
        output_path.with_name(output_path.stem + "__BUILDING.mov"),
    ]
    for f in patterns:
        if f.exists():
            try:
                f.unlink()
                log(f"{lang}: удалён старый временный файл: {f.name}")
            except Exception as exc:
                log(f"⚠️ {lang}: не смог удалить временный файл {f.name}: {exc}")


def parse_float(value: Optional[str]) -> Optional[float]:
    if not value:
        return None
    m = re.search(r"\d+(?:[,.]\d+)?", value)
    return float(m.group(0).replace(",", ".")) if m else None


def parse_timecode_to_seconds(value: str) -> float:
    text = str(value).strip().replace(",", ".")
    if ":" not in text:
        return float(text)
    parts = text.split(":")
    if len(parts) == 3:
        h, m, s = float(parts[0]), float(parts[1]), float(parts[2])
        return h * 3600 + m * 60 + s
    if len(parts) == 2:
        m, s = float(parts[0]), float(parts[1])
        return m * 60 + s
    raise ValueError(f"Не понимаю таймкод: {value}")


def seconds_to_timecode(seconds: float) -> str:
    seconds = max(0, float(seconds))
    ms = int(round((seconds - int(seconds)) * 1000))
    whole = int(seconds)
    if ms == 1000:
        whole += 1
        ms = 0
    h = whole // 3600
    m = (whole % 3600) // 60
    s = whole % 60
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def safe_timecode(value: str) -> str:
    return (
        value.strip()
        .replace(":", "-")
        .replace(",", "-")
        .replace(".", "-")
        .replace(" ", "")
    )


def seconds_to_safe_timecode(seconds: float) -> str:
    return safe_timecode(seconds_to_timecode(seconds))


def find_audio(lang: str) -> Optional[Path]:
    """
    Берём актуальную озвучку.
    Если есть точный RU.mp3 / EN.mp3 — берём его.
    Если таких несколько по смыслу — берём самый свежий файл.
    """
    if not AUDIO_DIR.exists():
        return None
    files = [p for p in AUDIO_DIR.iterdir() if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS]
    if not files:
        return None

    exact = [p for p in files if p.stem.upper() == lang]
    if exact:
        chosen = max(exact, key=lambda p: p.stat().st_mtime)
        log(f"{lang}: выбрана озвучка по точному имени: {chosen.name} | modified={time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(chosen.stat().st_mtime))}")
        return chosen

    cues = {
        "RU": ["ru", "rus", "russian", "рус", "русский"],
        "EN": ["en", "eng", "english", "англ", "английский"],
    }.get(lang, [lang.lower()])
    matches = [p for p in files if any(c in p.stem.lower() for c in cues)]
    if matches:
        chosen = max(matches, key=lambda p: p.stat().st_mtime)
        log(f"{lang}: выбрана самая свежая подходящая озвучка: {chosen.name} | modified={time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(chosen.stat().st_mtime))}")
        return chosen
    return None


def newest_image_mtime(visual_dir: Path) -> float:
    images = list_images(visual_dir) if visual_dir.exists() else []
    if not images:
        return 0.0
    return max(p.stat().st_mtime for p in images)


def find_visual_dir(lang: str, override: Optional[str]) -> Optional[Path]:
    """
    В one-click режиме не берём первую попавшуюся старую папку.
    Среди стандартных папок языка выбираем ту, где самые свежие картинки.
    Это помогает не смонтировать новый звук со старым визуалом.
    """
    if override:
        p = Path(override).expanduser()
        if p.exists():
            log(f"{lang}: visual-dir задан вручную: {p}")
            return p
        return None

    candidates: list[tuple[float, int, Path]] = []
    for folder in VISUAL_FOLDERS_BY_LANG.get(lang, []):
        p = VISUAL_ROOT_DIR / folder
        if not p.exists():
            continue
        imgs = list_images(p)
        if not imgs:
            log(f"{lang}: папка визуала пустая, пропускаю: {p}")
            continue
        newest = max(img.stat().st_mtime for img in imgs)
        candidates.append((newest, len(imgs), p))

    if not candidates:
        return None

    if AUTO_PICK_NEWEST_VISUAL_DIR:
        candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
        log(f"{lang}: кандидаты папок визуала по свежести:")
        for newest, count, p in candidates:
            log(f"  - {p.name}: images={count}, newest={time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(newest))}")
        chosen = candidates[0][2]
        log(f"{lang}: выбрана самая свежая папка визуала: {chosen}")
        return chosen

    # fallback старого поведения
    for folder in VISUAL_FOLDERS_BY_LANG.get(lang, []):
        p = VISUAL_ROOT_DIR / folder
        if p.exists():
            return p
    return None


def list_images(visual_dir: Path) -> list[Path]:
    return sorted([p for p in visual_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS], key=natural_key)


def read_timing_entries(path: Path) -> list[TimingEntry]:
    line_re = re.compile(
        r"^\s*(?P<idx>\d+)\s*\|\s*"
        r"(?P<start>\d{1,2}:\d{2}:\d{2}[,.]\d{1,3}|\d{1,2}:\d{2}[,.]\d{1,3}|\d+(?:[,.]\d+)?)"
        r"\s*-->\s*"
        r"(?P<end>\d{1,2}:\d{2}:\d{2}[,.]\d{1,3}|\d{1,2}:\d{2}[,.]\d{1,3}|\d+(?:[,.]\d+)?)"
        r"(?:\s*\|\s*duration\s*:\s*(?P<duration>\d+(?:[,.]\d+)?)\s*s?)?",
        re.IGNORECASE,
    )
    image_re = re.compile(r"\|\s*image\s*:\s*(?P<image>[^|]+)", re.IGNORECASE)

    entries = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        m = line_re.search(line)
        if not m:
            log(f"⚠️ Не понял строку тайминга #{line_number}: {line}")
            continue
        idx = int(m.group("idx"))
        start = parse_timecode_to_seconds(m.group("start"))
        end = parse_timecode_to_seconds(m.group("end"))
        duration = parse_float(m.group("duration")) or max(0, end - start)
        im = image_re.search(line)
        image_name = im.group("image").strip().strip('"').strip("'") if im else ""
        entries.append(TimingEntry(idx, start, end, duration, image_name, line))
    if not entries:
        raise RuntimeError(f"Не прочитал тайминги: {path}")
    entries.sort(key=lambda x: x.index)
    return entries


def read_voice_texts_from_prompts(path: Path) -> dict[int, str]:
    if not path.exists():
        return {}
    out: dict[int, str] = {}
    current_index: Optional[int] = None
    timing_re = re.compile(r"^\s*(\d+)\s*\|")
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        m = timing_re.match(line)
        if m:
            current_index = int(m.group(1))
            continue
        if current_index is not None and line.upper().startswith("TEXT:"):
            out[current_index] = line.split(":", 1)[1].strip()
    return out


def resolve_image_for_entry(entry: TimingEntry, visual_dir: Path, images: list[Path], allow_loose: bool = False) -> tuple[Path, str]:
    by_name = {p.name.lower(): p for p in images}
    by_stem = {p.stem.lower(): p for p in images}

    # 1) exact image name from image_times if exists
    if entry.image_name:
        raw = entry.image_name.strip()
        candidate = Path(raw)
        if candidate.is_absolute() and candidate.exists():
            return candidate, "exact_absolute_image_field"
        rel = visual_dir / raw
        if rel.exists():
            return rel, "exact_relative_image_field"
        if Path(raw).name.lower() in by_name:
            return by_name[Path(raw).name.lower()], "exact_name_image_field"
        if Path(raw).stem.lower() in by_stem:
            return by_stem[Path(raw).stem.lower()], "exact_stem_image_field"

    # 2) exact generator pattern with index + exact timecodes
    patterns = [
        f"{entry.index:04d}_{seconds_to_safe_timecode(entry.start)}__{seconds_to_safe_timecode(entry.end)}.*",
        f"{entry.index:03d}_{seconds_to_safe_timecode(entry.start)}__{seconds_to_safe_timecode(entry.end)}.*",
        f"{entry.index:04d}_*.*",
        f"{entry.index:03d}_*.*",
        f"{entry.index:04d}.*",
        f"{entry.index:03d}.*",
        f"{entry.index}.*",
    ]
    matches = []
    for pat in patterns:
        cur = sorted([p for p in visual_dir.glob(pat) if p.suffix.lower() in IMAGE_EXTENSIONS], key=natural_key)
        if cur:
            matches = cur
            break

    if len(matches) == 1:
        return matches[0], f"strict_index_pattern:{patterns[0] if patterns else ''}"
    if len(matches) > 1:
        # Prefer exact timecode match if present.
        exact_time = [
            p for p in matches
            if p.name.startswith(f"{entry.index:04d}_{seconds_to_safe_timecode(entry.start)}__{seconds_to_safe_timecode(entry.end)}")
            or p.name.startswith(f"{entry.index:03d}_{seconds_to_safe_timecode(entry.start)}__{seconds_to_safe_timecode(entry.end)}")
        ]
        if len(exact_time) == 1:
            return exact_time[0], "exact_index_and_timecode"
        newest = max(matches, key=lambda p: p.stat().st_mtime)
        if allow_loose:
            return newest, "loose_newest_among_multiple_index_matches"
        raise RuntimeError(
            f"Для блока {entry.index:03d} найдено несколько картинок: {', '.join(p.name for p in matches[:8])}. "
            f"Очисти папку визуала или запусти с --allow-loose-image-match."
        )

    if allow_loose:
        # Last resort: leading number.
        prefix_re = re.compile(rf"^0*{entry.index}(?:\D|$)")
        loose = sorted([p for p in images if prefix_re.match(p.stem)], key=natural_key)
        if len(loose) == 1:
            return loose[0], "loose_leading_number"
        if len(loose) > 1:
            return max(loose, key=lambda p: p.stat().st_mtime), "loose_newest_leading_number"

    raise FileNotFoundError(f"Не нашёл картинку для блока {entry.index:03d}. image_times image='{entry.image_name}'")


def find_next_available_image(
    entries: list[TimingEntry],
    current_position: int,
    visual_dir: Path,
    images: list[Path],
    allow_loose: bool,
) -> Optional[tuple[Path, str, int]]:
    """
    Если не хватает первой картинки или нескольких подряд, ищем ближайшую следующую доступную.
    Это аварийный fallback, чтобы видео всё равно собиралось.
    """
    for j in range(current_position + 1, len(entries)):
        try:
            path, how = resolve_image_for_entry(entries[j], visual_dir, images, allow_loose=allow_loose)
            return path, how, entries[j].index
        except Exception:
            continue
    return None


def collect_segments(lang: str, visual_dir: Path, timing_file: Path, prompts_file: Path, audio_duration: float, allow_loose: bool) -> list[Segment]:
    entries = read_timing_entries(timing_file)
    images = list_images(visual_dir)
    texts = read_voice_texts_from_prompts(prompts_file)

    segments: list[Segment] = []
    previous_path: Optional[Path] = None
    previous_index: Optional[int] = None
    missing_count = 0

    for pos, e in enumerate(entries):
        try:
            path, how = resolve_image_for_entry(e, visual_dir, images, allow_loose=allow_loose)
            previous_path = path
            previous_index = e.index
        except Exception as exc:
            missing_count += 1

            if previous_path is not None:
                path = previous_path
                how = f"missing_image_hold_previous:index_{previous_index:03d}; original_error={exc}"
                log(f"⚠️ {lang}: нет картинки для блока {e.index:03d}; держу предыдущую {previous_path.name}")
            else:
                next_found = find_next_available_image(entries, pos, visual_dir, images, allow_loose=allow_loose)
                if next_found:
                    path, next_how, next_index = next_found
                    how = f"missing_first_image_use_next:index_{next_index:03d}; {next_how}; original_error={exc}"
                    previous_path = path
                    previous_index = next_index
                    log(f"⚠️ {lang}: нет первой картинки для блока {e.index:03d}; временно использую следующую {path.name}")
                else:
                    raise RuntimeError(
                        f"Нет картинки для блока {e.index:03d}, предыдущей картинки тоже нет, "
                        f"и следующую доступную найти не удалось. Исходная ошибка: {exc}"
                    )

        segments.append(
            Segment(
                index=e.index,
                start=e.start,
                end=e.end,
                path=path,
                voice_text=texts.get(e.index, ""),
                image_name_from_times=e.image_name,
                resolved_by=how,
            )
        )

    if missing_count:
        log(f"⚠️ {lang}: пропущенных картинок закрыто удержанием соседнего кадра: {missing_count}")

    if segments and segments[-1].end < audio_duration:
        # Keep last frame to audio end by extending end.
        last = segments[-1]
        segments[-1] = Segment(last.index, last.start, audio_duration, last.path, last.voice_text, last.image_name_from_times, last.resolved_by)

    return segments


def write_audit_manifest(lang: str, segments: list[Segment]) -> Path:
    out = OUTPUT_VIDEO_DIR / f"{lang}_strict_sync_hold_previous_manifest.csv"
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "index", "start", "end", "duration",
                "voice_text", "image_file", "image_path",
                "image_from_times", "resolved_by", "image_modified",
            ],
        )
        w.writeheader()
        for s in segments:
            w.writerow({
                "index": s.index,
                "start": seconds_to_timecode(s.start),
                "end": seconds_to_timecode(s.end),
                "duration": f"{s.end - s.start:.3f}",
                "voice_text": s.voice_text,
                "image_file": s.path.name,
                "image_path": str(s.path),
                "image_from_times": s.image_name_from_times,
                "resolved_by": s.resolved_by,
                "image_modified": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(s.path.stat().st_mtime)),
            })
    return out


def read_image_unicode(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Не удалось прочитать картинку: {path}")
    return img


def resize_cover(frame, target_w: int, target_h: int):
    src_h, src_w = frame.shape[:2]
    scale = max(target_w / src_w, target_h / src_h)
    new_w = max(target_w, int(math.ceil(src_w * scale)))
    new_h = max(target_h, int(math.ceil(src_h * scale)))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    x1 = max(0, (new_w - target_w) // 2)
    y1 = max(0, (new_h - target_h) // 2)
    return resized[y1:y1 + target_h, x1:x1 + target_w]


def write_frame(proc, frame, count: int) -> int:
    if count <= 0:
        return 0
    raw = frame.tobytes()
    for _ in range(count):
        proc.stdin.write(raw)
    return count


def render_visual_track(segments: list[Segment], output_path: Path, total_duration: float, codec: str) -> float:
    total_frames = max(1, int(round(total_duration * FPS)))
    log(f"🎬 Старт рендера временного видео без аудио: {output_path}")
    log(f"Параметры видеорендера: duration={total_duration:.3f}s, fps={FPS}, frames={total_frames}, codec={codec}, size={TARGET_WIDTH}x{TARGET_HEIGHT}")
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{TARGET_WIDTH}x{TARGET_HEIGHT}",
        "-r", str(FPS),
        "-i", "-",
        "-vf", f"format={PIXEL_FORMAT}",
        "-an",
        "-c:v", codec,
        "-b:v", VIDEO_BITRATE,
        "-pix_fmt", PIXEL_FORMAT,
        str(output_path),
    ]
    log("FFmpeg-команда видеорендера:")
    print(quote_cmd(cmd), flush=True)
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.stdin is None:
        raise RuntimeError("Не открылся pipe ffmpeg")

    cache = {}
    current_frame = 0
    prev_frame = None
    written = 0

    def get_frame(path: Path):
        if path not in cache:
            cache[path] = resize_cover(read_image_unicode(path), TARGET_WIDTH, TARGET_HEIGHT)
        return cache[path]

    try:
        with tqdm(total=total_frames, desc=f"Рендер {output_path.stem}", unit="frame") as pbar:
            for s in sorted(segments, key=lambda x: (x.start, x.index)):
                start_f = max(0, int(round(s.start * FPS)))
                end_f = min(total_frames, max(start_f + 1, int(round(s.end * FPS))))
                frame = get_frame(s.path)

                if start_f > current_frame:
                    hold = prev_frame if prev_frame is not None else frame
                    n = write_frame(proc, hold, start_f - current_frame)
                    current_frame += n
                    written += n
                    pbar.update(n)

                n = write_frame(proc, frame, max(0, end_f - current_frame))
                current_frame += n
                written += n
                pbar.update(n)
                prev_frame = frame

            if current_frame < total_frames:
                hold = prev_frame if prev_frame is not None else get_frame(segments[-1].path)
                n = write_frame(proc, hold, total_frames - current_frame)
                written += n
                pbar.update(n)

        proc.stdin.close()
        err = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
        code = proc.wait()
        if code != 0:
            raise RuntimeError(err)
    except Exception:
        proc.kill()
        raise

    log(f"✅ Временное видео без аудио готово: {output_path} | written_frames={written} | video_duration={written / FPS:.3f}s")
    return written / FPS


def render_with_fallback(segments: list[Segment], output_path: Path, total_duration: float) -> float:
    try:
        return render_visual_track(segments, output_path, total_duration, VIDEO_CODEC)
    except Exception as exc:
        log(f"⚠️ {VIDEO_CODEC} не сработал: {exc}")
        log(f"Пробую {VIDEO_CODEC_FALLBACK}")
        return render_visual_track(segments, output_path, total_duration, VIDEO_CODEC_FALLBACK)


def collect_random_click_events(
    segments: list[Segment],
    total_duration: float,
    every_segments: int = CLICK_EVERY_SEGMENTS,
    seed: Optional[int] = None,
) -> list[ClickEvent]:
    """
    Делает один клик на каждую группу из 3 визуальных кадров.

    Пример для группы из трех кадров:
    - КАДР клик КАДР КАДР   -> клик после 1-го кадра;
    - КАДР КАДР клик КАДР   -> клик после 2-го кадра;
    - КАДР КАДР КАДР клик   -> клик после 3-го кадра.

    Здесь "кадр" — это один Segment из image_times, а не 1/25 секунды видео.
    """
    if every_segments <= 0 or not segments:
        return []

    rng = random.Random(seed) if seed is not None else random.SystemRandom()
    events: list[ClickEvent] = []
    ordered = sorted(segments, key=lambda x: (x.start, x.index))

    for group_start in range(0, len(ordered), every_segments):
        group = ordered[group_start:group_start + every_segments]
        if not group:
            continue

        chosen_pos = rng.randrange(len(group))
        chosen_segment = group[chosen_pos]
        click_time = min(max(0.0, chosen_segment.end), max(0.0, total_duration - 0.001))

        # Не ставим клик ровно в самый конец, если последний сегмент упирается в конец аудио.
        if click_time >= total_duration:
            click_time = max(0.0, total_duration - 0.05)

        events.append(
            ClickEvent(
                group_number=(group_start // every_segments) + 1,
                after_segment_index=chosen_segment.index,
                time=click_time,
            )
        )

    return events


def write_click_manifest(lang: str, events: list[ClickEvent]) -> Path:
    out = OUTPUT_VIDEO_DIR / f"{lang}_random_mouse_clicks_manifest.csv"
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["group_number", "after_segment_index", "click_time", "click_time_seconds"],
        )
        w.writeheader()
        for e in events:
            w.writerow({
                "group_number": e.group_number,
                "after_segment_index": e.after_segment_index,
                "click_time": seconds_to_timecode(e.time),
                "click_time_seconds": f"{e.time:.3f}",
            })
    return out


def mux_audio_plain(visual_path: Path, audio_path: Path, output_path: Path, duration: float) -> None:
    log(f"🎧 Старт финальной склейки БЕЗ кликов: video={visual_path.name}, audio={audio_path.name}, out={output_path.name}")
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "info", "-stats", "-progress", "pipe:1", "-stats_period", "1",
        "-i", str(visual_path),
        "-i", str(audio_path),
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", AUDIO_CODEC,
        "-b:a", AUDIO_BITRATE,
        "-t", f"{duration:.3f}",
        "-movflags", "+faststart",
        str(output_path),
    ]
    run_subprocess_verbose(cmd, f"FFmpeg mux plain {output_path.name}")
    log(f"✅ Финальная склейка БЕЗ кликов завершена: {output_path}")



def read_pcm16_wav(path: Path) -> tuple[int, int, np.ndarray]:
    """
    Читает WAV PCM16 и возвращает: sample_rate, channels, samples[int16].
    """
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        sample_rate = wf.getframerate()
        sample_width = wf.getsampwidth()
        frames = wf.getnframes()
        raw = wf.readframes(frames)

    if sample_width != 2:
        raise RuntimeError(f"Ожидался WAV PCM16, но sample_width={sample_width}: {path}")

    samples = np.frombuffer(raw, dtype=np.int16)
    if channels > 1:
        samples = samples.reshape(-1, channels)
    else:
        samples = samples.reshape(-1, 1)
    return sample_rate, channels, samples


def write_pcm16_wav(path: Path, sample_rate: int, samples: np.ndarray) -> None:
    """
    Пишет samples[int16] в WAV PCM16.
    """
    if samples.dtype != np.int16:
        samples = samples.astype(np.int16)
    if samples.ndim == 1:
        samples = samples.reshape(-1, 1)
    channels = samples.shape[1]
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())


def decode_click_to_wav(click_sound_path: Path, decoded_wav_path: Path, sample_rate: int = 48000, channels: int = 2) -> None:
    """
    Один раз декодирует МЫШЬ.MP3 в короткий WAV PCM16.
    Так Python потом быстро раскладывает клики по таймлайну.
    """
    log(f"🔊 Декодирую файл клика в WAV PCM16: {click_sound_path.name} -> {decoded_wav_path.name}")
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "info", "-stats",
        "-i", str(click_sound_path),
        "-vn",
        "-ac", str(channels),
        "-ar", str(sample_rate),
        "-acodec", "pcm_s16le",
        str(decoded_wav_path),
    ]
    run_subprocess_verbose(cmd, f"FFmpeg decode click {click_sound_path.name}")


def build_click_track_wav(
    click_sound_path: Path,
    click_track_path: Path,
    click_delays_ms: list[int],
    duration: float,
    sample_rate: int = 48000,
    channels: int = 2,
) -> None:
    """
    Быстрый способ добавить много кликов:
    1) декодируем один короткий клик в WAV;
    2) в Python создаём одну длинную WAV-дорожку тишины;
    3) накладываем в неё все клики;
    4) потом FFmpeg смешивает только голос + ОДНУ дорожку кликов.

    Это заменяет старый тяжёлый вариант с asplit=119 / adelay=119 / amix=120.
    """
    decoded_click_path = click_track_path.with_name(click_track_path.stem + "__single_click_decoded.wav")
    decode_click_to_wav(click_sound_path, decoded_click_path, sample_rate=sample_rate, channels=channels)

    click_sr, click_channels, click_samples = read_pcm16_wav(decoded_click_path)
    if click_sr != sample_rate:
        raise RuntimeError(f"Неожиданный sample_rate клика: {click_sr}, ожидался {sample_rate}")
    if click_channels != channels:
        raise RuntimeError(f"Неожиданное число каналов клика: {click_channels}, ожидалось {channels}")

    if CLICK_VOLUME != 1.0:
        click_samples = np.clip(click_samples.astype(np.float32) * float(CLICK_VOLUME), -32768, 32767).astype(np.int16)

    total_samples = max(1, int(math.ceil(duration * sample_rate)))
    log(
        f"🧱 Создаю одну WAV-дорожку кликов: duration={duration:.3f}s, "
        f"sample_rate={sample_rate}, channels={channels}, samples={total_samples}, clicks={len(click_delays_ms)}"
    )
    log(f"Размер дорожки кликов в памяти примерно: {total_samples * channels * 2 / 1024 / 1024:.1f} MB")

    track = np.zeros((total_samples, channels), dtype=np.int16)
    click_len = click_samples.shape[0]
    placed = 0
    skipped = 0

    for delay_ms in click_delays_ms:
        start_sample = int(round(delay_ms * sample_rate / 1000.0))
        if start_sample >= total_samples:
            skipped += 1
            continue
        end_sample = min(total_samples, start_sample + click_len)
        part_len = end_sample - start_sample
        if part_len <= 0:
            skipped += 1
            continue

        # Складываем с защитой от клиппинга.
        mixed = track[start_sample:end_sample].astype(np.int32) + click_samples[:part_len].astype(np.int32)
        np.clip(mixed, -32768, 32767, out=mixed)
        track[start_sample:end_sample] = mixed.astype(np.int16)
        placed += 1

    write_pcm16_wav(click_track_path, sample_rate, track)
    size = click_track_path.stat().st_size if click_track_path.exists() else 0
    log(f"✅ WAV-дорожка кликов готова: {click_track_path} | placed={placed}, skipped={skipped}, size={size} bytes")



def decode_audio_to_pcm16_wav(
    input_audio_path: Path,
    output_wav_path: Path,
    duration: float,
    sample_rate: int = 48000,
    channels: int = 2,
) -> None:
    """
    Декодирует основную озвучку в WAV PCM16 с фиксированными параметрами.
    Это убирает проблему FFmpeg, когда MP3 mono/stereo меняет параметры в середине filter graph.
    """
    log(f"🎙️ Декодирую основную озвучку в WAV PCM16: {input_audio_path.name} -> {output_wav_path.name}")
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "info", "-stats",
        "-i", str(input_audio_path),
        "-vn",
        "-ac", str(channels),
        "-ar", str(sample_rate),
        "-t", f"{duration:.3f}",
        "-acodec", "pcm_s16le",
        str(output_wav_path),
    ]
    run_subprocess_verbose(cmd, f"FFmpeg decode voice {input_audio_path.name}")


def build_mixed_voice_clicks_wav(
    audio_path: Path,
    click_sound_path: Path,
    mixed_audio_path: Path,
    click_delays_ms: list[int],
    duration: float,
    work_dir: Path,
    sample_rate: int = 48000,
    channels: int = 2,
) -> None:
    """
    Самая надёжная схема:
    1) декодируем голос в WAV PCM16;
    2) декодируем один клик в WAV PCM16;
    3) прямо в Python накладываем клики на голос;
    4) получаем ОДНУ финальную WAV-дорожку.

    После этого FFmpeg ничего не смешивает через amix — он только кладёт готовый WAV под видео.
    Поэтому звук не должен пропадать после минуты и не должен зависать на filter_complex.
    """
    voice_wav = work_dir / "voice_decoded_48000_stereo.wav"
    click_wav = work_dir / "single_click_decoded_48000_stereo.wav"

    decode_audio_to_pcm16_wav(audio_path, voice_wav, duration, sample_rate=sample_rate, channels=channels)
    decode_click_to_wav(click_sound_path, click_wav, sample_rate=sample_rate, channels=channels)

    voice_sr, voice_channels, voice_samples = read_pcm16_wav(voice_wav)
    click_sr, click_channels, click_samples = read_pcm16_wav(click_wav)

    if voice_sr != sample_rate or voice_channels != channels:
        raise RuntimeError(f"Голос декодировался не так: sr={voice_sr}, channels={voice_channels}")
    if click_sr != sample_rate or click_channels != channels:
        raise RuntimeError(f"Клик декодировался не так: sr={click_sr}, channels={click_channels}")

    total_samples = max(1, int(math.ceil(duration * sample_rate)))

    # Выравниваем голос строго под длину видео/аудио.
    if voice_samples.shape[0] < total_samples:
        pad = np.zeros((total_samples - voice_samples.shape[0], channels), dtype=np.int16)
        voice_samples = np.vstack([voice_samples, pad])
    elif voice_samples.shape[0] > total_samples:
        voice_samples = voice_samples[:total_samples]

    log(
        f"🧩 Миксую голос + клики в Python: duration={duration:.3f}s, "
        f"sample_rate={sample_rate}, channels={channels}, total_samples={total_samples}, clicks={len(click_delays_ms)}"
    )
    log(f"Размер рабочей аудиодорожки в памяти примерно: {total_samples * channels * 2 / 1024 / 1024:.1f} MB")

    mixed = voice_samples.astype(np.int32, copy=True)
    click = click_samples.astype(np.int32)
    if CLICK_VOLUME != 1.0:
        click = np.clip(click.astype(np.float32) * float(CLICK_VOLUME), -32768, 32767).astype(np.int32)

    click_len = click.shape[0]
    placed = 0
    skipped = 0

    for delay_ms in click_delays_ms:
        start_sample = int(round(delay_ms * sample_rate / 1000.0))
        if start_sample >= total_samples:
            skipped += 1
            continue
        end_sample = min(total_samples, start_sample + click_len)
        part_len = end_sample - start_sample
        if part_len <= 0:
            skipped += 1
            continue
        mixed[start_sample:end_sample] += click[:part_len]
        placed += 1

    np.clip(mixed, -32768, 32767, out=mixed)
    write_pcm16_wav(mixed_audio_path, sample_rate, mixed.astype(np.int16))
    size = mixed_audio_path.stat().st_size if mixed_audio_path.exists() else 0
    log(f"✅ Готова единая WAV-дорожка голос+клики: {mixed_audio_path} | placed={placed}, skipped={skipped}, size={size} bytes")


def mux_video_with_ready_audio_atomic(
    visual_path: Path,
    ready_audio_path: Path,
    output_path: Path,
    duration: float,
    lang: str,
) -> None:
    """
    Финальный mux без filter_complex и без amix.
    Пишем сначала во временный mp4 в той же папке, и только после успеха заменяем финальный файл.
    Это защищает от битого файла, если процесс оборвётся на середине.
    """
    temp_output = output_path.with_name(output_path.stem + "__BUILDING.mp4")
    if temp_output.exists():
        try:
            temp_output.unlink()
        except Exception:
            pass

    log(f"🎞️ Финальный mux: video={visual_path.name}, ready_audio={ready_audio_path.name}, temp={temp_output.name}")
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "info", "-stats", "-progress", "pipe:1", "-stats_period", "1",
        "-i", str(visual_path),
        "-i", str(ready_audio_path),
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", AUDIO_CODEC,
        "-b:a", AUDIO_BITRATE,
        "-t", f"{duration:.3f}",
        "-movflags", "+faststart",
        str(temp_output),
    ]
    run_subprocess_verbose(cmd, f"FFmpeg final mux safe {output_path.name}")

    if not temp_output.exists() or temp_output.stat().st_size <= 0:
        raise RuntimeError(f"Финальный временный файл не создан или пустой: {temp_output}")

    validate_final_mp4(temp_output, duration, lang)
    os.replace(temp_output, output_path)
    log(f"✅ Финальный файл атомарно заменён: {output_path}")


def mux_audio_with_random_clicks(
    visual_path: Path,
    audio_path: Path,
    output_path: Path,
    duration: float,
    click_sound_path: Path,
    click_events: list[ClickEvent],
) -> None:
    log(f"🎧 Старт безопасной финальной склейки С кликами: video={visual_path.name}, audio={audio_path.name}, clicks={len(click_events)}, out={output_path.name}")
    if not click_events:
        mux_audio_plain(visual_path, audio_path, output_path, duration)
        return

    if not click_sound_path.exists():
        raise FileNotFoundError(f"Не найден файл клика мыши: {click_sound_path}")

    # Последний клик не ставим прямо в край дорожки.
    max_click_ms = max(0, int(math.floor(duration * 1000)) - 250)
    click_delays_ms = sorted({
        min(max_click_ms, max(0, int(round(e.time * 1000))))
        for e in click_events
        if 0 <= e.time < duration
    })
    log(f"Кликов после фильтрации дублей/границ: {len(click_delays_ms)}")
    if click_delays_ms:
        log(f"Первый клик: {click_delays_ms[0]} ms, последний клик: {click_delays_ms[-1]} ms")
        log(f"Первые 10 кликов ms: {click_delays_ms[:10]}")
        log(f"Последние 10 кликов ms: {click_delays_ms[-10:]}")

    if not click_delays_ms:
        mux_audio_plain(visual_path, audio_path, output_path, duration)
        return

    work_dir = visual_path.parent
    mixed_audio_wav = work_dir / f"{output_path.stem}__VOICE_PLUS_CLICKS.wav"
    build_mixed_voice_clicks_wav(
        audio_path=audio_path,
        click_sound_path=click_sound_path,
        mixed_audio_path=mixed_audio_wav,
        click_delays_ms=click_delays_ms,
        duration=duration,
        work_dir=work_dir,
        sample_rate=48000,
        channels=2,
    )
    mux_video_with_ready_audio_atomic(visual_path, mixed_audio_wav, output_path, duration, lang=output_path.stem.split("_", 1)[0])
    log(f"✅ Финальная склейка С кликами завершена безопасной схемой: {output_path}")

def build(
    lang: str,
    visual_dir_override: Optional[str],
    allow_loose: bool,
    dry_run: bool,
    force: bool = False,
    enable_clicks: bool = True,
    click_seed: Optional[int] = None,
) -> bool:
    out = OUTPUT_VIDEO_DIR / f"{lang}_video_strict_sync_hold_previous.mp4"

    # One-click логика: по умолчанию всегда пересобираем и не доверяем старому mp4.
    if ALWAYS_REBUILD_BY_DEFAULT:
        force = True

    cleanup_partial_outputs(out, lang)

    if out.exists() and out.stat().st_size > 0 and not force:
        log(f"⏭️ {lang}: видео уже собрано, пропускаю: {out}")
        return True

    backup_path = None
    if out.exists() and force and BACKUP_OLD_OUTPUT_BEFORE_REBUILD and not dry_run:
        backup_path = backup_existing_output(out, lang)

    audio = find_audio(lang)
    if not audio:
        log(f"⚠️ {lang}: не найдена озвучка, пропускаю: {AUDIO_DIR}")
        return False

    visual_dir = find_visual_dir(lang, visual_dir_override)
    if not visual_dir:
        log(f"⚠️ {lang}: не найдена папка визуала, пропускаю.")
        return False

    timing_file = TIMING_FILE_BY_LANG[lang]
    prompts_file = PROMPTS_FILE_BY_LANG[lang]

    if not timing_file.exists():
        log(f"⚠️ {lang}: не найден timing file, пропускаю: {timing_file}")
        return False

    audio_duration = ffprobe_duration(audio)
    log(f"{lang}: audio={audio.name}, duration={audio_duration:.2f}s")
    log(f"{lang}: visual_dir={visual_dir}")
    log(f"{lang}: timing_file={timing_file.name}")

    segments = collect_segments(lang, visual_dir, timing_file, prompts_file, audio_duration, allow_loose=allow_loose)
    manifest = write_audit_manifest(lang, segments)
    log(f"{lang}: audit manifest: {manifest}")
    log(f"{lang}: first image {segments[0].index:03d} -> {segments[0].path.name}")
    log(f"{lang}: last image {segments[-1].index:03d} -> {segments[-1].path.name}")

    click_events: list[ClickEvent] = []
    if enable_clicks:
        click_events = collect_random_click_events(segments, audio_duration, seed=click_seed)
        click_manifest = write_click_manifest(lang, click_events)
        log(f"{lang}: mouse clicks every {CLICK_EVERY_SEGMENTS} frames: {len(click_events)}")
        log(f"{lang}: mouse clicks manifest: {click_manifest}")

    if dry_run:
        log(f"{lang}: DRY RUN — видео не собиралось.")
        return True

    if force:
        log(f"♻️ {lang}: one-click режим — пересобираю заново, старый финальный файл не используется")

    with tempfile.TemporaryDirectory(prefix=f"strict_{lang}_") as td:
        log(f"Временная папка сборки: {td}")
        visual_no_audio = Path(td) / f"{lang}_visual_no_audio.mp4"
        log(f"{lang}: этап 1/2 — создаю временную видеодорожку без аудио")
        render_with_fallback(segments, visual_no_audio, audio_duration)
        log(f"{lang}: этап 1/2 завершён, временный файл существует={visual_no_audio.exists()}, размер={visual_no_audio.stat().st_size if visual_no_audio.exists() else 0} bytes")
        if enable_clicks:
            log(f"{lang}: этап 2/2 — склеиваю видео + озвучку + клики")
            mux_audio_with_random_clicks(visual_no_audio, audio, out, audio_duration, CLICK_SOUND_PATH, click_events)
        else:
            log(f"{lang}: этап 2/2 — склеиваю видео + озвучку без кликов")
            mux_audio_plain(visual_no_audio, audio, out, audio_duration)
        log(f"{lang}: этап 2/2 завершён, финальный файл существует={out.exists()}, размер={out.stat().st_size if out.exists() else 0} bytes")

    validate_final_mp4(out, audio_duration, lang)
    log(f"✅ {lang}: готово: {out}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lang",
        choices=["RU", "EN", "ALL"],
        default="ALL",
        help="Что собирать: RU, EN или ALL. По умолчанию ALL. При запуске без аргументов собирает ALL.",
    )
    parser.add_argument("--visual-dir", default=None, help="Явная папка с картинками. Используй только когда собираешь один язык.")
    parser.add_argument("--allow-loose-image-match", action="store_true", help="Разрешить мягкий подбор картинки по номеру, если строгий не нашёл.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-clicks", action="store_true", help="Собрать без рандомных кликов мыши.")
    parser.add_argument("--click-seed", type=int, default=None, help="Seed для повторяемой рандомной расстановки кликов.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Совместимость со старыми командами. В этой версии пересборка и так включена по умолчанию.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Редкий режим: не пересобирать, если готовый mp4 уже есть.",
    )
    args = parser.parse_args()

    check_dependencies()
    log("🚀 ONE-CLICK SAFE режим: старые mp4 не используются, сборка идёт заново, клики включены")

    if args.lang == "ALL" and args.visual_dir:
        log("⚠️ --visual-dir передан вместе с --lang ALL. Одна папка не может подходить сразу RU и EN.")
        log("   Для ALL лучше не указывать --visual-dir, чтобы скрипт сам взял RU_SCENE_LOCK и EN_SCENE_LOCK.")
        log("   Либо запускай отдельно: --lang RU --visual-dir ... и --lang EN --visual-dir ...")

    langs = ["RU", "EN"] if args.lang == "ALL" else [args.lang]

    ok_count = 0
    fail_count = 0

    for lang in langs:
        log("=" * 70)
        log(f"▶️ Обработка {lang}")
        ok = build(
            lang=lang,
            visual_dir_override=args.visual_dir if args.lang != "ALL" else None,
            allow_loose=args.allow_loose_image_match,
            dry_run=args.dry_run,
            force=(not args.skip_existing),
            enable_clicks=not args.no_clicks,
            click_seed=args.click_seed,
        )
        if ok:
            ok_count += 1
        else:
            fail_count += 1

    log("=" * 70)
    log(f"Итог: успешно/пропущено готовых: {ok_count}; не собрано из-за отсутствия файлов: {fail_count}")

    if ok_count == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
