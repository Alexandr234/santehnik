# video_creator_times_autovenv_no_subs_FIXED_NO_SHAKE.py
# -*- coding: utf-8 -*-
#
# ДОКУМЕНТАЛЬНАЯ ВЕРСИЯ. Два ключевых отличия от старой сборки:
#
# 1) ПАУЗЫ МЕЖДУ ФРАЗАМИ (b-roll перебивки).
#    Файлы таймингов image_times_*.txt теперь могут содержать расширенные строки:
#       001 | 00:00:00,000 --> 00:00:06,640 | duration: 6.64s | type: speech | src: 00:00:00,000 --> 00:00:06,640
#       002 | 00:00:06,640 --> 00:00:08,840 | duration: 2.20s | type: broll
#    type: speech — фраза диктора; src говорит, откуда вырезать эту фразу из ИСХОДНОЙ озвучки.
#    type: broll  — пауза-перебивка: на экране кадры животных, диктор МОЛЧИТ.
#    Скрипт сам режет озвучку и вставляет тишину на время перебивок,
#    так что финальный звук диктора точно совпадает с растянутой шкалой.
#    Старый формат (без type/src) полностью поддерживается: всё считается speech,
#    озвучка идёт сплошным куском, как раньше.
#
#    ВАЖНО (исправление разрезанных слов и потерянных секунд):
#    озвучка НЕ режется по таймкодам Whisper — они неточные и могут попасть
#    в середину слова («обезь…яна»). Вместо этого:
#      - разрезы делаются ТОЛЬКО там, где вставляется пауза-перебивка;
#      - точное место разреза ищется по РЕАЛЬНОЙ ТИШИНЕ в озвучке
#        (ffmpeg silencedetect) рядом с границей фраз;
#      - между фразами без паузы озвучка вообще не режется — все естественные
#        микропаузы дыхания сохраняются, ни одна секунда звука не теряется;
#      - длительности видеосегментов подгоняются под фактические куски озвучки,
#        поэтому картинка и голос не расходятся.
#
# 2) ЖИВОЙ ЗВУК ПРИРОДЫ ИЗ КЛИПОВ.
#    Раньше звук исходных видео выбрасывался. Теперь родной звук каждого клипа
#    (натуральные звуки природы, сгенерированные вместе с видео) сохраняется:
#    он собирается в отдельную ambient-дорожку по той же шкале, что и видеоряд
#    (с теми же кроссфейдами), автоматически ПРИГЛУШАЕТСЯ под голос диктора
#    (sidechain ducking) и звучит В ПОЛНЫЙ ГОЛОС в паузах-перебивках.
#    Отключить: KEEP_CLIP_AUDIO = False.

from __future__ import annotations

# ------------------------------------------------------------
# АВТОЗАПУСК ЧЕРЕЗ VENV
# Если скрипт запустили системным Python, он сам перезапустится
# через /Users/aleksandrtomilov/Desktop/ПТИЦЫ РЕАЛИСТИЧНЫЕ/venv/bin/python.
# ------------------------------------------------------------
import os as _bootstrap_os
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath

_PROJECT_ROOT_FOR_VENV = _BootstrapPath("/Users/aleksandrtomilov/Desktop/ПТИЦЫ РЕАЛИСТИЧНЫЕ")
_VENV_DIR = _PROJECT_ROOT_FOR_VENV / "venv"
_VENV_PYTHON = _VENV_DIR / "bin" / "python"

def _restart_inside_venv_if_needed() -> None:
    current_python = _BootstrapPath(_bootstrap_sys.executable).resolve()

    if not _VENV_PYTHON.exists():
        return

    try:
        venv_python_resolved = _VENV_PYTHON.resolve()
    except Exception:
        venv_python_resolved = _VENV_PYTHON

    if current_python == venv_python_resolved:
        return

    env = _bootstrap_os.environ.copy()
    env["VIRTUAL_ENV"] = str(_VENV_DIR)
    env["PATH"] = str(_VENV_DIR / "bin") + _bootstrap_os.pathsep + env.get("PATH", "")

    print(f"Автозапуск через venv: {_VENV_PYTHON}", flush=True)
    _bootstrap_os.execve(
        str(_VENV_PYTHON),
        [str(_VENV_PYTHON), str(_BootstrapPath(__file__).resolve()), *_bootstrap_sys.argv[1:]],
        env,
    )

_restart_inside_venv_if_needed()

# Чистим служебные имена, чтобы не мешали основной программе.
del _restart_inside_venv_if_needed

import csv
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

try:
    import cv2
    import numpy as np
except ImportError:
    print("Ошибка: нужны numpy и opencv-python.", file=sys.stderr)
    print("Установи: pip install numpy opencv-python", file=sys.stderr)
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    print("Ошибка: нужна библиотека tqdm.", file=sys.stderr)
    print("Установи: pip install tqdm", file=sys.stderr)
    sys.exit(1)

################################################
# 1. КОНФИГ
################################################

PROJECT_ROOT = Path("/Users/aleksandrtomilov/Desktop/ПТИЦЫ РЕАЛИСТИЧНЫЕ")

# Здесь лежат 4 озвучки: RU / GE / PL / ES.
# Поддерживаются mp3, wav, m4a, aac, flac, ogg.
AUDIO_DIR = PROJECT_ROOT / "ОЗВУЧКА"

# Здесь лежат папки RU / GE / PL / ES.
# В каждой папке: сначала 2 видео, потом картинки.
VISUAL_ROOT_DIR = PROJECT_ROOT / "ВИЗУАЛ"

# Здесь лежат файлы с таймингами картинок/визуала:
# image_times_ru.txt, image_times_de.txt, image_times_pl.txt, image_times_es.txt.
PROMPTS_DIR = PROJECT_ROOT / "ПРОМПТЫ"

# Куда сохранять готовые видео.
OUTPUT_VIDEO_DIR = PROJECT_ROOT / "ГОТОВЫЕ ВИДЕО"
OUTPUT_VIDEO_DIR.mkdir(parents=True, exist_ok=True)

# Языки обрабатываются строго в этом порядке.
LANGUAGES = ["RU", "GE", "PL", "ES"]

TIMING_FILE_BY_FOLDER = {
    "RU": PROMPTS_DIR / "image_times_ru.txt",
    "GE": PROMPTS_DIR / "image_times_de.txt",
    "PL": PROMPTS_DIR / "image_times_pl.txt",
    "ES": PROMPTS_DIR / "image_times_es.txt",
}

# Точный план монтажа от analyze_voiceover_cutplan.py (если есть — используется он,
# иначе сборщик сам анализирует озвучку по тишине).
CUTPLAN_FILE_BY_FOLDER = {
    "RU": PROMPTS_DIR / "cutplan_ru.json",
    "GE": PROMPTS_DIR / "cutplan_de.json",
    "PL": PROMPTS_DIR / "cutplan_pl.json",
    "ES": PROMPTS_DIR / "cutplan_es.json",
}

# Видео параметры.
TARGET_WIDTH = 1920
TARGET_HEIGHT = 1080
FPS = 25

# Зум.
BASE_ZOOM = 1.15          # всё видео сразу зумировано на 15%
IMAGE_EXTRA_ZOOM = 0.15   # каждая картинка дополнительно плавно приближается ещё на 15%

# Плавный переход между всеми фрагментами: видео -> видео -> картинка -> картинка ...
TRANSITION_DURATION = 0.50  # сек. 0.35–0.70 обычно выглядит хорошо

# Минимальная длительность картинки после автоматической подгонки под аудио.
MIN_IMAGE_DURATION = 0.75

# FFmpeg параметры.
VIDEO_CODEC = "h264_videotoolbox"  # macOS. Если будет ошибка, замени на "libx264".
VIDEO_BITRATE = "5M"
PIXEL_FORMAT = "yuv420p"
AUDIO_CODEC = "aac"
AUDIO_BITRATE = "192k"

# Плёночное зерно. Поставь 0, если не нужно.
FILM_GRAIN_STRENGTH = 0

# --- ЗВУК ПРИРОДЫ ИЗ КЛИПОВ (документальный звук) ---
# Родной звук сгенерированных клипов (натуральные звуки природы) сохраняется и
# подмешивается под озвучку. В паузах-перебивках (type: broll) он звучит в полный голос.
KEEP_CLIP_AUDIO = True
AMBIENT_VOLUME = 0.9          # базовая громкость природы (до автоприглушения под голос)
DUCK_THRESHOLD = 0.02         # с какого уровня голоса начинать приглушать природу
DUCK_RATIO = 10               # насколько сильно приглушать (больше = тише природа под голосом)
DUCK_ATTACK_MS = 150          # как быстро приглушается, мс
DUCK_RELEASE_MS = 700         # как быстро возвращается после фразы, мс
AUDIO_SAMPLE_RATE = 48000

# --- ПОИСК ТИШИНЫ ДЛЯ РАЗРЕЗОВ ОЗВУЧКИ ---
# Разрез под паузу-перебивку делается в ближайшей реальной тишине, а не по таймкоду
# Whisper, чтобы никогда не резать слово пополам.
SILENCE_NOISE_DB = -35.0      # что считать тишиной (дБ); для шумных озвучек попробуй -30
SILENCE_MIN_DUR = 0.12        # минимальная длительность тишины, сек
CUT_SEARCH_WINDOW = 0.7       # насколько далеко (сек) от границы фраз можно искать тишину

# Субтитры отключены. Скрипт не запускает Whisper и не вшивает SRT.

AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

################################################
# 2. МОДЕЛИ И УТИЛИТЫ
################################################

@dataclass
class Segment:
    kind: str  # "video" или "image"
    path: Path
    duration: float


@dataclass
class TimingEntry:
    index: int
    start: float
    end: float
    duration: float
    kind: str = "speech"                 # "speech" (фраза диктора) или "broll" (пауза-перебивка)
    src_start: Optional[float] = None    # положение фразы в ИСХОДНОЙ озвучке (только speech)
    src_end: Optional[float] = None


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def natural_key(path: Path | str):
    """Сортировка 1, 2, 10 вместо 1, 10, 2."""
    name = Path(path).name.lower()
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", name)]


def check_dependencies() -> None:
    print("--- Проверка зависимостей ---")

    for cmd_name in ["ffmpeg", "ffprobe"]:
        try:
            subprocess.run([cmd_name, "-version"], capture_output=True, check=True, timeout=5)
            print(f"{cmd_name}... ОК")
        except Exception:
            print(f"Критическая ошибка: {cmd_name} не найден.", file=sys.stderr)
            print("Установи FFmpeg и проверь, что ffmpeg/ffprobe доступны из терминала.", file=sys.stderr)
            sys.exit(1)


    print("-----------------------------\n")


def ffprobe_duration(file_path: Path) -> float:
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(file_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe не смог прочитать длительность: {file_path}\n{result.stderr}")

    try:
        return float(result.stdout.strip())
    except ValueError:
        raise RuntimeError(f"Некорректная длительность от ffprobe для файла: {file_path}")


def parse_float(value: str) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip().replace(",", ".")
    match = re.search(r"\d+(?:\.\d+)?", text)
    if not match:
        return None
    return float(match.group(0))


def find_audio_for_language(lang: str) -> Optional[Path]:
    if not AUDIO_DIR.exists():
        return None

    audio_files = sorted(
        [p for p in AUDIO_DIR.iterdir() if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS],
        key=natural_key,
    )

    # 1) Идеальный вариант: RU.mp3 / GE.mp3 / PL.mp3 / ES.mp3.
    for p in audio_files:
        if p.stem.upper() == lang:
            return p

    # 2) Более мягкий вариант: voice_RU.mp3, RU_voice.mp3 и т.п.
    pattern = re.compile(rf"(^|[_\-\s]){re.escape(lang)}($|[_\-\s])", re.IGNORECASE)
    for p in audio_files:
        if pattern.search(p.stem):
            return p

    return None


def list_video_files(visual_dir: Path) -> list[Path]:
    return sorted(
        [p for p in visual_dir.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS],
        key=natural_key,
    )


def list_image_files(visual_dir: Path) -> list[Path]:
    return sorted(
        [p for p in visual_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS],
        key=natural_key,
    )


def parse_timecode_to_seconds(value: str) -> float:
    """
    Понимает форматы:
      00:00:06,640
      00:00:06.640
      00:06,640
      6.640
    """
    text = str(value).strip()
    text = text.replace(",", ".")

    if ":" not in text:
        return float(text)

    parts = text.split(":")
    if len(parts) == 3:
        hours = float(parts[0])
        minutes = float(parts[1])
        seconds = float(parts[2])
        return hours * 3600 + minutes * 60 + seconds
    if len(parts) == 2:
        minutes = float(parts[0])
        seconds = float(parts[1])
        return minutes * 60 + seconds

    raise ValueError(f"Не понимаю таймкод: {value}")


def read_timing_entries(lang: str) -> list[TimingEntry]:
    """
    Читает файл вида:
      001 | 00:00:00,000 --> 00:00:06,640 | duration: 6.64s
      002 | 00:00:06,640 --> 00:00:15,360 | duration: 8.72s

    Также допускает строки без duration — длительность будет рассчитана как end - start.
    """
    timing_path = TIMING_FILE_BY_FOLDER.get(lang.upper())
    if timing_path is None:
        raise FileNotFoundError(f"Для языка {lang} не задан файл таймингов.")
    if not timing_path.exists():
        raise FileNotFoundError(f"Не найден файл таймингов: {timing_path}")

    tc = r"\d{1,2}:\d{2}:\d{2}[,.]\d{1,3}|\d{1,2}:\d{2}[,.]\d{1,3}|\d+(?:[,.]\d+)?"
    line_re = re.compile(
        r"^\s*(?P<idx>\d+)\s*\|\s*"
        rf"(?P<start>{tc})"
        r"\s*-->\s*"
        rf"(?P<end>{tc})"
        r"(?:\s*\|\s*duration\s*:\s*(?P<duration>\d+(?:[,.]\d+)?)\s*s?)?"
        r"(?:\s*\|\s*type\s*:\s*(?P<kind>[a-zа-я_\-]+))?"
        rf"(?:\s*\|\s*src\s*:\s*(?P<src_start>{tc})\s*-->\s*(?P<src_end>{tc}))?",
        re.IGNORECASE,
    )

    entries: list[TimingEntry] = []
    with open(timing_path, "r", encoding="utf-8-sig") as f:
        for line_number, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue

            match = line_re.search(line)
            if not match:
                log(f"⚠️ Не понял строку тайминга #{line_number}, пропускаю: {line}")
                continue

            idx = int(match.group("idx"))
            start = parse_timecode_to_seconds(match.group("start"))
            end = parse_timecode_to_seconds(match.group("end"))
            duration_from_range = max(0.0, end - start)

            duration_text = match.group("duration")
            duration_from_text = parse_float(duration_text) if duration_text else None
            duration = duration_from_text if duration_from_text and duration_from_text > 0 else duration_from_range

            if duration <= 0:
                log(f"⚠️ Нулевая/ошибочная длительность в строке #{line_number}, пропускаю: {line}")
                continue

            kind_text = (match.group("kind") or "speech").strip().lower()
            kind = "broll" if kind_text in ("broll", "b-roll", "pause", "пауза") else "speech"

            src_start_text = match.group("src_start")
            src_end_text = match.group("src_end")
            if kind == "speech":
                # Старый формат без src: фраза лежит в озвучке там же, где и на финальной шкале.
                src_start = parse_timecode_to_seconds(src_start_text) if src_start_text else start
                src_end = parse_timecode_to_seconds(src_end_text) if src_end_text else end
            else:
                src_start = None
                src_end = None

            entries.append(TimingEntry(
                index=idx, start=start, end=end, duration=duration,
                kind=kind, src_start=src_start, src_end=src_end,
            ))

    if not entries:
        raise ValueError(f"Не удалось прочитать тайминги из файла: {timing_path}")

    entries.sort(key=lambda x: x.index)
    speech_n = sum(1 for e in entries if e.kind == "speech")
    broll_n = len(entries) - speech_n
    log(f"Тайминги {lang}: {len(entries)} интервалов из {timing_path.name} "
        f"(фраз: {speech_n}, пауз-перебивок: {broll_n})")
    return entries


def resolve_image_path(visual_dir: Path, raw_name: str, images_by_name: dict[str, Path], images_by_stem: dict[str, Path]) -> Optional[Path]:
    raw_name = str(raw_name).strip().strip('"').strip("'")
    if not raw_name:
        return None

    direct = Path(raw_name)
    candidates = []
    if direct.is_absolute():
        candidates.append(direct)
    else:
        candidates.append(visual_dir / raw_name)

    for c in candidates:
        if c.exists() and c.suffix.lower() in IMAGE_EXTENSIONS:
            return c

    lower_name = Path(raw_name).name.lower()
    if lower_name in images_by_name:
        return images_by_name[lower_name]

    lower_stem = Path(raw_name).stem.lower()
    if lower_stem in images_by_stem:
        return images_by_stem[lower_stem]

    return None


def collect_segments_for_language(
    lang: str,
    visual_dir: Path,
    audio_duration: float,
    timings: list[TimingEntry],
    entry_durations: Optional[list[float]] = None,
) -> list[Segment]:
    """
    Собирает визуальную дорожку по таймингам из image_times_*.txt.

    Правило (ВЕСЬ РОЛИК ИЗ ВИДЕО):
      - каждому интервалу тайминга (и фразе, и паузе-перебивке) соответствует одно
        видео из папки языка в естественном порядке сортировки (0001..., 0002..., ...);
      - картинки больше не используются.

    Видео подгоняются под длительность интервала:
      - видео, если короче интервала, будет удерживать последний кадр;
      - если видео длиннее интервала, оно будет обрезано по нужной длительности.

    Чтобы плавные переходы не укорачивали весь ролик, каждому сегменту,
    кроме последнего, добавляется запас TRANSITION_DURATION. Сам переход
    затем "съедает" этот запас, и итоговая длина остаётся близкой к таймингам.
    """
    videos = list_video_files(visual_dir)

    if len(timings) < 1:
        raise ValueError(f"В файле таймингов для {lang} должен быть минимум 1 интервал.")

    required_videos = len(timings)
    if len(videos) < required_videos:
        raise FileNotFoundError(
            f"Для {lang} не хватает видео: нужно {required_videos}, найдено {len(videos)} в {visual_dir}"
        )

    if len(videos) > required_videos:
        log(f"ℹ️ {lang}: видео больше, чем таймингов. Лишние видео будут проигнорированы: {len(videos)} > {required_videos}")

    selected_paths: list[tuple[str, Path]] = [("video", vid) for vid in videos[:required_videos]]

    segments: list[Segment] = []
    for i, ((kind, path), timing) in enumerate(zip(selected_paths, timings)):
        # Плановая длительность из выравнивания по озвучке (включает естественные
        # микропаузы дыхания); фолбэк — номинальная длительность из файла таймингов.
        duration = entry_durations[i] if entry_durations else timing.duration
        if i < len(selected_paths) - 1 and TRANSITION_DURATION > 0:
            duration += TRANSITION_DURATION
        segments.append(Segment(kind=kind, path=path, duration=duration))

    return segments


################################################
# 3. РЕНДЕР КАДРОВ: ВИДЕО/КАРТИНКИ + ЗУМ + ПЕРЕХОДЫ
################################################


def read_image_unicode(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Не удалось прочитать картинку: {path}")
    return img


def resize_cover(frame, target_w: int, target_h: int):
    src_h, src_w = frame.shape[:2]
    if src_w <= 0 or src_h <= 0:
        raise RuntimeError("Пустой кадр")

    scale = max(target_w / src_w, target_h / src_h)
    new_w = max(target_w, int(math.ceil(src_w * scale)))
    new_h = max(target_h, int(math.ceil(src_h * scale)))

    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    x1 = max(0, (new_w - target_w) // 2)
    y1 = max(0, (new_h - target_h) // 2)
    cropped = resized[y1:y1 + target_h, x1:x1 + target_w]

    if cropped.shape[1] != target_w or cropped.shape[0] != target_h:
        cropped = cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    return cropped


def apply_center_zoom(frame, zoom: float):
    """
    Стабильный зум по центру без микродрожи.

    Не используем схему crop -> resize, потому что при плавном изменении
    zoom размеры crop_w/crop_h каждый кадр округляются до целых пикселей.
    Из-за этого центр кадра может прыгать на 1 пиксель.

    Этот вариант повторяет принцип из твоего первого скрипта:
    фиксированный центр + affine-трансформация cv2.warpAffine.
    """
    if zoom <= 1.0001:
        return frame

    h, w = frame.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), 0, float(zoom))
    return cv2.warpAffine(
        frame,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def iter_image_frames(segment: Segment, total_frames: int) -> Iterator[np.ndarray]:
    base = read_image_unicode(segment.path)
    base = resize_cover(base, TARGET_WIDTH, TARGET_HEIGHT)

    if total_frames <= 1:
        yield apply_center_zoom(base, BASE_ZOOM)
        return

    for i in range(total_frames):
        progress = i / max(1, total_frames - 1)
        zoom = BASE_ZOOM + IMAGE_EXTRA_ZOOM * progress
        yield apply_center_zoom(base, zoom)


def iter_video_frames(segment: Segment, total_frames: int) -> Iterator[np.ndarray]:
    cap = cv2.VideoCapture(str(segment.path))
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {segment.path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if not src_fps or src_fps <= 1:
        src_fps = FPS

    last_frame = None
    last_src_idx = -1

    try:
        for out_i in range(total_frames):
            target_time = out_i / FPS
            src_idx = int(round(target_time * src_fps))

            if src_idx != last_src_idx + 1:
                cap.set(cv2.CAP_PROP_POS_FRAMES, src_idx)

            ok, frame = cap.read()
            if not ok:
                if last_frame is None:
                    frame = np.zeros((TARGET_HEIGHT, TARGET_WIDTH, 3), dtype=np.uint8)
                else:
                    frame = last_frame.copy()
            else:
                last_frame = frame

            last_src_idx = src_idx
            frame = resize_cover(frame, TARGET_WIDTH, TARGET_HEIGHT)
            frame = apply_center_zoom(frame, BASE_ZOOM)
            yield frame
    finally:
        cap.release()


def iter_segment_frames(segment: Segment, total_frames: int) -> Iterator[np.ndarray]:
    if segment.kind == "image":
        yield from iter_image_frames(segment, total_frames)
    elif segment.kind == "video":
        yield from iter_video_frames(segment, total_frames)
    else:
        raise ValueError(f"Неизвестный тип сегмента: {segment.kind}")


def compute_frame_counts_and_transitions(segments: list[Segment]) -> tuple[list[int], list[int]]:
    frame_counts = [max(1, int(round(s.duration * FPS))) for s in segments]
    base_transition_frames = max(0, int(round(TRANSITION_DURATION * FPS)))

    transitions = []
    for i in range(len(segments) - 1):
        # Переход не должен съесть слишком короткий фрагмент.
        tf = min(base_transition_frames, frame_counts[i] // 3, frame_counts[i + 1] // 3)
        transitions.append(max(0, tf))

    return frame_counts, transitions


def render_visual_track(segments: list[Segment], output_path: Path) -> float:
    """
    Рендерит весь видеоряд в один mp4 без аудио:
    первые 2 видео + картинки по таймингам image_times_*.txt, плавные переходы и зум.
    """
    frame_counts, transitions = compute_frame_counts_and_transitions(segments)
    expected_frames = sum(frame_counts) - sum(transitions)
    expected_duration = expected_frames / FPS

    vf_filters = []
    if FILM_GRAIN_STRENGTH and FILM_GRAIN_STRENGTH > 0:
        vf_filters.append(f"noise=alls={FILM_GRAIN_STRENGTH}:allf=t+u")
    vf_filters.append(f"format={PIXEL_FORMAT}")
    vf_arg = ",".join(vf_filters)

    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{TARGET_WIDTH}x{TARGET_HEIGHT}",
        "-r", str(FPS),
        "-i", "-",
        "-vf", vf_arg,
        "-an",
        "-c:v", VIDEO_CODEC,
        "-b:v", VIDEO_BITRATE,
        "-pix_fmt", PIXEL_FORMAT,
        str(output_path),
    ]

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.stdin is None:
        raise RuntimeError("Не удалось открыть stdin для ffmpeg")

    prev_tail: list[np.ndarray] = []
    written_frames = 0

    try:
        with tqdm(total=expected_frames, desc="Рендер видеоряда", unit="frame") as pbar:
            for index, segment in enumerate(segments):
                total_frames = frame_counts[index]
                tf_before = transitions[index - 1] if index > 0 else 0
                tf_after = transitions[index] if index < len(segments) - 1 else 0

                frame_iter = iter_segment_frames(segment, total_frames)

                head_frames: list[np.ndarray] = []
                for _ in range(tf_before):
                    try:
                        head_frames.append(next(frame_iter))
                    except StopIteration:
                        break

                if index > 0 and prev_tail and head_frames:
                    blend_count = min(len(prev_tail), len(head_frames))
                    for j in range(blend_count):
                        alpha = (j + 1) / (blend_count + 1)
                        blended = cv2.addWeighted(prev_tail[j], 1.0 - alpha, head_frames[j], alpha, 0.0)
                        proc.stdin.write(blended.tobytes())
                        written_frames += 1
                        pbar.update(1)

                    # Если вдруг head длиннее tail, дописываем остаток без blend.
                    for frame in head_frames[blend_count:]:
                        proc.stdin.write(frame.tobytes())
                        written_frames += 1
                        pbar.update(1)

                elif index == 0:
                    # У первого сегмента нет входного перехода.
                    for frame in head_frames:
                        proc.stdin.write(frame.tobytes())
                        written_frames += 1
                        pbar.update(1)

                tail_buffer = deque(maxlen=tf_after if tf_after > 0 else 1)

                for frame in frame_iter:
                    if tf_after > 0:
                        if len(tail_buffer) == tf_after:
                            out_frame = tail_buffer.popleft()
                            proc.stdin.write(out_frame.tobytes())
                            written_frames += 1
                            pbar.update(1)
                        tail_buffer.append(frame)
                    else:
                        proc.stdin.write(frame.tobytes())
                        written_frames += 1
                        pbar.update(1)

                if tf_after > 0:
                    prev_tail = list(tail_buffer)
                else:
                    prev_tail = []

                if index == len(segments) - 1 and prev_tail:
                    for frame in prev_tail:
                        proc.stdin.write(frame.tobytes())
                        written_frames += 1
                        pbar.update(1)
                    prev_tail = []

        proc.stdin.close()
        stderr = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
        return_code = proc.wait()
        if return_code != 0:
            raise RuntimeError(f"FFmpeg завершился с ошибкой при рендере видеоряда:\n{stderr}")

    except BrokenPipeError:
        stderr = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
        proc.kill()
        raise RuntimeError(f"FFmpeg закрыл pipe во время рендера:\n{stderr}")
    except Exception:
        proc.kill()
        raise

    log(f"Видеоряд сохранён: {output_path}")
    log(f"Кадров записано: {written_frames}, длительность ≈ {expected_duration:.2f} сек.")
    return expected_duration

################################################
# 4. ЗВУК: ОЗВУЧКА С ПАУЗАМИ + ПРИРОДА ИЗ КЛИПОВ + ФИНАЛЬНАЯ СБОРКА
################################################


def has_audio_stream(file_path: Path) -> bool:
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "a",
        "-show_entries", "stream=codec_type",
        "-of", "csv=p=0",
        str(file_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0 and "audio" in (result.stdout or "")


def run_ffmpeg(cmd: list[str], what: str) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg ошибка ({what}):\n{result.stderr}")


def detect_silences(audio_path: Path, audio_duration: float) -> list[tuple[float, float]]:
    """Возвращает интервалы реальной тишины в озвучке (по ffmpeg silencedetect)."""
    cmd = [
        "ffmpeg", "-hide_banner", "-nostats",
        "-i", str(audio_path),
        "-af", f"silencedetect=noise={SILENCE_NOISE_DB}dB:d={SILENCE_MIN_DUR}",
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


def choose_cut_point(
    gap_lo: float,
    gap_hi: float,
    silences: list[tuple[float, float]],
    prev_cut: float,
) -> tuple[float, bool]:
    """Ищет безопасную точку разреза озвучки возле границы фраз [gap_lo..gap_hi].

    Точка обязана лежать ВНУТРИ реальной тишины (иначе можно разрезать слово,
    т.к. таймкоды Whisper неточные). Возвращает (точка, нашлась ли тишина).
    Если тишины рядом нет — фолбэк на середину номинального зазора.
    """
    nominal = (gap_lo + gap_hi) / 2.0
    window_lo = gap_lo - CUT_SEARCH_WINDOW
    window_hi = gap_hi + CUT_SEARCH_WINDOW

    best: Optional[float] = None
    best_dist: float = 0.0
    for s, e in silences:
        if e < window_lo or s > window_hi:
            continue
        # Середина пересечения тишины с окном поиска; точка остаётся внутри тишины.
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


def plan_voice_alignment(
    timings: list[TimingEntry],
    audio_duration: float,
    silences: list[tuple[float, float]],
) -> tuple[list[float], list[tuple]]:
    """Выравнивает шкалу по РЕАЛЬНОЙ озвучке. Ничего из озвучки не выбрасывается.

    Принципы:
      - озвучка режется ТОЛЬКО в местах пауз-перебивок, и только по реальной тишине;
      - подряд идущие фразы без паузы остаются ОДНИМ непрерывным куском звука
        (все естественные микропаузы дыхания сохраняются);
      - первая фраза начинается с 0.0, последняя заканчивается концом озвучки;
      - длительность каждого видеосегмента подгоняется под фактический кусок звука.

    Возвращает:
      entry_durations — плановая длительность каждой позиции таймлайна (по порядку);
      sequence        — план склейки голоса: ('audio', from, to) | ('silence', dur).
    """
    # Группируем подряд идущие записи одного типа: speech-раны и broll-группы.
    groups: list[tuple[str, list[TimingEntry]]] = []
    for t in timings:
        if groups and groups[-1][0] == t.kind:
            groups[-1][1].append(t)
        else:
            groups.append((t.kind, [t]))

    speech_groups = [g[1] for g in groups if g[0] == "speech"]
    if not speech_groups:
        raise ValueError("В таймингах нет ни одной речевой позиции.")

    # Точки разреза озвучки между соседними speech-ранами (там, где стоят паузы).
    cuts: list[float] = []
    prev_cut = 0.0
    snapped_count = 0
    for gi in range(len(speech_groups) - 1):
        a = speech_groups[gi][-1]       # последняя фраза рана
        b = speech_groups[gi + 1][0]    # первая фраза следующего рана
        gap_lo = a.src_end if a.src_end is not None else a.end
        gap_hi = b.src_start if b.src_start is not None else b.start
        if gap_hi < gap_lo:
            gap_lo, gap_hi = gap_hi, gap_lo
        cut, snapped = choose_cut_point(gap_lo, gap_hi, silences, prev_cut)
        # Разрез не может уйти далеко от границы фраз и обязан оставаться монотонным.
        cut = min(max(cut, gap_lo - CUT_SEARCH_WINDOW), gap_hi + CUT_SEARCH_WINDOW)
        cut = max(cut, prev_cut + 0.05)
        cut = min(cut, max(prev_cut + 0.05, audio_duration - 0.05))
        cuts.append(cut)
        prev_cut = cut
        if snapped:
            snapped_count += 1
        else:
            log(f"    ⚠️ разрез #{gi + 1} у {cut:.2f}s: тишина рядом не найдена, "
                f"режу в середине зазора фраз (попробуй SILENCE_NOISE_DB=-30, если слышен обрыв)")
    if cuts:
        log(f"Разрезы озвучки: {len(cuts)}, из них по реальной тишине: {snapped_count}")

    # Диапазон озвучки каждого speech-рана: от предыдущего разреза до следующего.
    ranges: list[tuple[float, float]] = []
    for gi in range(len(speech_groups)):
        start = 0.0 if gi == 0 else cuts[gi - 1]
        end = audio_duration if gi == len(speech_groups) - 1 else cuts[gi]
        ranges.append((start, end))

    # Плановые длительности позиций: внутри рана границы сегментов идут по src_start
    # следующих фраз (сдвиг картинки может попасть на слово — для ВИДЕО это нормально),
    # а весь ран целиком ТОЧНО равен своему куску озвучки.
    #
    # ЖЁСТКИЙ ИНВАРИАНТ: все границы зажимаются внутрь [run_start, run_end], поэтому
    # сумма длительностей сегментов рана == длине его куска озвучки, что бы ни лежало
    # в src (кривые/немонотонные значения не могут раздуть или сжать шкалу — иначе
    # видеоряд становится длиннее звука и в конце ролика пропадает звук).
    entry_durations: list[float] = [0.0] * len(timings)
    pos_of = {id(t): i for i, t in enumerate(timings)}

    for ents, (run_start, run_end) in zip(speech_groups, ranges):
        raw_bounds = [e.src_start if e.src_start is not None else run_start for e in ents[1:]]
        # Кривые src (вне рана / немонотонные) => равномерное деление рана.
        sane = all(run_start < b < run_end for b in raw_bounds) and all(
            b2 > b1 for b1, b2 in zip(raw_bounds, raw_bounds[1:])
        )
        if not sane and raw_bounds:
            log(f"    ⚠️ src-границы внутри рана {run_start:.2f}-{run_end:.2f}s кривые — делю ран поровну")
            step = (run_end - run_start) / len(ents)
            raw_bounds = [run_start + step * (k + 1) for k in range(len(ents) - 1)]

        bounds = [run_start] + raw_bounds + [run_end]
        for i in range(1, len(bounds) - 1):
            bounds[i] = min(max(bounds[i], bounds[i - 1] + 0.05), run_end - 0.05 * (len(bounds) - 1 - i))
        for e, seg_start, seg_end in zip(ents, bounds, bounds[1:]):
            entry_durations[pos_of[id(e)]] = max(0.05, seg_end - seg_start)

    for t in timings:
        if t.kind == "broll":
            entry_durations[pos_of[id(t)]] = t.duration

    # План склейки голоса: audio-раны и тишина пауз в порядке таймлайна.
    sequence: list[tuple] = []
    run_index = 0
    for kind, ents in groups:
        if kind == "speech":
            sequence.append(("audio", ranges[run_index][0], ranges[run_index][1]))
            run_index += 1
        else:
            sequence.append(("silence", sum(e.duration for e in ents)))

    natural_gaps = audio_duration - sum(
        (t.src_end - t.src_start)
        for t in timings
        if t.kind == "speech" and t.src_start is not None and t.src_end is not None
    )
    if natural_gaps > 0.3:
        log(f"Естественные паузы дыхания в озвучке: ~{natural_gaps:.1f}s — сохраняются полностью")

    # КОНТРОЛЬ ИНВАРИАНТА: шкала видео обязана совпадать со звуковой дорожкой.
    timeline_total = sum(entry_durations)
    voice_total = sum(
        (item[2] - item[1]) if item[0] == "audio" else item[1]
        for item in sequence
    )
    if abs(timeline_total - voice_total) > 0.5:
        log(f"⚠️ РАССИНХРОН ПЛАНА: видеошкала {timeline_total:.2f}s != звук {voice_total:.2f}s. "
            f"Проверь src-поля в файле таймингов (это не должно происходить — сообщи разработчику).")
    else:
        log(f"План монтажа согласован: видеошкала == звук == {timeline_total:.2f}s")

    return entry_durations, sequence


def load_cutplan(lang: str, timings: list[TimingEntry], audio_duration: float) -> Optional[tuple[list[float], list[tuple]]]:
    """Читает точный план монтажа от analyze_voiceover_cutplan.py, если он есть и актуален."""
    path = CUTPLAN_FILE_BY_FOLDER.get(lang.upper())
    if path is None or not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        entries = data.get("entries", [])
        sequence_raw = data.get("sequence", [])
        plan_audio = float(data.get("audio_duration", 0.0))
    except Exception as exc:
        log(f"⚠️ Не удалось прочитать cutplan {path.name}: {exc} — анализирую озвучку сам")
        return None

    if len(entries) != len(timings):
        log(f"⚠️ cutplan {path.name} устарел: позиций {len(entries)}, а в таймингах {len(timings)} "
            f"— анализирую озвучку сам")
        return None
    if abs(plan_audio - audio_duration) > 0.5:
        log(f"⚠️ cutplan {path.name} сделан для другой озвучки "
            f"({plan_audio:.2f}s vs {audio_duration:.2f}s) — анализирую озвучку сам")
        return None
    # Структура (порядок speech/broll) обязана совпадать: если после анализатора
    # запускался перемонтаж/пайплайн, план мог устареть при том же числе позиций.
    plan_kinds = [str(e.get("kind", "speech")) for e in entries]
    now_kinds = [t.kind for t in timings]
    if plan_kinds != now_kinds:
        log(f"⚠️ cutplan {path.name} не совпадает с текущим порядком speech/broll "
            f"(тайминги менялись после анализа) — анализирую озвучку сам. "
            f"Перезапусти analyze_voiceover_cutplan.py, чтобы обновить план.")
        return None

    durations = [float(e["duration"]) for e in entries]
    sequence: list[tuple] = []
    for item in sequence_raw:
        if item[0] == "audio":
            sequence.append(("audio", float(item[1]), float(item[2])))
        else:
            sequence.append(("silence", float(item[1])))
    log(f"Использую точный план монтажа: {path.name}")
    return durations, sequence


def build_voiceover_from_plan(audio_path: Path, sequence: list[tuple], out_wav: Path) -> None:
    """Склеивает дорожку диктора по плану: непрерывные куски озвучки + тишина пауз."""
    fmt = f"aresample={AUDIO_SAMPLE_RATE},aformat=sample_fmts=s16:channel_layouts=stereo"
    filter_parts: list[str] = []
    labels: list[str] = []

    for k, item in enumerate(sequence):
        lab = f"p{k}"
        if item[0] == "audio":
            _tag, a, b = item
            filter_parts.append(f"[0:a]atrim={a:.3f}:{b:.3f},asetpts=PTS-STARTPTS,{fmt}[{lab}]")
        else:
            filter_parts.append(
                f"aevalsrc=0:d={item[1]:.3f}:s={AUDIO_SAMPLE_RATE},"
                f"aformat=sample_fmts=s16:channel_layouts=stereo[{lab}]"
            )
        labels.append(f"[{lab}]")

    filter_parts.append("".join(labels) + f"concat=n={len(labels)}:v=0:a=1[vout]")

    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(audio_path),
        "-filter_complex", ";".join(filter_parts),
        "-map", "[vout]",
        "-c:a", "pcm_s16le",
        str(out_wav),
    ]
    audio_parts = sum(1 for item in sequence if item[0] == "audio")
    silence_parts = len(sequence) - audio_parts
    log(f"Озвучка: {audio_parts} непрерывных кусков голоса + {silence_parts} пауз")
    run_ffmpeg(cmd, "озвучка с паузами")


def build_ambient_track(segments: list[Segment], out_wav: Path) -> bool:
    """
    Собирает ambient-дорожку ПРИРОДЫ из родного звука сгенерированных клипов.

    Каждый клип занимает на шкале то же место, что и в видеоряде (учитываются те же
    кроссфейды, что и в кадрах: fade-in/fade-out на длину перехода), поэтому звук
    природы точно следует за картинкой. Возвращает False, если ни у одного клипа
    нет аудиодорожки (тогда финал собирается только с озвучкой, как раньше).
    """
    frame_counts, transitions = compute_frame_counts_and_transitions(segments)

    starts: list[float] = []
    cursor = 0.0
    for i in range(len(segments)):
        starts.append(cursor)
        tf = transitions[i] if i < len(transitions) else 0
        cursor += (frame_counts[i] - tf) / FPS
    total_duration = (sum(frame_counts) - sum(transitions)) / FPS

    with_audio = [i for i, seg in enumerate(segments) if has_audio_stream(seg.path)]
    if not with_audio:
        log("ℹ️ Ни у одного клипа нет аудиодорожки — ambient-дорожка природы пропущена.")
        return False
    if len(with_audio) < len(segments):
        log(f"ℹ️ Звук есть только у {len(with_audio)}/{len(segments)} клипов — остальные пойдут без природы.")

    fmt = f"aresample={AUDIO_SAMPLE_RATE},aformat=sample_fmts=s16:channel_layouts=stereo"
    cmd: list[str] = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
    for i in with_audio:
        cmd.extend(["-i", str(segments[i].path)])

    filter_parts: list[str] = []
    labels: list[str] = []
    for n, i in enumerate(with_audio):
        dur = frame_counts[i] / FPS
        fade_in = (transitions[i - 1] / FPS) if i > 0 and i - 1 < len(transitions) else 0.0
        fade_out = (transitions[i] / FPS) if i < len(transitions) else 0.0
        delay_ms = int(round(starts[i] * 1000))

        chain = f"[{n}:a]atrim=0:{dur:.3f},asetpts=PTS-STARTPTS,{fmt},apad=whole_dur={dur:.3f}"
        if fade_in > 0:
            chain += f",afade=t=in:st=0:d={fade_in:.3f}"
        if fade_out > 0:
            chain += f",afade=t=out:st={max(0.0, dur - fade_out):.3f}:d={fade_out:.3f}"
        chain += f",adelay={delay_ms}|{delay_ms}[a{n}]"
        filter_parts.append(chain)
        labels.append(f"[a{n}]")

    if len(labels) == 1:
        filter_parts.append(f"{labels[0]}atrim=0:{total_duration:.3f}[aout]")
    else:
        filter_parts.append(
            "".join(labels)
            + f"amix=inputs={len(labels)}:duration=longest:normalize=0,atrim=0:{total_duration:.3f}[aout]"
        )

    cmd.extend([
        "-filter_complex", ";".join(filter_parts),
        "-map", "[aout]",
        "-c:a", "pcm_s16le",
        str(out_wav),
    ])
    log(f"Природа: собираю ambient-дорожку из {len(with_audio)} клипов")
    run_ffmpeg(cmd, "ambient-дорожка природы")
    return True


def mux_final(
    visual_video_path: Path,
    voice_wav: Path,
    ambient_wav: Optional[Path],
    output_path: Path,
    total_duration: float,
) -> None:
    """
    Финальная сборка:
      - видеоряд из visual_video_path;
      - голос диктора (уже с паузами) из voice_wav;
      - если есть ambient_wav — природа подмешивается с автоприглушением под голос
        (sidechain ducking): под фразами тише, в паузах-перебивках в полный голос;
      - субтитры не генерируются и не вшиваются.
    """
    if ambient_wav is None:
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "warning",
            "-i", str(visual_video_path),
            "-i", str(voice_wav),
            "-map", "0:v:0",
            "-map", "1:a:0",
            # Страховка: если голос чуть короче видеоряда, добиваем тишиной до конца,
            # чтобы аудиопоток не обрывался раньше картинки.
            "-af", f"apad=whole_dur={total_duration:.3f}",
            "-c:v", "copy",
            "-c:a", AUDIO_CODEC,
            "-b:a", AUDIO_BITRATE,
            "-t", f"{total_duration:.3f}",
            "-movflags", "+faststart",
            str(output_path),
        ]
        log("Финальная сборка: только голос диктора (ambient недоступен)")
        run_ffmpeg(cmd, "финальная сборка")
        return

    # duration=longest + apad: даже при небольшом расхождении длин голоса/природы
    # звук гарантированно тянется до конца видеоряда.
    filter_complex = (
        f"[2:a]volume={AMBIENT_VOLUME:.3f}[amb0];"
        f"[1:a]asplit=2[vo1][vo2];"
        f"[amb0][vo1]sidechaincompress="
        f"threshold={DUCK_THRESHOLD}:ratio={DUCK_RATIO}:attack={DUCK_ATTACK_MS}:release={DUCK_RELEASE_MS}[ambduck];"
        f"[vo2][ambduck]amix=inputs=2:duration=longest:normalize=0,"
        f"apad=whole_dur={total_duration:.3f}[aout]"
    )
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "warning",
        "-i", str(visual_video_path),
        "-i", str(voice_wav),
        "-i", str(ambient_wav),
        "-filter_complex", filter_complex,
        "-map", "0:v:0",
        "-map", "[aout]",
        "-c:v", "copy",
        "-c:a", AUDIO_CODEC,
        "-b:a", AUDIO_BITRATE,
        "-t", f"{total_duration:.3f}",
        "-movflags", "+faststart",
        str(output_path),
    ]
    log("Финальная сборка: голос диктора + природа из клипов (ducking под голосом)")
    run_ffmpeg(cmd, "финальная сборка")


def verify_output_audio(output_path: Path, total_duration: float) -> None:
    """Самопроверка готового ролика: ищет длинные мёртвые зоны в звуке (>8s тишины).

    Обычные паузы-перебивки короче и заполнены природой, поэтому длинная тишина —
    признак проблемы (рассинхрон плана, оборванная дорожка). Печатает предупреждение
    с таймкодами, чтобы проблему было видно сразу в консоли, а не после просмотра.
    """
    cmd = [
        "ffmpeg", "-hide_banner", "-nostats",
        "-i", str(output_path),
        "-af", "silencedetect=noise=-60dB:d=8",
        "-f", "null", "-",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    text = result.stderr or ""
    problems: list[tuple[float, float]] = []
    pending: Optional[float] = None
    for m in re.finditer(r"silence_(start|end):\s*([0-9.]+)", text):
        kind, value = m.group(1), float(m.group(2))
        if kind == "start":
            pending = value
        elif pending is not None:
            problems.append((pending, value))
            pending = None
    if pending is not None and total_duration - pending > 2.0:
        problems.append((pending, total_duration))

    if problems:
        log("⚠️ ПРОВЕРКА ЗВУКА: в готовом ролике найдены длинные немые участки:")
        for a, b in problems:
            log(f"    тишина {a:.1f}s — {b:.1f}s (длина {b - a:.1f}s)")
        log("    Это признак рассинхрона. Проверь предупреждения выше и перезапусти "
            "analyze_voiceover_cutplan.py, затем сборку.")
    else:
        log("Проверка звука: длинных немых участков нет ✅")

################################################
# 5. ОДИН ЯЗЫКОВОЙ РОЛИК
################################################


def build_video_for_language(lang: str, audio_path: Path, visual_dir: Path) -> bool:
    start_time = time.time()
    output_path = OUTPUT_VIDEO_DIR / f"{lang}_video.mp4"

    log("\n======================")
    log(f"🎬 Старт ролика {lang}")
    log(f"  Озвучка: {audio_path}")
    log(f"  Визуал:  {visual_dir}")
    log(f"  Выход:   {output_path}")
    log("======================\n")

    try:
        audio_duration = ffprobe_duration(audio_path)
        log(f"Длительность озвучки: {audio_duration:.2f} сек.")

        timings = read_timing_entries(lang)

        # Выравнивание по РЕАЛЬНОЙ озвучке: готовый cutplan от анализатора,
        # либо собственный анализ тишины (разрезы только в тишине, звук не теряется).
        plan = load_cutplan(lang, timings, audio_duration)
        if plan is None:
            silences = detect_silences(audio_path, audio_duration)
            log(f"Найдено интервалов тишины в озвучке: {len(silences)}")
            entry_durations, sequence = plan_voice_alignment(timings, audio_duration, silences)
        else:
            entry_durations, sequence = plan

        segments = collect_segments_for_language(lang, visual_dir, audio_duration, timings, entry_durations)

        video_count = sum(1 for s in segments if s.kind == "video")
        image_count = sum(1 for s in segments if s.kind == "image")
        broll_count = sum(1 for t in timings if t.kind == "broll")
        log(f"Сегменты: {video_count} видео + {image_count} картинок (из них пауз-перебивок: {broll_count})")
        log(f"Первое видео: {segments[0].path.name} | всего видео: {video_count}")

        with tempfile.TemporaryDirectory(prefix=f"video_{lang}_") as td:
            temp_dir = Path(td)
            visual_track_path = temp_dir / f"{lang}_visual_no_audio.mp4"
            total_duration = render_visual_track(segments, visual_track_path)

            # 1) Голос диктора по плану: непрерывные куски озвучки + тишина в паузах.
            voice_wav = temp_dir / f"{lang}_voice.wav"
            build_voiceover_from_plan(audio_path, sequence, voice_wav)

            # 2) Природа: родной звук клипов, выровненный по видеоряду.
            ambient_wav: Optional[Path] = None
            if KEEP_CLIP_AUDIO:
                candidate = temp_dir / f"{lang}_ambient.wav"
                if build_ambient_track(segments, candidate):
                    ambient_wav = candidate

            # 3) Финал: голос + приглушаемая под голос природа.
            mux_final(
                visual_video_path=visual_track_path,
                voice_wav=voice_wav,
                ambient_wav=ambient_wav,
                output_path=output_path,
                total_duration=total_duration,
            )

            # 4) Самопроверка звука готового файла (мёртвые зоны видно сразу).
            verify_output_audio(output_path, total_duration)

        elapsed_min = (time.time() - start_time) / 60
        log(f"✅ Готово {lang}: {output_path}")
        log(f"Время: {elapsed_min:.2f} минут\n")
        return True

    except Exception as exc:
        log(f"❌ Ошибка в ролике {lang}: {exc}")
        return False

################################################
# 6. ГЛАВНАЯ: 4 РОЛИКА ПОДРЯД
################################################


def main() -> None:
    check_dependencies()

    if not AUDIO_DIR.exists():
        print(f"❌ Папка с озвучками не найдена: {AUDIO_DIR}", file=sys.stderr)
        return

    if not VISUAL_ROOT_DIR.exists():
        print(f"❌ Папка с визуалом не найдена: {VISUAL_ROOT_DIR}", file=sys.stderr)
        return

    if not PROMPTS_DIR.exists():
        print(f"❌ Папка с таймингами не найдена: {PROMPTS_DIR}", file=sys.stderr)
        return

    jobs = []
    for lang in LANGUAGES:
        audio_path = find_audio_for_language(lang)
        visual_dir = VISUAL_ROOT_DIR / lang

        if audio_path is None:
            log(f"⚠️ Пропускаю {lang}: не найдена озвучка в {AUDIO_DIR}")
            continue
        if not visual_dir.exists():
            log(f"⚠️ Пропускаю {lang}: не найдена папка визуала {visual_dir}")
            continue

        jobs.append((lang, audio_path, visual_dir))

    if not jobs:
        print("❌ Не найдено ни одной полной пары: озвучка + папка визуала.", file=sys.stderr)
        return

    print("\n=== План работ ===")
    for i, (lang, audio_path, visual_dir) in enumerate(jobs, start=1):
        print(f"{i}. {lang}: {audio_path.name} -> {visual_dir.name}/")
    print("==================\n")

    success = 0
    for index, (lang, audio_path, visual_dir) in enumerate(jobs, start=1):
        print(f"\n##### РОЛИК {index}/{len(jobs)} — {lang} #####")
        if build_video_for_language(lang, audio_path, visual_dir):
            success += 1

    print("\n==================")
    print(f"Готово роликов: {success}/{len(jobs)}")
    print(f"Папка результата: {OUTPUT_VIDEO_DIR}")
    print("==================")


if __name__ == "__main__":
    main()
