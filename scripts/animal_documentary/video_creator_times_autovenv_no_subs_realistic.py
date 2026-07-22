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
#    Скрипт сам режет озвучку по фразам и вставляет тишину на время перебивок,
#    так что финальный звук диктора точно совпадает с растянутой шкалой.
#    Старый формат (без type/src) полностью поддерживается: всё считается speech,
#    озвучка идёт сплошным куском, как раньше.
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


def collect_segments_for_language(lang: str, visual_dir: Path, audio_duration: float, timings: list[TimingEntry]) -> list[Segment]:
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
        duration = timing.duration
        if i < len(selected_paths) - 1 and TRANSITION_DURATION > 0:
            duration += TRANSITION_DURATION
        segments.append(Segment(kind=kind, path=path, duration=duration))

    # Сверяем длину озвучки с суммарной длительностью РЕЧЕВЫХ интервалов
    # (паузы-перебивки в озвучке отсутствуют — на их месте будет тишина).
    speech_total = sum(
        (t.src_end - t.src_start) if (t.src_start is not None and t.src_end is not None) else t.duration
        for t in timings if t.kind == "speech"
    )
    if abs(audio_duration - speech_total) > 1.5:
        log(
            f"⚠️ Длина озвучки ({audio_duration:.2f} сек.) отличается от суммы речевых интервалов "
            f"({speech_total:.2f} сек.). Проверь, что тайминги сделаны из этой озвучки."
        )

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


def build_voiceover_with_pauses(audio_path: Path, timings: list[TimingEntry], out_wav: Path) -> None:
    """
    Собирает дорожку ДИКТОРА по финальной шкале ролика:
      - для каждой строки type: speech вырезается кусок исходной озвучки (диапазон src);
      - для каждой строки type: broll вставляется тишина такой же длины (пауза-перебивка);
      - всё склеивается по порядку — голос точно совпадает с растянутой шкалой видео.
    Если пауз нет и src совпадает с основной шкалой (старый формат) —
    озвучка просто конвертируется в wav сплошным куском, как раньше.
    """
    has_pauses = any(t.kind == "broll" for t in timings)
    shifted = any(
        t.kind == "speech" and t.src_start is not None and abs(t.src_start - t.start) > 0.01
        for t in timings
    )

    if not has_pauses and not shifted:
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(audio_path),
            "-vn", "-ac", "2", "-ar", str(AUDIO_SAMPLE_RATE),
            "-c:a", "pcm_s16le",
            str(out_wav),
        ]
        run_ffmpeg(cmd, "конвертация озвучки")
        return

    fmt = f"aresample={AUDIO_SAMPLE_RATE},aformat=sample_fmts=s16:channel_layouts=stereo"
    filter_parts: list[str] = []
    labels: list[str] = []

    for k, t in enumerate(timings):
        lab = f"p{k}"
        if t.kind == "speech":
            ss = max(0.0, t.src_start if t.src_start is not None else t.start)
            se = max(ss + 0.01, t.src_end if t.src_end is not None else t.end)
            filter_parts.append(
                f"[0:a]atrim={ss:.3f}:{se:.3f},asetpts=PTS-STARTPTS,{fmt}[{lab}]"
            )
        else:
            # Пауза-перебивка: диктор молчит, звучит только природа из ambient-дорожки.
            filter_parts.append(
                f"aevalsrc=0:d={t.duration:.3f}:s={AUDIO_SAMPLE_RATE},"
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
    log(f"Озвучка: режу на {sum(1 for t in timings if t.kind == 'speech')} фраз, "
        f"вставляю {sum(1 for t in timings if t.kind == 'broll')} пауз")
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

    filter_complex = (
        f"[2:a]volume={AMBIENT_VOLUME:.3f}[amb0];"
        f"[1:a]asplit=2[vo1][vo2];"
        f"[amb0][vo1]sidechaincompress="
        f"threshold={DUCK_THRESHOLD}:ratio={DUCK_RATIO}:attack={DUCK_ATTACK_MS}:release={DUCK_RELEASE_MS}[ambduck];"
        f"[vo2][ambduck]amix=inputs=2:duration=first:normalize=0[aout]"
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
        segments = collect_segments_for_language(lang, visual_dir, audio_duration, timings)

        video_count = sum(1 for s in segments if s.kind == "video")
        image_count = sum(1 for s in segments if s.kind == "image")
        broll_count = sum(1 for t in timings if t.kind == "broll")
        log(f"Сегменты: {video_count} видео + {image_count} картинок (из них пауз-перебивок: {broll_count})")
        log(f"Первое видео: {segments[0].path.name} | всего видео: {video_count}")

        with tempfile.TemporaryDirectory(prefix=f"video_{lang}_") as td:
            temp_dir = Path(td)
            visual_track_path = temp_dir / f"{lang}_visual_no_audio.mp4"
            total_duration = render_visual_track(segments, visual_track_path)

            # 1) Голос диктора по финальной шкале: фразы + тишина в паузах-перебивках.
            voice_wav = temp_dir / f"{lang}_voice.wav"
            build_voiceover_with_pauses(audio_path, timings, voice_wav)

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
