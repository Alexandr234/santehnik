# video_creator_DE_PL_RU_even_visuals_zoom_0_to_15.py
# -*- coding: utf-8 -*-

from __future__ import annotations

# ------------------------------------------------------------
# АВТОЗАПУСК ЧЕРЕЗ VENV
# Если скрипт запустили системным Python, он сам перезапустится
# через старый venv, если он существует:
# /Users/aleksandrtomilov/Desktop/ПТИЦЫ АВТОМАТИЗАЦИЯ/venv/bin/python
# ------------------------------------------------------------
import os as _bootstrap_os
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath

_PROJECT_ROOT_FOR_VENV = _BootstrapPath("/Users/aleksandrtomilov/Desktop/ПТИЦЫ АВТОМАТИЗАЦИЯ")
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
del _restart_inside_venv_if_needed

import math
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
# 1. КОНФИГ ДЛЯ ПРОЕКТА DE / PL / RU БЕЗ ТАЙМКОДОВ
################################################

PROJECT_ROOT = Path("/Users/aleksandrtomilov/Desktop/ПСИХОЛОГИЯ ГЕРМАНИЯ ПОЛЬША")

AUDIO_DIR = PROJECT_ROOT / "СКРИПТ ОЗВУЧКИ"
VISUAL_ROOT_DIR = PROJECT_ROOT / "ВИЗУАЛ"
OUTPUT_VIDEO_DIR = PROJECT_ROOT / "ГОТОВЫЕ ВИДЕО"
OUTPUT_VIDEO_DIR.mkdir(parents=True, exist_ok=True)

# Обрабатываем три языковые версии ролика.
LANGUAGES = ["DE", "PL", "RU"]

# Озвучку ищем по имени файла: RU / PL / DE (без учёта регистра),
# с любым поддерживаемым аудиорасширением. Конкретные пути находим в рантайме
# через find_audio_for_lang(), потому что расширение заранее неизвестно.

VISUAL_DIR_BY_LANG = {
    "DE": VISUAL_ROOT_DIR / "DE",
    "PL": VISUAL_ROOT_DIR / "PL",
    "RU": VISUAL_ROOT_DIR / "RU",
}

# Видео параметры.
TARGET_WIDTH = 1920
TARGET_HEIGHT = 1080
FPS = 25

# ЗУМ (эффект Кена Бёрнса): плавный наезд на КАЖДЫЙ визуал.
# Начинаем с 0% приближения (1.00) и к концу показа визуала доходим до 15% (1.15).
# Ramp сбрасывается на каждой новой картинке/видео, то есть каждый визуал
# самостоятельно проезжает путь 0% -> 15%.
ZOOM_PERCENT = 0.07                    # насколько приближаем к концу визуала (0.07 = 7%)
ZOOM_START = 1.00                      # старт: без приближения
ZOOM_END = 1.00 + ZOOM_PERCENT         # финиш: +15%

# Переходов нет: следующий визуал начинается сразу после предыдущего.
TRANSITION_DURATION = 0.0

# FFmpeg параметры.
VIDEO_CODEC = "h264_videotoolbox"  # macOS. Если будет ошибка, замени на "libx264".
VIDEO_BITRATE = "5M"
PIXEL_FORMAT = "yuv420p"
AUDIO_CODEC = "aac"
AUDIO_BITRATE = "192k"

# Плёночное зерно. Поставь 0, если не нужно.
FILM_GRAIN_STRENGTH = 0

AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VISUAL_EXTENSIONS = VIDEO_EXTENSIONS | IMAGE_EXTENSIONS

################################################
# 2. МОДЕЛИ И УТИЛИТЫ
################################################


@dataclass
class Segment:
    kind: str  # "video", "image" или "black"
    path: Optional[Path]
    duration: float
    freeze_at: float = 0.0


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
        duration = float(result.stdout.strip())
    except ValueError:
        raise RuntimeError(f"Некорректная длительность от ffprobe для файла: {file_path}")

    if duration <= 0:
        raise RuntimeError(f"Нулевая или некорректная длительность аудио: {file_path}")

    return duration


def find_audio_for_lang(lang: str) -> Optional[Path]:
    """
    Ищет файл озвучки по имени языка: RU / PL / DE.

    Имя файла (без расширения) должно точно совпадать с кодом языка
    без учёта регистра, например: RU.mp3, ru.wav, De.m4a, pl.aac.
    Расширение — любое из AUDIO_EXTENSIONS. Возвращает None, если не нашли.
    """
    if not AUDIO_DIR.exists():
        return None

    candidates = [
        p
        for p in AUDIO_DIR.iterdir()
        if p.is_file()
        and p.suffix.lower() in AUDIO_EXTENSIONS
        and p.stem.lower() == lang.lower()
    ]
    if not candidates:
        return None

    # Если вдруг несколько (RU.mp3 и RU.wav) — берём стабильно первый по имени.
    return sorted(candidates, key=natural_key)[0]


def list_visual_files(visual_dir: Path) -> list[Path]:
    return sorted(
        [p for p in visual_dir.iterdir() if p.is_file() and p.suffix.lower() in VISUAL_EXTENSIONS],
        key=natural_key,
    )


def kind_from_path(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in VIDEO_EXTENSIONS:
        return "video"
    if suffix in IMAGE_EXTENSIONS:
        return "image"
    raise ValueError(f"Неподдерживаемый визуальный файл: {path}")


def collect_segments_for_language(lang: str, visual_dir: Path, audio_duration: float) -> list[Segment]:
    """
    Собирает видеоряд БЕЗ таймкодов.

    Логика:
      1) Берём все картинки/видео из папки языка: ВИЗУАЛ/DE, ВИЗУАЛ/PL или ВИЗУАЛ/RU.
      2) Сортируем естественно: 1, 2, 3 ... 10, 11.
      3) Равномерно распределяем все визуалы по всей длине озвучки.

    Пример:
      аудио 600 сек, в папке 100 картинок -> каждая картинка длится примерно 6 сек.
    """
    visual_files = list_visual_files(visual_dir)
    if not visual_files:
        raise FileNotFoundError(f"В папке визуала нет поддерживаемых файлов: {visual_dir}")

    visual_count = len(visual_files)
    segments: list[Segment] = []

    previous_boundary = 0.0
    for i, path in enumerate(visual_files, start=1):
        # Через границы, а не просто audio_duration / visual_count, чтобы сумма всегда точно совпала с аудио.
        next_boundary = audio_duration if i == visual_count else (audio_duration * i / visual_count)
        duration = max(0.0, next_boundary - previous_boundary)
        previous_boundary = next_boundary

        if duration <= 0:
            continue

        segments.append(Segment(kind=kind_from_path(path), path=path, duration=duration))

    if not segments:
        raise RuntimeError(f"Не удалось собрать сегменты для {lang}: {visual_dir}")

    total_duration = sum(segment.duration for segment in segments)
    avg_duration = total_duration / len(segments)

    total_frames = int(round(audio_duration * FPS))
    if len(segments) > total_frames:
        log(
            f"⚠️ {lang}: визуалов ({len(segments)}) больше, чем кадров в аудио ({total_frames}). "
            "Некоторые фрагменты будут длиться 1 кадр, итог может быть чуть длиннее аудио."
        )

    log(f"Визуалы {lang}: {len(segments)} файлов из {visual_dir.name}/")
    log(f"Распределение {lang}: примерно по {avg_duration:.2f} сек. на визуал, всего {total_duration:.2f} сек.")

    return segments


################################################
# 3. РЕНДЕР КАДРОВ: ВИДЕО/КАРТИНКИ С ЗУМОМ 0→15% + БЕЗ ПЕРЕХОДОВ
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
    """Стабильный зум по центру без микродрожи."""
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


def zoom_for_frame(frame_index: int, total_frames: int) -> float:
    """
    Плавный линейный наезд на визуал: от ZOOM_START (0%) на первом кадре
    до ZOOM_END (+15%) на последнем кадре сегмента.
    Отсчёт кадров — внутри одного сегмента, поэтому каждый визуал проезжает
    полный путь 0% -> 15% заново.
    """
    if total_frames <= 1:
        return ZOOM_START
    t = frame_index / (total_frames - 1)
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    return ZOOM_START + (ZOOM_END - ZOOM_START) * t


def normalize_frame(frame, zoom: float = ZOOM_START):
    frame = resize_cover(frame, TARGET_WIDTH, TARGET_HEIGHT)
    frame = apply_center_zoom(frame, zoom)
    return frame


def iter_image_frames(segment: Segment, total_frames: int) -> Iterator[np.ndarray]:
    if segment.path is None:
        raise RuntimeError("У image-сегмента нет пути к файлу")

    # Готовим базовый кадр 1920x1080 один раз, а зум применяем покадрово,
    # чтобы получить плавный наезд 0% -> 15% на протяжении показа картинки.
    base = read_image_unicode(segment.path)
    base = resize_cover(base, TARGET_WIDTH, TARGET_HEIGHT)

    for i in range(total_frames):
        zoom = zoom_for_frame(i, total_frames)
        yield apply_center_zoom(base, zoom)


def iter_video_frames(segment: Segment, total_frames: int) -> Iterator[np.ndarray]:
    if segment.path is None:
        raise RuntimeError("У video-сегмента нет пути к файлу")

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
            zoom = zoom_for_frame(out_i, total_frames)
            yield normalize_frame(frame, zoom=zoom)
    finally:
        cap.release()


def iter_black_frames(total_frames: int) -> Iterator[np.ndarray]:
    frame = np.zeros((TARGET_HEIGHT, TARGET_WIDTH, 3), dtype=np.uint8)
    for _ in range(total_frames):
        yield frame.copy()


def iter_segment_frames(segment: Segment, total_frames: int) -> Iterator[np.ndarray]:
    if segment.kind == "image":
        yield from iter_image_frames(segment, total_frames)
    elif segment.kind == "video":
        yield from iter_video_frames(segment, total_frames)
    elif segment.kind == "black":
        yield from iter_black_frames(total_frames)
    else:
        raise ValueError(f"Неизвестный тип сегмента: {segment.kind}")


def compute_frame_counts_and_transitions(segments: list[Segment]) -> tuple[list[int], list[int]]:
    # Считаем кадры накопительно, а не округляем каждый сегмент отдельно.
    # Так длинный ролик не теряет доли секунды из-за постоянных округлений duration * FPS.
    frame_counts: list[int] = []
    accumulated_time = 0.0
    accumulated_frames = 0

    for segment in segments:
        accumulated_time += segment.duration
        target_total_frames = int(round(accumulated_time * FPS))

        # Каждый сегмент должен получить хотя бы 1 кадр, даже если интервал очень короткий.
        if target_total_frames <= accumulated_frames:
            target_total_frames = accumulated_frames + 1

        frame_count = target_total_frames - accumulated_frames
        frame_counts.append(frame_count)
        accumulated_frames = target_total_frames

    base_transition_frames = max(0, int(round(TRANSITION_DURATION * FPS)))

    transitions = []
    for i in range(len(segments) - 1):
        tf = min(base_transition_frames, frame_counts[i] // 3, frame_counts[i + 1] // 3)
        transitions.append(max(0, tf))

    return frame_counts, transitions


def render_visual_track(segments: list[Segment], output_path: Path) -> float:
    """Рендерит весь видеоряд в один mp4 без аудио и без плавных переходов."""
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

                    for frame in head_frames[blend_count:]:
                        proc.stdin.write(frame.tobytes())
                        written_frames += 1
                        pbar.update(1)

                elif index == 0:
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
# 4. ФИНАЛЬНАЯ СБОРКА БЕЗ СУБТИТРОВ И БЕЗ ЗВУКА ИСХОДНЫХ ВИДЕО
################################################


def mux_audio_only(
    visual_video_path: Path,
    audio_path: Path,
    output_path: Path,
    audio_duration: float,
) -> None:
    """
    Финальная сборка:
      - берём только видеоряд из visual_video_path;
      - берём только озвучку из audio_path;
      - звук исходных видео не используется вообще;
      - субтитры не генерируются и не вшиваются.
    """
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "warning",
        "-i", str(visual_video_path),
        "-i", str(audio_path),
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", AUDIO_CODEC,
        "-b:a", AUDIO_BITRATE,
        "-t", f"{audio_duration:.3f}",
        "-movflags", "+faststart",
        str(output_path),
    ]

    log("Финальная сборка: добавляю только озвучку, звук исходных видео не используется")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Финальная сборка FFmpeg завершилась с ошибкой:\n{result.stderr}")


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
    log("  Таймкоды: не используются")
    log(f"  Выход:   {output_path}")
    log("======================\n")

    try:
        audio_duration = ffprobe_duration(audio_path)
        log(f"Длительность озвучки: {audio_duration:.2f} сек.")

        segments = collect_segments_for_language(lang, visual_dir, audio_duration)

        video_count = sum(1 for s in segments if s.kind == "video")
        image_count = sum(1 for s in segments if s.kind == "image")
        black_count = sum(1 for s in segments if s.kind == "black")
        log(f"Сегменты: {video_count} видео + {image_count} картинок + {black_count} чёрных пауз")
        log("Режим: без таймкодов, все визуалы равномерно растянуты на всю длину аудио.")
        log(f"Переходы: отключены. Зум: наезд 0% -> {int(round(ZOOM_PERCENT * 100))}% на каждый визуал. Субтитры: отключены.")

        with tempfile.TemporaryDirectory(prefix=f"video_{lang}_") as td:
            temp_dir = Path(td)
            visual_track_path = temp_dir / f"{lang}_visual_no_audio.mp4"
            render_visual_track(segments, visual_track_path)

            mux_audio_only(
                visual_video_path=visual_track_path,
                audio_path=audio_path,
                output_path=output_path,
                audio_duration=audio_duration,
            )

        elapsed_min = (time.time() - start_time) / 60
        log(f"✅ Готово {lang}: {output_path}")
        log(f"Время: {elapsed_min:.2f} минут\n")
        return True

    except Exception as exc:
        log(f"❌ Ошибка в ролике {lang}: {exc}")
        return False


################################################
# 6. ГЛАВНАЯ: 3 РОЛИКА ПОДРЯД
################################################


def main() -> None:
    check_dependencies()

    if not AUDIO_DIR.exists():
        print(f"❌ Папка с озвучками не найдена: {AUDIO_DIR}", file=sys.stderr)
        return

    if not VISUAL_ROOT_DIR.exists():
        print(f"❌ Папка с визуалом не найдена: {VISUAL_ROOT_DIR}", file=sys.stderr)
        return

    jobs = []
    for lang in LANGUAGES:
        audio_path = find_audio_for_lang(lang)
        visual_dir = VISUAL_DIR_BY_LANG[lang]

        if audio_path is None:
            log(f"⚠️ Пропускаю {lang}: не найден аудиофайл с именем {lang} в {AUDIO_DIR}")
            continue
        if not visual_dir.exists():
            log(f"⚠️ Пропускаю {lang}: не найдена папка визуала {visual_dir}")
            continue

        visual_files = list_visual_files(visual_dir)
        if not visual_files:
            log(f"⚠️ Пропускаю {lang}: в папке нет картинок/видео {visual_dir}")
            continue

        jobs.append((lang, audio_path, visual_dir, len(visual_files)))

    if not jobs:
        print("❌ Не найдено ни одной полной пары: озвучка + папка визуала.", file=sys.stderr)
        return

    print("\n=== План работ ===")
    for i, (lang, audio_path, visual_dir, visual_count) in enumerate(jobs, start=1):
        print(f"{i}. {lang}: {audio_path.name} + {visual_dir.name}/ ({visual_count} визуалов), без таймкодов")
    print("==================\n")

    success = 0
    for index, (lang, audio_path, visual_dir, _visual_count) in enumerate(jobs, start=1):
        print(f"\n##### РОЛИК {index}/{len(jobs)} — {lang} #####")
        if build_video_for_language(lang, audio_path, visual_dir):
            success += 1

    print("\n==================")
    print(f"Готово роликов: {success}/{len(jobs)}")
    print(f"Папка результата: {OUTPUT_VIDEO_DIR}")
    print("==================")


if __name__ == "__main__":
    main()
