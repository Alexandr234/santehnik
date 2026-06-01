# video_creator_timecoded.py
# -*- coding: utf-8 -*-
"""
Монтажный скрипт для DE / PL / RU.
Читает таймкоды из image_times_{lang}.json (генерирует psych_prompt_pipeline.py),
сопоставляет каждый таймкод с картинкой/видео из папки ВИЗУАЛ/{lang}/,
и собирает итоговое видео с озвучкой.

Структура папок проекта:
  ПСИХОЛОГИЯ ГЕРМАНИЯ ПОЛЬША/
    СКРИПТ ОЗВУЧКИ/
      scenario_de.mp3   (или любой файл с _de/_pl/_ru в имени)
      scenario_pl.mp3
      scenario_ru.mp3
    ПРОМПТЫ/
      image_times_de.json   <- генерирует psych_prompt_pipeline.py
      image_times_pl.json
      image_times_ru.json
    ВИЗУАЛ/
      DE/   001.png, 002.png ...  (имена из поля image_filename в JSON, или просто по порядку)
      PL/
      RU/
    ГОТОВЫЕ ВИДЕО/   <- сюда сохраняется результат

Запуск:
    python3 video_creator_timecoded.py
    python3 video_creator_timecoded.py --lang DE
    python3 video_creator_timecoded.py --lang PL RU
"""

from __future__ import annotations

# ── автозапуск через venv если он существует ─────────────────
import os as _os, sys as _sys
from pathlib import Path as _P

_VENV = _P("/Users/aleksandrtomilov/Desktop/ПТИЦЫ АВТОМАТИЗАЦИЯ/venv/bin/python")

def _venv_restart() -> None:
    cur = _P(_sys.executable).resolve()
    if not _VENV.exists() or cur == _VENV.resolve():
        return
    env = _os.environ.copy()
    env["VIRTUAL_ENV"] = str(_VENV.parent.parent)
    env["PATH"] = str(_VENV.parent) + _os.pathsep + env.get("PATH", "")
    print(f"Автозапуск через venv: {_VENV}", flush=True)
    _os.execve(str(_VENV), [str(_VENV), str(_P(__file__).resolve()), *_sys.argv[1:]], env)

_venv_restart()
del _venv_restart

import argparse
import json
import math
import re
import subprocess
import sys
import tempfile
import time
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

PROJECT_ROOT     = Path("/Users/aleksandrtomilov/Desktop/ПСИХОЛОГИЯ ГЕРМАНИЯ ПОЛЬША")
AUDIO_DIR        = PROJECT_ROOT / "СКРИПТ ОЗВУЧКИ"
PROMPTS_DIR      = PROJECT_ROOT / "ПРОМПТЫ"
VISUAL_ROOT_DIR  = PROJECT_ROOT / "ВИЗУАЛ"
OUTPUT_VIDEO_DIR = PROJECT_ROOT / "ГОТОВЫЕ ВИДЕО"
OUTPUT_VIDEO_DIR.mkdir(parents=True, exist_ok=True)

LANGUAGES = ["DE", "PL", "RU"]

# Явные пути к аудио (оставь None — найдёт автоматически по маркеру в имени файла)
AUDIO_FILE_BY_LANG: dict[str, Optional[Path]] = {
    "DE": None,
    "PL": None,
    "RU": None,
}

TARGET_WIDTH  = 1920
TARGET_HEIGHT = 1080
FPS           = 25

ZOOM_START = 1.00
ZOOM_END   = 1.15

VIDEO_CODEC   = "h264_videotoolbox"  # замени на "libx264" если нет Apple Silicon
VIDEO_BITRATE = "5M"
PIXEL_FORMAT  = "yuv420p"
AUDIO_CODEC   = "aac"
AUDIO_BITRATE = "192k"

FILM_GRAIN_STRENGTH = 0

AUDIO_EXTENSIONS  = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg"}
VIDEO_EXTENSIONS  = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}
IMAGE_EXTENSIONS  = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VISUAL_EXTENSIONS = VIDEO_EXTENSIONS | IMAGE_EXTENSIONS

LANG_CUES: dict[str, list[str]] = {
    "DE": ["_de", "-de", " de", "de_", "de-", "deu", "deutsch", "german"],
    "PL": ["_pl", "-pl", " pl", "pl_", "pl-", "pol", "polski", "polish"],
    "RU": ["_ru", "-ru", " ru", "ru_", "ru-", "рус", "русский", "russian", "rus"],
}


################################################
# 2. УТИЛИТЫ
################################################

@dataclass
class Segment:
    kind: str               # "video", "image", "black"
    path: Optional[Path]
    duration: float
    start_seconds: float = 0.0


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def natural_key(path: Path | str):
    name = Path(path).name.lower()
    return [int(p) if p.isdigit() else p for p in re.split(r"(\d+)", name)]


def check_dependencies() -> None:
    print("--- Проверка зависимостей ---")
    for cmd in ["ffmpeg", "ffprobe"]:
        try:
            subprocess.run([cmd, "-version"], capture_output=True, check=True, timeout=5)
            print(f"  {cmd} ... ОК")
        except Exception:
            print(f"Критическая ошибка: {cmd} не найден.", file=sys.stderr)
            sys.exit(1)
    print("-----------------------------\n")


def ffprobe_duration(path: Path) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        raise RuntimeError(f"ffprobe не смог прочитать длительность: {path}\n{r.stderr}")
    try:
        d = float(r.stdout.strip())
    except ValueError:
        raise RuntimeError(f"Некорректная длительность от ffprobe: {path}")
    if d <= 0:
        raise RuntimeError(f"Нулевая длительность аудио: {path}")
    return d


def kind_from_path(path: Path) -> str:
    s = path.suffix.lower()
    if s in VIDEO_EXTENSIONS:
        return "video"
    if s in IMAGE_EXTENSIONS:
        return "image"
    raise ValueError(f"Неподдерживаемый визуальный файл: {path}")


def list_visual_files(visual_dir: Path) -> list[Path]:
    return sorted(
        [p for p in visual_dir.iterdir() if p.is_file() and p.suffix.lower() in VISUAL_EXTENSIONS],
        key=natural_key,
    )


def find_audio_file(lang: str) -> Optional[Path]:
    explicit = AUDIO_FILE_BY_LANG.get(lang)
    if explicit and Path(explicit).exists():
        return Path(explicit)
    if not AUDIO_DIR.exists():
        return None
    cues = LANG_CUES.get(lang, [])
    candidates = [p for p in AUDIO_DIR.iterdir() if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS]
    for p in candidates:
        stem = f" {p.stem.lower()} "
        if any(c in stem for c in cues):
            return p
    # Если файл один и язык один — взять единственный
    if len(candidates) == 1:
        return candidates[0]
    return None


def find_timecodes_json(lang: str) -> Optional[Path]:
    ll = lang.lower()
    candidates = [
        PROMPTS_DIR / f"image_times_{ll}.json",
        PROMPTS_DIR / lang / f"image_times_{ll}.json",
        PROMPTS_DIR / lang.upper() / f"image_times_{ll}.json",
    ]
    for p in candidates:
        if p.exists():
            return p
    # Рекурсивный поиск
    for p in PROMPTS_DIR.rglob(f"*image_times*{ll}*.json"):
        return p
    return None


################################################
# 3. СБОРКА СЕГМЕНТОВ
################################################

def collect_segments_by_timecodes(
    lang: str,
    visual_dir: Path,
    tc_path: Path,
    audio_duration: float,
) -> list[Segment]:
    """
    Читает image_times_{lang}.json и сопоставляет каждый таймкод
    с картинкой/видео из папки визуала.

    Поиск файла по приоритетам:
      1. Точное имя из поля image_filename (001.png, 002.png ...)
      2. По индексу записи в натуральной сортировке папки
      3. Циклически если визуалов меньше чем таймкодов
    """
    raw = json.loads(tc_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise RuntimeError(f"Пустой или некорректный файл таймкодов: {tc_path}")

    visual_files = list_visual_files(visual_dir)
    if not visual_files:
        raise FileNotFoundError(f"В папке визуала нет поддерживаемых файлов: {visual_dir}")

    by_filename: dict[str, Path] = {p.name.lower(): p for p in visual_files}

    segments: list[Segment] = []
    for entry in raw:
        idx      = int(entry.get("index", 1))
        fname    = str(entry.get("image_filename", "")).strip().lower()
        start    = float(entry.get("start_seconds", 0.0))
        end      = float(entry.get("end_seconds", start + 2.35))
        duration = max(0.04, end - start)

        path = by_filename.get(fname)
        if path is None:
            path = visual_files[(idx - 1) % len(visual_files)]

        segments.append(Segment(kind=kind_from_path(path), path=path,
                                duration=duration, start_seconds=start))

    if not segments:
        raise RuntimeError(f"Не удалось собрать сегменты для {lang}")

    # Масштабируем длительности так, чтобы сумма точно совпала с аудио
    total = sum(s.duration for s in segments)
    if abs(total - audio_duration) > 0.05:
        scale = audio_duration / total
        for s in segments:
            s.duration *= scale
        log(f"  {lang}: сумма таймкодов {total:.2f}s -> {audio_duration:.2f}s (×{scale:.4f})")

    if len(visual_files) < len(segments):
        log(f"  ⚠️  {lang}: визуалов ({len(visual_files)}) < таймкодов ({len(segments)}) — файлы повторяются")
    elif len(visual_files) > len(segments):
        log(f"  ℹ️  {lang}: визуалов ({len(visual_files)}) > таймкодов ({len(segments)}) — лишние не используются")

    log(f"  {lang}: {len(segments)} сегментов по таймкодам из {tc_path.name}")
    return segments


def collect_segments_evenly(lang: str, visual_dir: Path, audio_duration: float) -> list[Segment]:
    """Резервный режим без таймкодов: равномерное распределение."""
    visual_files = list_visual_files(visual_dir)
    if not visual_files:
        raise FileNotFoundError(f"В папке визуала нет файлов: {visual_dir}")
    n = len(visual_files)
    segments, prev = [], 0.0
    for i, path in enumerate(visual_files, 1):
        boundary = audio_duration if i == n else (audio_duration * i / n)
        dur = max(0.0, boundary - prev)
        prev = boundary
        if dur > 0:
            segments.append(Segment(kind=kind_from_path(path), path=path, duration=dur))
    log(f"  {lang}: равномерный режим — {n} файлов, ~{audio_duration / n:.2f}s каждый")
    return segments


################################################
# 4. РЕНДЕР КАДРОВ
################################################

def read_image_unicode(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Не удалось прочитать: {path}")
    return img


def resize_cover(frame, w: int, h: int):
    sh, sw = frame.shape[:2]
    scale = max(w / sw, h / sh)
    nw = max(w, int(math.ceil(sw * scale)))
    nh = max(h, int(math.ceil(sh * scale)))
    r = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    x1 = (nw - w) // 2
    y1 = (nh - h) // 2
    c = r[y1:y1 + h, x1:x1 + w]
    if c.shape[1] != w or c.shape[0] != h:
        c = cv2.resize(c, (w, h), interpolation=cv2.INTER_LINEAR)
    return c


def apply_zoom(frame, zoom: float):
    if zoom <= 1.0001:
        return frame
    h, w = frame.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), 0, float(zoom))
    return cv2.warpAffine(frame, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def zoom_at(i: int, total: int) -> float:
    return ZOOM_START if total <= 1 else ZOOM_START + (ZOOM_END - ZOOM_START) * (i / max(1, total - 1))


def iter_image_frames(seg: Segment, total: int) -> Iterator[np.ndarray]:
    base = resize_cover(read_image_unicode(seg.path), TARGET_WIDTH, TARGET_HEIGHT)
    for i in range(total):
        yield apply_zoom(base, zoom_at(i, total))


def iter_video_frames(seg: Segment, total: int) -> Iterator[np.ndarray]:
    cap = cv2.VideoCapture(str(seg.path))
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {seg.path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or FPS
    if src_fps <= 1:
        src_fps = FPS
    last_frame, last_src = None, -1
    try:
        for i in range(total):
            src_idx = int(round(i / FPS * src_fps))
            if src_idx != last_src + 1:
                cap.set(cv2.CAP_PROP_POS_FRAMES, src_idx)
            ok, frame = cap.read()
            if not ok:
                frame = last_frame.copy() if last_frame is not None else np.zeros((TARGET_HEIGHT, TARGET_WIDTH, 3), dtype=np.uint8)
            else:
                last_frame = frame
            last_src = src_idx
            frame = resize_cover(frame, TARGET_WIDTH, TARGET_HEIGHT)
            yield apply_zoom(frame, zoom_at(i, total))
    finally:
        cap.release()


def iter_black_frames(total: int) -> Iterator[np.ndarray]:
    f = np.zeros((TARGET_HEIGHT, TARGET_WIDTH, 3), dtype=np.uint8)
    for _ in range(total):
        yield f.copy()


def iter_segment_frames(seg: Segment, total: int) -> Iterator[np.ndarray]:
    if seg.kind == "image":
        yield from iter_image_frames(seg, total)
    elif seg.kind == "video":
        yield from iter_video_frames(seg, total)
    else:
        yield from iter_black_frames(total)


def compute_frame_counts(segments: list[Segment]) -> list[int]:
    counts, acc_t, acc_f = [], 0.0, 0
    for seg in segments:
        acc_t += seg.duration
        target = max(int(round(acc_t * FPS)), acc_f + 1)
        counts.append(target - acc_f)
        acc_f = target
    return counts


def render_visual_track(segments: list[Segment], output_path: Path) -> None:
    frame_counts = compute_frame_counts(segments)
    total_frames = sum(frame_counts)

    vf = [f"format={PIXEL_FORMAT}"]
    if FILM_GRAIN_STRENGTH:
        vf.insert(0, f"noise=alls={FILM_GRAIN_STRENGTH}:allf=t+u")

    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{TARGET_WIDTH}x{TARGET_HEIGHT}",
        "-r", str(FPS), "-i", "-",
        "-vf", ",".join(vf), "-an",
        "-c:v", VIDEO_CODEC, "-b:v", VIDEO_BITRATE, "-pix_fmt", PIXEL_FORMAT,
        str(output_path),
    ]

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    written = 0
    try:
        with tqdm(total=total_frames, desc="  Рендер", unit="кадр") as pbar:
            for seg, n in zip(segments, frame_counts):
                for frame in iter_segment_frames(seg, n):
                    proc.stdin.write(frame.tobytes())
                    written += 1
                    pbar.update(1)
        proc.stdin.close()
        stderr = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
        if proc.wait() != 0:
            raise RuntimeError(f"FFmpeg ошибка рендера:\n{stderr}")
    except BrokenPipeError:
        stderr = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
        proc.kill()
        raise RuntimeError(f"FFmpeg закрыл pipe:\n{stderr}")
    except Exception:
        proc.kill()
        raise

    log(f"  Видеоряд: {written} кадров -> {output_path.name}")


################################################
# 5. ФИНАЛЬНАЯ СБОРКА
################################################

def mux_audio(visual_path: Path, audio_path: Path, output_path: Path, audio_duration: float) -> None:
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "warning",
        "-i", str(visual_path), "-i", str(audio_path),
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy", "-c:a", AUDIO_CODEC, "-b:a", AUDIO_BITRATE,
        "-t", f"{audio_duration:.3f}", "-movflags", "+faststart",
        str(output_path),
    ]
    log("  Добавляю озвучку...")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"FFmpeg ошибка финальной сборки:\n{r.stderr}")


################################################
# 6. ОДИН ЯЗЫКОВОЙ РОЛИК
################################################

def build_video_for_language(lang: str) -> bool:
    t0 = time.time()
    output_path = OUTPUT_VIDEO_DIR / f"{lang}_video.mp4"
    visual_dir  = VISUAL_ROOT_DIR / lang

    log(f"\n{'=' * 42}")
    log(f"  Ролик {lang}")

    audio_path = find_audio_file(lang)
    if audio_path is None:
        log(f"  ⚠️  Озвучка для {lang} не найдена в {AUDIO_DIR} — пропускаю")
        return False
    log(f"  Озвучка:  {audio_path.name}")

    if not visual_dir.exists():
        log(f"  ⚠️  Нет папки визуала: {visual_dir} — пропускаю")
        return False

    tc_path = find_timecodes_json(lang)
    if tc_path:
        log(f"  Таймкоды: {tc_path.relative_to(PROJECT_ROOT)}")
    else:
        log(f"  ⚠️  Таймкоды (image_times_{lang.lower()}.json) не найдены — режим: равномерно")

    log(f"  Выход:    {output_path.name}")
    log(f"{'=' * 42}")

    try:
        audio_duration = ffprobe_duration(audio_path)
        log(f"  Длительность: {audio_duration:.2f}s")

        if tc_path:
            segments = collect_segments_by_timecodes(lang, visual_dir, tc_path, audio_duration)
        else:
            segments = collect_segments_evenly(lang, visual_dir, audio_duration)

        avg = audio_duration / len(segments) if segments else 0
        n_img = sum(1 for s in segments if s.kind == "image")
        n_vid = sum(1 for s in segments if s.kind == "video")
        log(f"  Сегментов: {len(segments)} ({n_img} картинок + {n_vid} видео), ~{avg:.2f}s в среднем")

        with tempfile.TemporaryDirectory(prefix=f"psych_{lang}_") as td:
            tmp = Path(td)
            visual_track = tmp / f"{lang}_visual.mp4"
            render_visual_track(segments, visual_track)
            mux_audio(visual_track, audio_path, output_path, audio_duration)

        log(f"  ✅ Готово: {output_path}")
        log(f"  Время:    {(time.time() - t0) / 60:.2f} мин")
        return True

    except Exception as exc:
        log(f"  ❌ Ошибка {lang}: {exc}")
        import traceback; traceback.print_exc()
        return False


################################################
# 7. ГЛАВНАЯ
################################################

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Монтаж DE/PL/RU: таймкоды из psych_prompt_pipeline.py + визуал + озвучка"
    )
    parser.add_argument("--lang", nargs="+", choices=LANGUAGES, default=None,
                        help="Языки для обработки. По умолчанию — все три.")
    args = parser.parse_args()

    check_dependencies()

    for folder, name in [(AUDIO_DIR, "СКРИПТ ОЗВУЧКИ"), (VISUAL_ROOT_DIR, "ВИЗУАЛ")]:
        if not folder.exists():
            print(f"❌ Папка не найдена: {folder}", file=sys.stderr)
            sys.exit(1)

    langs_to_process = args.lang or LANGUAGES

    print("\n=== План работ ===")
    jobs = []
    for lang in langs_to_process:
        audio   = find_audio_file(lang)
        vis_dir = VISUAL_ROOT_DIR / lang
        tc      = find_timecodes_json(lang)
        ok_a = audio is not None and audio.exists()
        ok_v = vis_dir.exists() and bool(list_visual_files(vis_dir))
        mode = f"по таймкодам ({tc.name})" if tc else "равномерно (без таймкодов)"
        flag = "✅" if ok_a and ok_v else "⚠️  пропущу"
        print(f"  {lang}: аудио={'✅' if ok_a else '❌'} | визуал={'✅' if ok_v else '❌'} | {mode} — {flag}")
        if ok_a and ok_v:
            jobs.append(lang)
    print("==================\n")

    if not jobs:
        print("❌ Ничего для обработки.", file=sys.stderr)
        sys.exit(1)

    success = 0
    for i, lang in enumerate(jobs, 1):
        print(f"\n##### РОЛИК {i}/{len(jobs)} — {lang} #####")
        if build_video_for_language(lang):
            success += 1

    print(f"\n{'=' * 42}")
    print(f"Готово: {success}/{len(jobs)} роликов")
    print(f"Папка:  {OUTPUT_VIDEO_DIR}")
    print(f"{'=' * 42}")


if __name__ == "__main__":
    main()
