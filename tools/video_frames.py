#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
video_frames.py — раскадровка видео для анализа нейросетью.

Что делает:
  * берёт все видео из указанной папки (по умолчанию — "ФАБРИКА ВИДЕО" на рабочем столе);
  * скринит каждые N секунд (по умолчанию 2 -> 30 кадров в минуту, 300 кадров на 10 минут);
  * складывает кадры по порядку в отдельную папку для каждого видео;
  * собирает PDF-контактный лист (сетка кадров с таймкодами, по 30 кадров = 1 минута на страницу);
  * пакует кадры в ZIP-архив.

PDF и ZIP удобно закинуть в нейросеть, чтобы она "прочитала" логику и стиль кадров по порядку.

Зависимости:
  * ffmpeg  — системная утилита (macOS:  brew install ffmpeg)
  * Pillow  — python-библиотека (pip3 install pillow)   [нужна только для PDF]

Примеры запуска:
  python3 video_frames.py
  python3 video_frames.py "/Users/aleksandrtomilov/Desktop/ФАБРИКА ВИДЕО"
  python3 video_frames.py --interval 2 --cols 5 --rows 6 --no-zip
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm", ".mpg", ".mpeg", ".flv", ".wmv"}
DEFAULT_INPUT = os.path.expanduser("~/Desktop/ФАБРИКА ВИДЕО")


def die(msg: str, code: int = 1):
    print(f"\n❌ {msg}", file=sys.stderr)
    sys.exit(code)


def check_ffmpeg():
    if shutil.which("ffmpeg") is None:
        die("Не найден ffmpeg. Установите его:  brew install ffmpeg")


def load_pillow():
    try:
        from PIL import Image, ImageDraw, ImageFont  # noqa
        return Image, ImageDraw, ImageFont
    except ImportError:
        die("Не найден Pillow (нужен для PDF). Установите:  pip3 install pillow")


def human_time(seconds: float) -> str:
    seconds = int(round(seconds))
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:d}:{s:02d}"


def extract_frames(video: Path, out_dir: Path, interval: float) -> list[Path]:
    """Достаёт по одному кадру каждые `interval` секунд. Кадры именуются по порядку."""
    out_dir.mkdir(parents=True, exist_ok=True)
    # чистим старые кадры, чтобы не смешивать прогоны
    for old in out_dir.glob("frame_*.jpg"):
        old.unlink()

    pattern = str(out_dir / "frame_%05d.jpg")
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", str(video),
        "-vf", f"fps=1/{interval}",
        "-q:v", "2",              # высокое качество JPEG
        "-fps_mode", "vfr",
        pattern,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        # старые сборки ffmpeg не знают -fps_mode -> пробуем без него
        cmd_fallback = [c for c in cmd if c not in ("-fps_mode", "vfr")]
        result = subprocess.run(cmd_fallback, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"   ⚠️  ffmpeg не смог обработать {video.name}: {result.stderr.strip()[:300]}")
            return []

    frames = sorted(out_dir.glob("frame_*.jpg"))
    return frames


def build_contact_pdf(frames: list[Path], pdf_path: Path, interval: float,
                      cols: int, rows: int, thumb_w: int = 480):
    """Собирает PDF: сетка кадров с таймкодами, cols*rows кадров на страницу."""
    Image, ImageDraw, ImageFont = load_pillow()

    # шрифт для подписей-таймкодов
    font = None
    for fp in [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/SFNS.ttf",
        "/Library/Fonts/Arial.ttf",
    ]:
        if os.path.exists(fp):
            try:
                font = ImageFont.truetype(fp, 20)
                break
            except Exception:
                pass
    if font is None:
        font = ImageFont.load_default()

    label_h = 28
    pad = 10
    per_page = cols * rows
    pages = []

    # определяем высоту миниатюры по первому кадру (сохраняем пропорции)
    with Image.open(frames[0]) as im0:
        w0, h0 = im0.size
    thumb_h = max(1, int(thumb_w * h0 / w0))

    cell_w = thumb_w + pad
    cell_h = thumb_h + label_h + pad
    page_w = cols * cell_w + pad
    page_h = rows * cell_h + pad

    for start in range(0, len(frames), per_page):
        chunk = frames[start:start + per_page]
        page = Image.new("RGB", (page_w, page_h), "white")
        draw = ImageDraw.Draw(page)

        for idx, frame in enumerate(chunk):
            global_idx = start + idx
            timecode = human_time(global_idx * interval)
            r, c = divmod(idx, cols)
            x = pad + c * cell_w
            y = pad + r * cell_h

            try:
                with Image.open(frame) as im:
                    im = im.convert("RGB").resize((thumb_w, thumb_h))
                    page.paste(im, (x, y + label_h))
            except Exception as e:
                draw.rectangle([x, y + label_h, x + thumb_w, y + label_h + thumb_h],
                               outline="red")
                draw.text((x + 5, y + label_h + 5), f"err: {e}", fill="red", font=font)

            draw.text((x + 2, y + 3), f"#{global_idx + 1}  {timecode}",
                      fill="black", font=font)

        pages.append(page)

    if not pages:
        return
    pages[0].save(pdf_path, "PDF", save_all=True, append_images=pages[1:], resolution=150)


def zip_frames(frames: list[Path], zip_path: Path):
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in frames:
            zf.write(f, arcname=f.name)


def safe_name(name: str) -> str:
    return re.sub(r"[^\w\-.]+", "_", name, flags=re.UNICODE).strip("_") or "video"


def process_video(video: Path, out_root: Path, interval: float,
                  cols: int, rows: int, make_pdf: bool, make_zip: bool):
    stem = safe_name(video.stem)
    out_dir = out_root / stem
    frames_dir = out_dir / "frames"

    print(f"\n🎬 {video.name}")
    frames = extract_frames(video, frames_dir, interval)
    if not frames:
        print("   (кадры не извлечены — пропуск)")
        return
    print(f"   кадров: {len(frames)}  (~{human_time(len(frames) * interval)} видео)")

    if make_pdf:
        pdf_path = out_dir / f"{stem}_кадры.pdf"
        build_contact_pdf(frames, pdf_path, interval, cols, rows)
        print(f"   📄 PDF:  {pdf_path}")

    if make_zip:
        zip_path = out_dir / f"{stem}_кадры.zip"
        zip_frames(frames, zip_path)
        print(f"   🗜  ZIP:  {zip_path}")


def main():
    ap = argparse.ArgumentParser(
        description="Раскадровка видео каждые N секунд + сборка PDF/ZIP для анализа нейросетью."
    )
    ap.add_argument("input", nargs="?", default=DEFAULT_INPUT,
                    help=f"Папка с видео (по умолчанию: {DEFAULT_INPUT})")
    ap.add_argument("-o", "--output", default=None,
                    help="Папка для результатов (по умолчанию: <input>/_РАСКАДРОВКА)")
    ap.add_argument("-i", "--interval", type=float, default=2.0,
                    help="Интервал между кадрами в секундах (по умолчанию 2)")
    ap.add_argument("--cols", type=int, default=5, help="Колонок в PDF-сетке (по умолчанию 5)")
    ap.add_argument("--rows", type=int, default=6, help="Строк в PDF-сетке (по умолчанию 6 -> 30 кадров/страница = 1 минута)")
    ap.add_argument("--no-pdf", action="store_true", help="Не создавать PDF")
    ap.add_argument("--no-zip", action="store_true", help="Не создавать ZIP")
    args = ap.parse_args()

    check_ffmpeg()

    in_dir = Path(args.input).expanduser()
    if not in_dir.is_dir():
        die(f"Папка не найдена: {in_dir}")

    out_root = Path(args.output).expanduser() if args.output else in_dir / "_РАСКАДРОВКА"
    out_root.mkdir(parents=True, exist_ok=True)

    videos = sorted(p for p in in_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in VIDEO_EXTS)
    if not videos:
        die(f"В папке нет видеофайлов: {in_dir}\nПоддерживаемые форматы: {', '.join(sorted(VIDEO_EXTS))}")

    print(f"Найдено видео: {len(videos)}")
    print(f"Интервал: каждые {args.interval:g} c  ->  {round(60/args.interval)} кадров/минуту")
    print(f"Результаты: {out_root}")

    for video in videos:
        process_video(video, out_root, args.interval, args.cols, args.rows,
                      make_pdf=not args.no_pdf, make_zip=not args.no_zip)

    print(f"\n✅ Готово. Всё лежит в: {out_root}")


if __name__ == "__main__":
    main()
