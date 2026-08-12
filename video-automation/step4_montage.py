#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
СКРИПТ 4 — МОНТАЖ ГОТОВОГО РОЛИКА.

Собирает ролик ровно по структуре вашего референса IMG_1645.MOV:

    [1] оживлённое фото + надпись     ~1.7-2.8 с   (в референсе 1.7 с)
    [2] видеосреднее.mp4              вся длина    (в референсе 3.3 с)
    [3] итоговое фото + медленный зум ~1.6-2.6 с   (в референсе 1.65 с)

Плюс:
  * надпись белая с чёрной обводкой и мягкой тенью, по центру, в верхней трети —
    как в референсе; размер шрифта подбирается автоматически под длину текста;
  * переход на финальное фото — плавный кроссфейд;
  * музыка берётся случайная из папки «мелодии», обрезается С КОНЦА
    (начало не трогаем) под длину ролика, в конце фейд;
  * длительности каждый раз слегка разные, чтобы ролики не были под копирку.

Запуск:
    python3 step4_montage.py                # смонтировать все готовые идеи
    python3 step4_montage.py --idea 7
    python3 step4_montage.py --limit 3
    python3 step4_montage.py --no-crossfade

Нужен ffmpeg:  brew install ffmpeg
"""

from __future__ import annotations

import argparse
import random
import shutil
import subprocess
import tempfile
from pathlib import Path

import config
from ideas_store import (
    STATUS_DONE,
    STATUS_VIDEO,
    Idea,
    load_ideas,
    save_ideas,
    stamp_today,
)

MUSIC_EXTENSIONS = (".mp3", ".m4a", ".wav", ".aac", ".ogg", ".flac")


# =============================================================================
# FFMPEG ХЕЛПЕРЫ
# =============================================================================

def require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise SystemExit("Не найден ffmpeg. Установите: brew install ffmpeg")


def run_ffmpeg(args: list[str]) -> None:
    result = subprocess.run(
        ["ffmpeg", "-y", "-v", "error", *args],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg упал:\n{result.stderr.strip()[:2000]}")


def media_duration(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True,
    )
    return float(result.stdout.strip())


def normalize_filter() -> str:
    """Приведение любого исходника к вертикали 1080x1920/30fps с центральным кропом."""
    return (
        f"scale={config.VIDEO_W}:{config.VIDEO_H}:force_original_aspect_ratio=increase,"
        f"crop={config.VIDEO_W}:{config.VIDEO_H},"
        f"fps={config.VIDEO_FPS},setsar=1,format=yuv420p"
    )


# =============================================================================
# НАДПИСЬ
# =============================================================================

# Средняя ширина символа в долях кегля для жирного гротеска с кириллицей.
# Используется только если недоступен PIL — намеренно с запасом, чтобы
# строка гарантированно не упёрлась в край кадра.
CHAR_WIDTH_FACTOR = 0.58


def measure_text_width(text: str, font_path_str: str, size: int) -> float:
    """Ширина строки в пикселях. Точно через PIL, иначе — оценка с запасом."""
    try:
        from PIL import ImageFont  # noqa: PLC0415 — необязательная зависимость

        font = ImageFont.truetype(font_path_str, size)
        return font.getlength(text)
    except Exception:
        return len(text) * size * CHAR_WIDTH_FACTOR


MAX_CAPTION_LINES = 3


def split_balanced(words: list[str], lines_count: int, width_of) -> list[str]:
    """Делит слова ровно на lines_count строк, минимизируя ширину самой широкой.

    Классическая задача о разбиении на отрезки: dp[i][k] — минимально возможная
    ширина самой широкой строки, если первые i слов разложены на k строк.
    Слов в надписи мало, поэтому перебор дешёвый.
    """
    n = len(words)
    if lines_count >= n:
        return words[:]

    infinity = float("inf")
    dp = [[infinity] * (lines_count + 1) for _ in range(n + 1)]
    cut = [[0] * (lines_count + 1) for _ in range(n + 1)]
    dp[0][0] = 0.0

    for i in range(1, n + 1):
        for k in range(1, lines_count + 1):
            for j in range(k - 1, i):
                if dp[j][k - 1] == infinity:
                    continue
                candidate = max(dp[j][k - 1], width_of(" ".join(words[j:i])))
                if candidate < dp[i][k]:
                    dp[i][k] = candidate
                    cut[i][k] = j

    lines: list[str] = []
    i, k = n, lines_count
    while k > 0:
        j = cut[i][k]
        lines.append(" ".join(words[j:i]))
        i, k = j, k - 1
    lines.reverse()
    return lines


def fit_caption(caption: str, font_path_str: str) -> tuple[list[str], int]:
    """Подбирает кегль и разбивку: минимум строк, максимум читаемый размер."""
    words = caption.split()
    if not words:
        return [], config.FONT_SIZE_MAX

    max_width = config.VIDEO_W * config.TEXT_MAX_WIDTH_RATIO

    # Идём от крупного кегля к мелкому и от одной строки к трём:
    # сначала пробуем самый крупный шрифт на минимальном числе строк.
    for size in range(config.FONT_SIZE_MAX, 33, -2):
        def width_of(text: str, _size: int = size) -> float:
            return measure_text_width(text, font_path_str, _size)

        for lines_count in range(1, MAX_CAPTION_LINES + 1):
            lines = split_balanced(words, lines_count, width_of)
            if lines and max(width_of(line) for line in lines) <= max_width:
                return lines, size

    # Совсем длинная надпись — минимальный кегль, три строки
    size = 34
    return split_balanced(
        words, MAX_CAPTION_LINES, lambda t: measure_text_width(t, font_path_str, size)
    ), size


def build_caption_filters(caption: str, work_dir: Path) -> str:
    """Строит цепочку drawtext — по одному фильтру на строку, каждая по центру.

    Отдельный drawtext на строку нужен, чтобы каждая строка центрировалась
    сама по себе (в ffmpeg 6 многострочный блок центрируется целиком,
    и короткие строки прижимаются влево).
    """
    caption = (caption or "").strip()
    if not caption:
        return ""

    font = config.font_path()
    lines, size = fit_caption(caption, font)

    line_height = size * config.TEXT_LINE_SPACING
    top = config.VIDEO_H * config.TEXT_TOP_RATIO

    filters = []
    for index, line in enumerate(lines):
        # Текст кладём в файл — так не нужно экранировать кавычки и спецсимволы
        line_file = work_dir / f"caption_{index}.txt"
        line_file.write_text(line, encoding="utf-8")
        y = int(top + index * line_height)
        filters.append(
            f"drawtext=fontfile='{font}':textfile='{line_file}'"
            f":fontsize={size}:fontcolor=white"
            f":borderw={max(4, size // 12)}:bordercolor=black@0.92"
            f":shadowcolor=black@0.45:shadowx=0:shadowy={max(2, size // 22)}"
            f":x=(w-text_w)/2:y={y}"
        )
    return ",".join(filters)


# =============================================================================
# СЕГМЕНТЫ
# =============================================================================

def build_hook(video_path: Path, caption: str, seconds: float, work_dir: Path) -> Path:
    """Часть 1: оживлённое фото + надпись."""
    out = work_dir / "seg1.mp4"
    source_duration = media_duration(video_path)
    take = min(seconds, source_duration)

    chain = normalize_filter()
    caption_filters = build_caption_filters(caption, work_dir)
    if caption_filters:
        chain = f"{chain},{caption_filters}"

    run_ffmpeg([
        "-i", str(video_path), "-t", f"{take:.3f}",
        "-vf", chain, "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        str(out),
    ])
    return out


def build_middle(video_path: Path, max_seconds: float, work_dir: Path) -> Path:
    """Часть 2: постоянный скринкаст бота."""
    out = work_dir / "seg2.mp4"
    take = min(max_seconds, media_duration(video_path))
    run_ffmpeg([
        "-i", str(video_path), "-t", f"{take:.3f}",
        "-vf", normalize_filter(), "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        str(out),
    ])
    return out


def build_final(photo_path: Path, seconds: float, work_dir: Path) -> Path:
    """Часть 3: итоговое фото с медленным зумом."""
    out = work_dir / "seg3.mp4"
    frames = int(seconds * config.VIDEO_FPS)
    run_ffmpeg([
        "-loop", "1", "-i", str(photo_path), "-t", f"{seconds:.3f}",
        "-vf",
        # Увеличиваем перед zoompan, иначе зум даёт заметное дрожание
        f"scale={config.VIDEO_W * 2}:-2,"
        f"zoompan=z='min(1+0.0011*on,1.10)'"
        f":x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
        f":d={frames}:s={config.VIDEO_W}x{config.VIDEO_H}:fps={config.VIDEO_FPS},"
        f"setsar=1,format=yuv420p",
        "-frames:v", str(frames),
        "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        str(out),
    ])
    return out


def concat_segments(segments: list[Path], work_dir: Path) -> Path:
    """Простая склейка встык (как в референсе — жёсткие склейки)."""
    out = work_dir / "joined.mp4"
    list_file = work_dir / "concat.txt"
    list_file.write_text("".join(f"file '{p}'\n" for p in segments), encoding="utf-8")
    run_ffmpeg(["-f", "concat", "-safe", "0", "-i", str(list_file), "-c", "copy", str(out)])
    return out


def join_with_crossfade(first: Path, second: Path, work_dir: Path, fade: float) -> Path:
    """Склейка с плавным переходом (используется перед финальным фото)."""
    out = work_dir / "joined_xfade.mp4"
    offset = max(0.0, media_duration(first) - fade)
    run_ffmpeg([
        "-i", str(first), "-i", str(second),
        "-filter_complex",
        f"[0:v][1:v]xfade=transition=fade:duration={fade:.2f}:offset={offset:.3f},"
        f"format=yuv420p[v]",
        "-map", "[v]", "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        str(out),
    ])
    return out


# =============================================================================
# МУЗЫКА
# =============================================================================

def pick_music(music_dir: Path, rng: random.Random) -> Path | None:
    """Случайная мелодия из папки."""
    if not music_dir.exists():
        print(f"[WARN] Папка с мелодиями не найдена: {music_dir} — ролик будет без музыки.")
        return None
    tracks = sorted(
        p for p in music_dir.iterdir()
        if p.is_file() and p.suffix.lower() in MUSIC_EXTENSIONS and not p.name.startswith(".")
    )
    if not tracks:
        print(f"[WARN] В папке нет аудиофайлов: {music_dir} — ролик будет без музыки.")
        return None
    return rng.choice(tracks)


def add_music(video_path: Path, music_path: Path, out_path: Path) -> None:
    """Подкладывает музыку: начало трека сохраняем, лишнее режем с конца."""
    duration = media_duration(video_path)
    fade_start = max(0.0, duration - config.MUSIC_FADE_OUT)
    run_ffmpeg([
        "-i", str(video_path),
        "-stream_loop", "-1", "-i", str(music_path),   # на случай, если трек короче ролика
        "-filter_complex",
        f"[1:a]atrim=0:{duration:.3f},asetpts=PTS-STARTPTS,"
        f"afade=t=out:st={fade_start:.3f}:d={config.MUSIC_FADE_OUT}[a]",
        "-map", "0:v", "-map", "[a]",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest",
        str(out_path),
    ])


# =============================================================================
# СБОРКА ОДНОГО РОЛИКА
# =============================================================================

def _safe_name(text: str) -> str:
    keep = [c if (c.isalnum() or c in " -_") else "_" for c in text]
    return "".join(keep).strip().replace(" ", "_")[:40] or "reel"


def assemble(idea: Idea, rng: random.Random, use_crossfade: bool) -> Path:
    hook_video = Path(idea.video_file).expanduser()
    final_photo = Path(idea.photo_file).expanduser()

    if not hook_video.exists():
        raise RuntimeError(f"нет оживлённого видео: {hook_video}")
    if not final_photo.exists():
        raise RuntimeError(f"нет итогового фото: {final_photo}")
    if not config.MIDDLE_VIDEO.exists():
        raise RuntimeError(f"нет средней части: {config.MIDDLE_VIDEO}")

    hook_seconds = rng.uniform(*config.HOOK_SECONDS_RANGE)
    final_seconds = rng.uniform(*config.FINAL_SECONDS_RANGE)

    out_path = config.OUTPUT_DIR / f"{idea.number:03d}_{_safe_name(idea.title)}.mp4"

    with tempfile.TemporaryDirectory(prefix="reel_") as tmp:
        work_dir = Path(tmp)

        print(f"      [1/3] хук {hook_seconds:.1f}с + надпись")
        seg1 = build_hook(hook_video, idea.caption, hook_seconds, work_dir)

        print("      [2/3] средняя часть")
        seg2 = build_middle(config.MIDDLE_VIDEO, config.MIDDLE_MAX_SECONDS, work_dir)

        print(f"      [3/3] финальное фото {final_seconds:.1f}с")
        seg3 = build_final(final_photo, final_seconds, work_dir)

        if use_crossfade:
            head = concat_segments([seg1, seg2], work_dir)
            joined = join_with_crossfade(head, seg3, work_dir, config.CROSSFADE_SECONDS)
        else:
            joined = concat_segments([seg1, seg2, seg3], work_dir)

        music = pick_music(config.MUSIC_DIR, rng)
        if music:
            print(f"      музыка: {music.name}")
            add_music(joined, music, out_path)
        else:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(joined, out_path)

    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Монтаж готовых роликов по структуре референса.")
    parser.add_argument("--limit", type=int, default=0, help="Максимум роликов за запуск (0 = все).")
    parser.add_argument("--idea", type=int, default=0, help="Смонтировать только идею с этим номером.")
    parser.add_argument("--no-crossfade", action="store_true", help="Без перехода на финальное фото.")
    parser.add_argument("--seed", type=int, default=0, help="Фиксировать случайность (для повторяемости).")
    parser.add_argument("--ideas-file", default=str(config.IDEAS_FILE))
    args = parser.parse_args()

    require_ffmpeg()
    config.ensure_dirs()

    ideas_path = Path(args.ideas_file).expanduser()
    ideas = load_ideas(ideas_path)
    if not ideas:
        raise SystemExit(f"В файле нет идей: {ideas_path}")

    if args.idea:
        queue = [i for i in ideas if i.number == args.idea]
        if not queue:
            raise SystemExit(f"Идея №{args.idea} не найдена.")
    else:
        queue = [i for i in ideas if i.status == STATUS_VIDEO and i.video_file]
        if args.limit:
            queue = queue[: args.limit]

    if not queue:
        print("Нет идей с оживлённым видео. Сначала запустите: python3 step3_animate.py")
        return

    rng = random.Random(args.seed) if args.seed else random.Random()

    print(f"Средняя часть: {config.MIDDLE_VIDEO}")
    print(f"Мелодии:       {config.MUSIC_DIR}")
    print(f"К монтажу:     {len(queue)} шт.\n")

    done = 0
    for idea in queue:
        print(f"[{idea.number:03d}] {idea.title} — «{idea.caption}»")
        try:
            out_path = assemble(idea, rng, use_crossfade=not args.no_crossfade)
        except Exception as exc:  # noqa: BLE001
            print(f"      ОШИБКА: {exc}\n")
            continue

        idea.reel_file = str(out_path)
        idea.status = STATUS_DONE            # идея отработана
        idea.date = idea.date or stamp_today()
        save_ideas(ideas_path, ideas, make_backup=False)
        done += 1
        print(f"      ГОТОВО: {out_path.name} ({media_duration(out_path):.2f} с)\n")

    print(f"Смонтировано: {done} из {len(queue)}")
    print(f"Папка: {config.OUTPUT_DIR}")
    if done:
        print(
            "\nПосле публикации впишите просмотры в ИДЕИ.txt (поле ПРОСМОТРЫ)\n"
            "и запустите python3 step1_ideas.py — новые идеи будут по сработавшим паттернам."
        )


if __name__ == "__main__":
    main()
