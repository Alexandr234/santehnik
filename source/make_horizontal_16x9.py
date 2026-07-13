# -*- coding: utf-8 -*-
"""
ГОРИЗОНТАЛИЗАЦИЯ: находит ВЕРТИКАЛЬНЫЕ картинки и видео и переделывает их в ГОРИЗОНТАЛЬНЫЕ (16:9).

Зачем:
  Часть картинок/видео сгенерировалась вертикальными (портрет), а нужно, чтобы ВСЁ было
  горизонтальным (ландшафт 16:9). Этот скрипт:

  ФАЗА 1 — КАРТИНКИ:
    • сканирует КАРТИНКИ/, определяет размеры;
    • вертикальные (высота > ширины) ПЕРЕГЕНЕРИРУЕТ горизонтально (16:9) из промпта сцены
      через flower_image_generate (провайдер flower), с лёгкой лестницей безопасности:
      как есть -> санитайз -> анонимизация -> безопасный fallback;
    • если промпта для сцены нет — по возможности делает горизонтальный центр-кроп (Pillow),
      иначе пропускает с предупреждением (чтобы не выдумывать чужой контент);
    • оригинал бэкапится в КАРТИНКИ/_vertical_backup/.

  ФАЗА 2 — ВИДЕО:
    • сканирует ВИДЕО/, определяет размеры (ffprobe -> иначе разбор mp4/tkhd);
    • вертикальные видео (а также видео, чья картинка была только что перегенерирована)
      ПЕРЕДЕЛЫВАЕТ горизонтально: оживляет уже ГОРИЗОНТАЛЬНУЮ картинку через
      flower_video_from_image (Veo 3.1) с aspect_ratio 16:9;
    • старое видео бэкапится в ВИДЕО/_vertical_backup/.

Переиспользует flower-логику (генерация, промпты, анонимизация) из
video_repair_characters_100percent.py, лежащего рядом.

Запуск:
   export FAST_GEN_API_KEY="ТВОЙ_КЛЮЧ"
   python make_horizontal_16x9.py                 # картинки + видео
   python make_horizontal_16x9.py --dry-run       # только показать, что вертикальное
   python make_horizontal_16x9.py --images-only
   python make_horizontal_16x9.py --videos-only
   python make_horizontal_16x9.py --fix-square     # чинить и квадратные тоже
   python make_horizontal_16x9.py --force-video-all # переоживить ВСЕ видео из горизонтальных картинок
"""

from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Tuple

# Импортируем flower-логику из соседнего скрипта.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import video_repair_characters_100percent as rep  # noqa: E402


# =========================================================
# НАСТРОЙКИ
# =========================================================

TARGET_ASPECT_RATIO = os.getenv("FAST_GEN_TARGET_ASPECT_RATIO", "16:9")
MAX_WORKERS = int(os.getenv("FAST_GEN_HORIZ_WORKERS", "4"))

IMAGE_BACKUP_DIRNAME = "_vertical_backup"
VIDEO_BACKUP_DIRNAME = "_vertical_backup"

# Разрешить горизонтальный центр-кроп картинки (Pillow), если нет промпта для перегенерации.
HORIZ_CROP_FALLBACK = os.getenv("FAST_GEN_HORIZ_CROP_FALLBACK", "1").strip().lower() in {"1", "true", "yes", "on"}


def log(msg: str) -> None:
    rep.log(msg)


def parse_ratio(text: str) -> Tuple[int, int]:
    try:
        w, h = text.split(":")
        return int(w), int(h)
    except Exception:
        return 16, 9


# =========================================================
# ОПРЕДЕЛЕНИЕ РАЗМЕРОВ КАРТИНКИ (Pillow -> разбор заголовков)
# =========================================================

def _img_size_from_header(data: bytes) -> Optional[Tuple[int, int]]:
    # PNG
    if data[:8] == b"\x89PNG\r\n\x1a\n" and len(data) >= 24:
        w = int.from_bytes(data[16:20], "big")
        h = int.from_bytes(data[20:24], "big")
        if w and h:
            return w, h
    # GIF
    if data[:6] in (b"GIF87a", b"GIF89a") and len(data) >= 10:
        w = int.from_bytes(data[6:8], "little")
        h = int.from_bytes(data[8:10], "little")
        if w and h:
            return w, h
    # JPEG
    if data[:2] == b"\xff\xd8":
        i, n = 2, len(data)
        sof_markers = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}
        while i < n - 1:
            if data[i] != 0xFF:
                i += 1
                continue
            marker = data[i + 1]
            i += 2
            if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7 or marker == 0x01:
                continue
            if i + 2 > n:
                break
            seglen = int.from_bytes(data[i:i + 2], "big")
            if marker in sof_markers and i + 7 <= n:
                h = int.from_bytes(data[i + 3:i + 5], "big")
                w = int.from_bytes(data[i + 5:i + 7], "big")
                if w and h:
                    return w, h
            i += seglen
    # WEBP
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP" and len(data) >= 30:
        fmt = data[12:16]
        if fmt == b"VP8X":
            w = 1 + int.from_bytes(data[24:27], "little")
            h = 1 + int.from_bytes(data[27:30], "little")
            return w, h
        if fmt == b"VP8 ":
            idx = data.find(b"\x9d\x01\x2a", 20)
            if idx != -1 and idx + 7 <= len(data):
                w = int.from_bytes(data[idx + 3:idx + 5], "little") & 0x3FFF
                h = int.from_bytes(data[idx + 5:idx + 7], "little") & 0x3FFF
                if w and h:
                    return w, h
        if fmt == b"VP8L" and len(data) >= 25 and data[20] == 0x2F:
            bits = int.from_bytes(data[21:25], "little")
            w = (bits & 0x3FFF) + 1
            h = ((bits >> 14) & 0x3FFF) + 1
            return w, h
    return None


def get_image_size(path: Path) -> Optional[Tuple[int, int]]:
    try:
        from PIL import Image
        with Image.open(path) as im:
            w, h = im.size
            if w and h:
                return int(w), int(h)
    except Exception:
        pass
    try:
        data = path.read_bytes()
    except Exception:
        return None
    return _img_size_from_header(data)


# =========================================================
# ОПРЕДЕЛЕНИЕ РАЗМЕРОВ ВИДЕО (ffprobe -> разбор mp4/tkhd)
# =========================================================

def _video_size_ffprobe(path: Path) -> Optional[Tuple[int, int]]:
    exe = shutil.which("ffprobe")
    if not exe:
        return None
    try:
        out = subprocess.run(
            [exe, "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height,side_data_list:stream_tags=rotate",
             "-of", "json", str(path)],
            capture_output=True, text=True, timeout=60,
        )
        data = json.loads(out.stdout or "{}")
        streams = data.get("streams") or []
        if not streams:
            return None
        s = streams[0]
        w = int(s.get("width") or 0)
        h = int(s.get("height") or 0)
        if not (w and h):
            return None
        rot = 0
        tags = s.get("tags") or {}
        if "rotate" in tags:
            try:
                rot = int(tags["rotate"])
            except Exception:
                rot = 0
        for sd in (s.get("side_data_list") or []):
            if "rotation" in sd:
                try:
                    rot = int(sd["rotation"])
                except Exception:
                    pass
        if abs(rot) % 180 == 90:
            w, h = h, w
        return w, h
    except Exception:
        return None


def _read_boxes(buf: io.BytesIO, end: int):
    boxes = []
    while buf.tell() < end:
        start = buf.tell()
        hdr = buf.read(8)
        if len(hdr) < 8:
            break
        size = int.from_bytes(hdr[:4], "big")
        typ = hdr[4:8]
        header = 8
        if size == 1:
            size = int.from_bytes(buf.read(8), "big")
            header = 16
        elif size == 0:
            size = end - start
        if size < header:
            break
        boxes.append((typ, start, size, header))
        buf.seek(start + size)
    return boxes


def _parse_tkhd(buf: io.BytesIO, start: int, header: int) -> Optional[Tuple[int, int]]:
    buf.seek(start + header)
    vf = buf.read(4)
    if len(vf) < 4:
        return None
    version = vf[0]
    # creation/mod/trackid/reserved/duration
    buf.seek(buf.tell() + (32 if version == 1 else 20))
    buf.seek(buf.tell() + 8)   # reserved (2 x uint32)
    buf.seek(buf.tell() + 8)   # layer, altgroup, volume, reserved
    matrix = [int.from_bytes(buf.read(4), "big", signed=True) for _ in range(9)]
    wraw = int.from_bytes(buf.read(4), "big")
    hraw = int.from_bytes(buf.read(4), "big")
    w = wraw / 65536.0
    h = hraw / 65536.0
    if w <= 0 or h <= 0:
        return None
    # Поворот из матрицы: 90/270 -> меняем стороны местами.
    a, b, c, d = matrix[0], matrix[1], matrix[3], matrix[4]
    if a == 0 and d == 0 and b != 0 and c != 0:
        w, h = h, w
    return int(round(w)), int(round(h))


def _video_size_mp4(path: Path) -> Optional[Tuple[int, int]]:
    try:
        raw = path.read_bytes()
    except Exception:
        return None
    buf = io.BytesIO(raw)
    end = len(raw)
    top = _read_boxes(buf, end)
    moov = next((bx for bx in top if bx[0] == b"moov"), None)
    if not moov:
        return None
    _typ, mstart, msize, mheader = moov
    buf.seek(mstart + mheader)
    best: Optional[Tuple[int, int]] = None
    for typ, tstart, tsize, theader in _read_boxes(buf, mstart + msize):
        if typ != b"trak":
            continue
        buf.seek(tstart + theader)
        for btyp, bstart, bsize, bheader in _read_boxes(buf, tstart + tsize):
            if btyp == b"tkhd":
                dims = _parse_tkhd(buf, bstart, bheader)
                if dims and dims[0] and dims[1]:
                    area = dims[0] * dims[1]
                    if best is None or area > best[0] * best[1]:
                        best = dims
                buf.seek(tstart + theader)  # вернуться для следующих боксов trak
    return best


def get_video_size(path: Path) -> Optional[Tuple[int, int]]:
    dims = _video_size_ffprobe(path)
    if dims:
        return dims
    return _video_size_mp4(path)


# =========================================================
# ОРИЕНТАЦИЯ
# =========================================================

def is_vertical(size: Optional[Tuple[int, int]], fix_square: bool) -> Optional[bool]:
    """True если портрет (нужно чинить). None если размер неизвестен."""
    if not size:
        return None
    w, h = size
    if not (w and h):
        return None
    if fix_square:
        return h >= w  # всё, что не строго ландшафт
    return h > w


# =========================================================
# BACKUP + ПЕРЕГЕНЕРАЦИЯ
# =========================================================

def backup_file(path: Path, backup_dirname: str) -> None:
    if not path.exists():
        return
    backup_dir = path.parent / backup_dirname
    backup_dir.mkdir(parents=True, exist_ok=True)
    dest = backup_dir / path.name
    if not dest.exists():
        try:
            shutil.copy2(path, dest)
        except Exception as e:
            log(f"      [backup] не смог сохранить оригинал {path.name}: {e}")


def crop_image_to_landscape(path: Path, ratio: Tuple[int, int]) -> bool:
    """Горизонтальный центр-кроп через Pillow (fallback, когда нет промпта)."""
    try:
        from PIL import Image
    except Exception:
        return False
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            w, h = im.size
            tr = ratio[0] / ratio[1]
            new_w, new_h = w, int(round(w / tr))
            if new_h > h:
                new_h = h
                new_w = int(round(h * tr))
            left = max(0, (w - new_w) // 2)
            top = max(0, (h - new_h) // 2)
            im.crop((left, top, left + new_w, top + new_h)).save(path)
        return True
    except Exception as e:
        log(f"      [crop] не удалось: {e}")
        return False


def regen_image_horizontal(image_path: Path, scene, known_names) -> Tuple[Optional[str], Optional[str]]:
    """Перегенерирует картинку горизонтально из промпта сцены (rep.IMAGE_ASPECT_RATIO = target)."""
    base = rep.scene_image_base_prompt(scene)
    aliases = list(scene.aliases) if scene else []
    ladder = [
        ("as-is", base),
        ("sanitized", rep.heuristic_sanitize(base, 1)),
        ("anonymized", rep.rewrite_anonymize(base, aliases, known_names)),
        ("safe", rep.SAFE_IMAGE_FALLBACK),
    ]
    for mode, prompt in ladder:
        try:
            gen = rep.generate_image_attempt(prompt, image_path)
            return mode, gen
        except rep.PermanentError as e:
            log(f"      [img {mode}] заблокировано: {e} — эскалирую")
        except rep.TransientError as e:
            log(f"      [img {mode}] не удалось: {e} — эскалирую")
    return None, None


def regen_video_horizontal(image_path: Path, video_path: Path) -> Tuple[Optional[str], Optional[str]]:
    """Оживляет горизонтальную картинку -> горизонтальное видео (rep.ASPECT_RATIO = target)."""
    try:
        uri = rep.image_to_data_uri(image_path)
    except Exception as e:
        log(f"      [vid] картинка не читается: {e}")
        return None, None
    for vmode, vprompt in (("motion-preserve", rep.MOTION_PRESERVE_PROMPT),
                           ("motion-minimal", rep.MOTION_MINIMAL_PROMPT)):
        try:
            gen = rep.generate_video_attempt(vprompt, uri, video_path)
            return vmode, gen
        except Exception as e:
            log(f"      [vid {vmode}] не удалось: {e}")
    return None, None


# =========================================================
# СКАН + ОБРАБОТКА
# =========================================================

def list_media(directory: Path, exts) -> list:
    if not directory.exists():
        return []
    items = [p for p in directory.iterdir()
             if p.is_file() and p.suffix.lower() in exts and not p.name.startswith("_")]
    items.sort(key=rep.path_sort_key)
    return items


def process_image(image_path: Path, order: int, scenes, known_names,
                  ratio: Tuple[int, int], fix_square: bool, dry_run: bool) -> dict:
    size = get_image_size(image_path)
    vert = is_vertical(size, fix_square)
    idx = rep.scene_index_for(image_path, order)
    if vert is None:
        log(f"[IMG ?] {image_path.name}: размер не определён — пропускаю")
        return {"file": str(image_path), "status": "unknown_size", "kind": "image"}
    if not vert:
        return {"file": str(image_path), "status": "already_horizontal", "kind": "image", "size": size}

    log(f"[IMG ↕] {image_path.name} {size[0]}x{size[1]} — вертикальная, делаю горизонтальной")
    if dry_run:
        return {"file": str(image_path), "status": "would_fix", "kind": "image", "size": size, "index": idx}

    scene = scenes.get(idx) or scenes.get(order)
    has_prompt = bool(scene and (scene.image_body or scene.body))

    backup_file(image_path, IMAGE_BACKUP_DIRNAME)

    if has_prompt:
        mode, gen = regen_image_horizontal(image_path, scene, known_names)
        if mode:
            new = get_image_size(image_path)
            log(f"    OK картинка горизонтальна ({mode}) {new}: {image_path.name}")
            return {"file": str(image_path), "status": "fixed", "kind": "image", "mode": mode,
                    "index": idx, "generation_id": gen, "new_size": new}
        log(f"    не удалось перегенерировать {image_path.name}")
        return {"file": str(image_path), "status": "failed", "kind": "image", "index": idx}

    # Нет промпта — по возможности кроп.
    if HORIZ_CROP_FALLBACK and crop_image_to_landscape(image_path, ratio):
        new = get_image_size(image_path)
        log(f"    OK картинка обрезана до горизонтали {new}: {image_path.name} (нет промпта — кроп)")
        return {"file": str(image_path), "status": "cropped", "kind": "image", "index": idx, "new_size": new}

    log(f"    нет промпта и кроп недоступен — пропускаю {image_path.name}")
    return {"file": str(image_path), "status": "skipped_no_prompt", "kind": "image", "index": idx}


def process_video(video_path: Path, order: int, image_by_stem, fixed_stems, scenes,
                  ratio: Tuple[int, int], fix_square: bool, dry_run: bool,
                  force_all: bool) -> dict:
    size = get_video_size(video_path)
    vert = is_vertical(size, fix_square)
    stem = video_path.stem
    image_path = image_by_stem.get(stem)

    need = force_all or (stem in fixed_stems) or (vert is True)
    reason = ("force" if force_all else
              "image_regenerated" if stem in fixed_stems else
              "vertical" if vert is True else "")

    if vert is None and not need:
        log(f"[VID ?] {video_path.name}: размер не определён — пропускаю (нет др. причины)")
        return {"file": str(video_path), "status": "unknown_size", "kind": "video"}
    if not need:
        return {"file": str(video_path), "status": "already_horizontal", "kind": "video", "size": size}

    label = f"{size[0]}x{size[1]}" if size else "размер?"
    log(f"[VID ↕] {video_path.name} {label} — переделываю горизонтально (причина: {reason})")
    if dry_run:
        return {"file": str(video_path), "status": "would_fix", "kind": "video", "size": size, "reason": reason}

    if not image_path or not image_path.exists():
        log(f"    нет исходной картинки для {video_path.name} — не могу переоживить")
        return {"file": str(video_path), "status": "skipped_no_image", "kind": "video"}

    # Убедимся, что картинка горизонтальна (если вдруг ещё вертикальная — фиксируем её).
    isize = get_image_size(image_path)
    if is_vertical(isize, fix_square) is True:
        idx = rep.scene_index_for(image_path, order)
        scene = scenes.get(idx) or scenes.get(order)
        if scene and (scene.image_body or scene.body):
            backup_file(image_path, IMAGE_BACKUP_DIRNAME)
            regen_image_horizontal(image_path, scene, known_names=rep.load_known_names())
        elif HORIZ_CROP_FALLBACK:
            backup_file(image_path, IMAGE_BACKUP_DIRNAME)
            crop_image_to_landscape(image_path, ratio)

    backup_file(video_path, VIDEO_BACKUP_DIRNAME)
    vmode, gen = regen_video_horizontal(image_path, video_path)
    if vmode:
        new = get_video_size(video_path)
        log(f"    OK видео горизонтально ({vmode}) {new}: {video_path.name}")
        return {"file": str(video_path), "status": "fixed", "kind": "video", "mode": vmode,
                "generation_id": gen, "new_size": new, "reason": reason}
    log(f"    не удалось переоживить {video_path.name}")
    return {"file": str(video_path), "status": "failed", "kind": "video", "reason": reason}


# =========================================================
# MAIN
# =========================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Сделать все вертикальные картинки и видео горизонтальными (16:9) через flower."
    )
    parser.add_argument("--images-dir", default=str(rep.IMAGES_DIR))
    parser.add_argument("--videos-dir", default=str(rep.VIDEOS_DIR))
    parser.add_argument("--api-base", default=rep.BASE_URL)
    parser.add_argument("--aspect-ratio", default=TARGET_ASPECT_RATIO, help="Целевое соотношение (по умолч. 16:9)")
    parser.add_argument("--images-only", action="store_true")
    parser.add_argument("--videos-only", action="store_true")
    parser.add_argument("--fix-square", action="store_true", help="Чинить и квадратные (не строго ландшафт).")
    parser.add_argument("--force-video-all", action="store_true",
                        help="Переоживить ВСЕ видео из горизонтальных картинок (не только вертикальные).")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="Максимум файлов за запуск на фазу (0 = все).")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    args = parser.parse_args()

    if not rep.API_KEY:
        raise SystemExit('Не найден API ключ. export FAST_GEN_API_KEY="ТВОЙ_КЛЮЧ"')

    ratio = parse_ratio(args.aspect_ratio)
    target = f"{ratio[0]}:{ratio[1]}"

    # Прокидываем настройки в переиспользуемый модуль.
    rep.BASE_URL = args.api_base
    rep.IMAGES_DIR = Path(args.images_dir).expanduser()
    rep.VIDEOS_DIR = Path(args.videos_dir).expanduser()
    rep.ASPECT_RATIO = target          # видео будет 16:9
    rep.IMAGE_ASPECT_RATIO = target    # картинка будет 16:9

    images_dir = rep.IMAGES_DIR
    videos_dir = rep.VIDEOS_DIR

    scenes = rep.load_scenes()
    known_names = rep.load_known_names()

    do_images = not args.videos_only
    do_videos = not args.images_only

    log("ГОРИЗОНТАЛИЗАЦИЯ (flower / Veo 3.1)")
    log(f"Картинки: {images_dir}")
    log(f"Видео:    {videos_dir}")
    log(f"Целевое соотношение: {target}")
    log(f"Чинить квадратные: {'да' if args.fix_square else 'нет'}")
    log(f"ffprobe: {'есть' if shutil.which('ffprobe') else 'нет (разбираю mp4 сам)'}")

    fixed_stems = set()
    image_results = []

    # ---- ФАЗА 1: КАРТИНКИ ----
    if do_images:
        images = list_media(images_dir, rep.IMAGE_EXTS)
        log(f"\n===== ФАЗА 1: КАРТИНКИ ({len(images)}) =====")
        todo = list(enumerate(images, start=1))
        if args.limit and args.limit > 0 and not args.dry_run:
            # ограничиваем только реальные починки — но проще ограничить общий список
            todo = todo[:args.limit] if args.limit < len(todo) else todo
        workers = max(1, min(args.workers, len(todo) or 1))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(process_image, img, order, scenes, known_names, ratio,
                              args.fix_square, args.dry_run): (order, img) for order, img in todo}
            for f in as_completed(futs):
                try:
                    r = f.result()
                    image_results.append(r)
                    if r.get("status") in ("fixed", "cropped"):
                        fixed_stems.add(Path(r["file"]).stem)
                except Exception as e:
                    log(f"[IMG-FUTURE-ERROR] {e}")

        vert = [r for r in image_results if r.get("status") in ("fixed", "cropped", "would_fix", "failed")]
        log(f"Фаза 1 итог: вертикальных найдено {len(vert)}; "
            f"починено {sum(1 for r in image_results if r.get('status') in ('fixed','cropped'))}; "
            f"не удалось {sum(1 for r in image_results if r.get('status')=='failed')}; "
            f"пропущено без промпта {sum(1 for r in image_results if r.get('status')=='skipped_no_prompt')}.")

    # ---- ФАЗА 2: ВИДЕО ----
    if do_videos:
        images_now = list_media(images_dir, rep.IMAGE_EXTS)
        image_by_stem = {p.stem: p for p in images_now}
        videos = list_media(videos_dir, {".mp4", ".mov", ".webm", ".m4v"})
        log(f"\n===== ФАЗА 2: ВИДЕО ({len(videos)}) =====")
        todo = list(enumerate(videos, start=1))
        if args.limit and args.limit > 0 and not args.dry_run:
            todo = todo[:args.limit] if args.limit < len(todo) else todo
        video_results = []
        workers = max(1, min(args.workers, len(todo) or 1))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(process_video, vid, order, image_by_stem, fixed_stems, scenes,
                              ratio, args.fix_square, args.dry_run, args.force_video_all): vid
                    for order, vid in todo}
            for f in as_completed(futs):
                try:
                    video_results.append(f.result())
                except Exception as e:
                    log(f"[VID-FUTURE-ERROR] {e}")

        log(f"Фаза 2 итог: переделано {sum(1 for r in video_results if r.get('status')=='fixed')}; "
            f"уже горизонтальных {sum(1 for r in video_results if r.get('status')=='already_horizontal')}; "
            f"не удалось {sum(1 for r in video_results if r.get('status')=='failed')}; "
            f"без картинки {sum(1 for r in video_results if r.get('status')=='skipped_no_image')}.")

    log("\nALL DONE")


if __name__ == "__main__":
    main()
