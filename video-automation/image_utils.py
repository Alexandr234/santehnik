#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Работа с пропорциями картинок.

ЗАЧЕМ ЭТО НУЖНО.
Генератор видео получает aspect_ratio 9:16. Если подать ему картинку с другими
пропорциями (например gpt-image-1 отдаёт 1024x1536 — это 2:3), провайдер
НАТЯГИВАЕТ её на 9:16, и лицо вытягивается по вертикали.

Поэтому перед оживлением картинка всегда приводится к 9:16 честным кропом
по центру: пропорции лица сохраняются, обрезаются только края.
"""

from __future__ import annotations

from pathlib import Path

TARGET_W = 9
TARGET_H = 16
TOLERANCE = 0.01   # 2:3 против 9:16 — разница ~0.10, ловится с запасом


def image_size(path: Path) -> tuple[int, int] | None:
    try:
        from PIL import Image  # noqa: PLC0415

        with Image.open(path) as image:
            return image.size
    except Exception:
        return None


def aspect_of(path: Path) -> float | None:
    size = image_size(path)
    if not size or not size[1]:
        return None
    return size[0] / size[1]


def ensure_vertical(path: Path, target: float = TARGET_W / TARGET_H) -> bool:
    """Приводит картинку к 9:16 центральным кропом. Перезаписывает файл.

    Возвращает True, если файл был изменён. Если Pillow нет или пропорции
    уже верные — ничего не делает и возвращает False.
    """
    try:
        from PIL import Image  # noqa: PLC0415
    except ImportError:
        return False

    try:
        with Image.open(path) as image:
            width, height = image.size
            if not width or not height:
                return False

            current = width / height
            if abs(current - target) <= TOLERANCE:
                return False

            if current > target:
                # Слишком широкая — режем по бокам
                new_width = int(round(height * target))
                left = (width - new_width) // 2
                box = (left, 0, left + new_width, height)
            else:
                # Слишком высокая — режем сверху и снизу.
                # Смещаем окно вверх: лицо обычно в верхней половине кадра.
                new_height = int(round(width / target))
                top = int((height - new_height) * 0.35)
                box = (0, top, width, top + new_height)

            cropped = image.crop(box)
            if cropped.mode not in ("RGB", "L"):
                cropped = cropped.convert("RGB")
            cropped.save(path)
            return True
    except Exception:
        return False
