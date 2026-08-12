#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ЕДИНЫЙ КОНФИГ ВСЕГО КОНВЕЙЕРА.

Здесь лежат все пути и все ключи. Больше нигде ключи прописывать не надо —
все четыре скрипта берут их отсюда.

=============================================================================
КЛЮЧИ БЕРУТСЯ ОТТУДА ЖЕ, ОТКУДА ИХ БРАЛИ СТАРЫЕ СКРИПТЫ
=============================================================================
1) OPENAI_API_KEY  — из переменной окружения, ровно как в doc_prompt_pipeline.
2) FAST_GEN_API_KEY — из переменной окружения, ровно как в
   flower_veo31_visual_batch_generator (поддержаны и старые псевдонимы
   FASTGEN_API_KEY / MEDIA_GEN_API_KEY).

То есть если ключи уже прописаны в ~/.zshrc — вписывать ничего не нужно,
всё подхватится само. Если удобнее вписать строкой — можно подставить
значение прямо в кавычки ниже, оно перекроет переменную окружения.
=============================================================================
"""

from __future__ import annotations

import os
from pathlib import Path

# =============================================================================
# КЛЮЧИ — как в старых скриптах, из переменных окружения
# =============================================================================

# Ровно как в doc_prompt_pipeline_universal_ru_de_es_pl.py
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

# Ровно как в flower_veo31_visual_batch_generator_realistic.py
FAST_GEN_API_KEY = (
    os.getenv("FAST_GEN_API_KEY", "").strip()
    or os.getenv("FASTGEN_API_KEY", "").strip()
    or os.getenv("MEDIA_GEN_API_KEY", "").strip()
)


# =============================================================================
# ПУТИ ПРОЕКТА
# =============================================================================

BASE_DIR = Path(
    os.getenv("VITYA_BASE_DIR", "/Users/aleksandrtomilov/Desktop/ВИТЯ АВТОМАТИЗАЦИЯ")
).expanduser()

# Файл с идеями — его читают и дополняют все скрипты
IDEAS_FILE = BASE_DIR / "ИДЕИ.txt"

# Фото вашего лица — эталон внешности для всех генераций
FACE_PHOTO = BASE_DIR / "telegram-cloud-document-2-5463411800257112310.jpg"

# Постоянная средняя часть ролика (скринкаст бота)
MIDDLE_VIDEO = BASE_DIR / "видеосреднее.mp4"

# Папка с мелодиями (1.mp3, 2.mp3, 3.mp3 ...)
MUSIC_DIR = BASE_DIR / "мелодии"

# Куда складываются результаты
OUTPUT_DIR = BASE_DIR / "ГОТОВОЕ"          # смонтированные ролики
PHOTOS_DIR = BASE_DIR / "ФОТО"             # сгенерированные фото
VIDEOS_DIR = BASE_DIR / "ВИДЕО"            # оживлённые фото
CACHE_DIR = BASE_DIR / "_кэш"              # профиль лица, логи, аналитика


# =============================================================================
# МОДЕЛИ OPENAI
# =============================================================================

TEXT_MODEL = os.getenv("VITYA_TEXT_MODEL", "gpt-4o")          # идеи и промпты
VISION_MODEL = os.getenv("VITYA_VISION_MODEL", "gpt-4o")      # разбор лица по фото
OPENAI_IMAGE_MODEL = os.getenv("VITYA_IMAGE_MODEL", "gpt-image-1")


# =============================================================================
# ГЕНЕРАЦИЯ ФОТО
# =============================================================================
# "openai"  — gpt-image-1 с вашим фото как референсом (лучше держит лицо)
# "fastgen" — api.fast-gen.ai, как в вашем рабочем скрипте
IMAGE_BACKEND = os.getenv("VITYA_IMAGE_BACKEND", "openai").strip().lower()

# Вертикальный размер под рилс. gpt-image-1 поддерживает 1024x1536.
OPENAI_IMAGE_SIZE = os.getenv("VITYA_IMAGE_SIZE", "1024x1536")
IMAGE_ASPECT_RATIO = os.getenv("VITYA_IMAGE_ASPECT", "9:16")   # для fast-gen


# =============================================================================
# ГЕНЕРАЦИЯ ВИДЕО (fast-gen, как в вашем рабочем скрипте)
# =============================================================================

FAST_GEN_BASE_URL = os.getenv("FAST_GEN_API_BASE", "https://api.fast-gen.ai")
OP_IMAGE_GENERATE = os.getenv("FAST_GEN_IMAGE_OPERATION", "flower_image_generate")
OP_VIDEO_FROM_IMAGE = os.getenv("FAST_GEN_VIDEO_OPERATION", "flow_video_from_ingredients")
VIDEO_MODEL = os.getenv("FAST_GEN_VIDEO_MODEL") or None
VIDEO_ASPECT_RATIO = os.getenv("FAST_GEN_VIDEO_ASPECT_RATIO", "9:16")

REQUEST_TIMEOUT = int(os.getenv("FAST_GEN_REQUEST_TIMEOUT", "300"))
OPERATION_POLL_SEC = int(os.getenv("FAST_GEN_OPERATION_POLL_SEC", "10"))
RETRY_DELAY_SEC = int(os.getenv("FAST_GEN_RETRY_DELAY_SEC", "8"))
MAX_ATTEMPTS = max(1, int(os.getenv("FAST_GEN_MAX_ATTEMPTS", "4")))
MAX_IMAGE_BYTES = 5 * 1024 * 1024   # лимит inline data URI в fast-gen


# =============================================================================
# МОНТАЖ (тайминги взяты из вашего референса IMG_1645.MOV)
# =============================================================================
# Референс: хук 1.7с + скринкаст 3.3с + финальное фото 1.65с = 6.64с
# Вариативность ±2 секунды задаётся диапазонами ниже.

VIDEO_W = 1080
VIDEO_H = 1920
VIDEO_FPS = 30

HOOK_SECONDS_RANGE = (1.7, 2.8)     # первая часть: оживлённое фото + надпись
FINAL_SECONDS_RANGE = (1.6, 2.6)    # третья часть: итоговое фото
MIDDLE_MAX_SECONDS = float(os.getenv("VITYA_MIDDLE_MAX_SECONDS", "5.0"))
CROSSFADE_SECONDS = 0.35            # переход на финальное фото
MUSIC_FADE_OUT = 0.8

# Шрифт надписи. Берётся первый существующий из списка.
FONT_CANDIDATES = [
    os.getenv("VITYA_FONT", ""),
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/Library/Fonts/Arial Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/SFNSDisplay.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
]

FONT_SIZE_MAX = 64          # подбирается автоматически вниз, чтобы строка влезла
TEXT_TOP_RATIO = 0.135      # позиция надписи от верха кадра (как в референсе)
TEXT_MAX_WIDTH_RATIO = 0.90
TEXT_LINE_SPACING = 1.18


# =============================================================================
# СЛУЖЕБНОЕ
# =============================================================================

def font_path() -> str:
    """Первый существующий шрифт из списка кандидатов."""
    for candidate in FONT_CANDIDATES:
        if candidate and Path(candidate).exists():
            return candidate
    raise RuntimeError(
        "Не найден шрифт для надписи. Укажите его явно:\n"
        "  export VITYA_FONT='/System/Library/Fonts/Supplemental/Arial Bold.ttf'"
    )


def ensure_dirs() -> None:
    for folder in (OUTPUT_DIR, PHOTOS_DIR, VIDEOS_DIR, CACHE_DIR):
        folder.mkdir(parents=True, exist_ok=True)


def require_openai_key() -> str:
    if not OPENAI_API_KEY:
        raise SystemExit(
            "Не найден OPENAI_API_KEY (берётся из окружения, как в старом скрипте).\n"
            "  export OPENAI_API_KEY='sk-...'\n"
            "Чтобы прописать навсегда:\n"
            "  echo 'export OPENAI_API_KEY=\"sk-...\"' >> ~/.zshrc && source ~/.zshrc"
        )
    return OPENAI_API_KEY


def require_fastgen_key() -> str:
    if not FAST_GEN_API_KEY:
        raise SystemExit(
            "Не найден FAST_GEN_API_KEY (берётся из окружения, как в старом скрипте).\n"
            "  echo 'export FAST_GEN_API_KEY=\"ТВОЙ_КЛЮЧ\"' >> ~/.zshrc && source ~/.zshrc"
        )
    return FAST_GEN_API_KEY
