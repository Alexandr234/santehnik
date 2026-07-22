#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
РЕМОНТНИК: догенерация недостающих видео до 100%.

Что делает:
  1) Для каждого языка (RU/GE/PL/ES) читает файл промптов
     (RU_русский/ru_RU_prompts.txt и т.д.) — это ПОЛНЫЙ список нужных позиций
     (index + тайминги + текст промпта).
  2) Смотрит папку ВИЗУАЛ/<ЯЗЫК>/ и по префиксу имени файла (0001_..., 0002_...)
     определяет, какие индексы уже сгенерированы (mp4 существует и не пустой).
  3) Находит ПРОПУЩЕННЫЕ позиции (есть в промптах, нет среди видео).
  4) Для каждой пропущенной позиции:
        - пробует исходный промпт;
        - если провайдер блокирует по safety-фильтру — ПЕРЕПИСЫВАЕТ промпт
          в безопасный вид (через OpenAI, если задан OPENAI_API_KEY, иначе
          встроенным эвристическим "очистителем") и пробует снова;
        - эскалация уровней: чем выше уровень, тем нейтральнее промпт;
        - последний уровень — гарантированно безопасная нейтральная сцена
          (спокойный природный кадр), которая проходит фильтр практически всегда.
     Схема генерации та же, что в основном скрипте: текст -> картинка
     (nano_banana_pro_image_generate, flow) -> видео из картинки (flow_video_from_ingredients),
     стартовый кадр передаётся inline как data:image/...;base64 в inputs[].
     Т.е. полностью на flow. Модель картинки переопределяется через FAST_GEN_IMAGE_OPERATION
     (альт.: nano_banana_2_image_generate, nano_banana_2_lite_image_generate).
  5) Успешный (возможно переписанный) промпт записывается обратно в файл промптов
     (с бэкапом), чтобы повторные прогоны были стабильными. Отключается флагом
     --no-update-prompts.
  6) Имя итогового файла совпадает с тем, что делает основной генератор
     ({index:04d}_{start}__{end}.mp4), поэтому финальный сборщик его увидит.

Запуск:
   export FAST_GEN_API_KEY="ТВОЙ_КЛЮЧ"
   export OPENAI_API_KEY="sk-..."          # опционально, для умного переписывания
   python flow_repair_missing_videos_100percent.py

Полезное:
   python flow_repair_missing_videos_100percent.py --locales RU GE
   python flow_repair_missing_videos_100percent.py --dry-run          # только показать пропуски
   python flow_repair_missing_videos_100percent.py --no-update-prompts
   export FAST_GEN_MAX_ATTEMPTS="4"        # ретраи при ВРЕМЕННЫХ ошибках
   export FAST_GEN_VIDEO_WORKERS="4"
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import re
import sys
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests


# =========================
# CONFIG (совпадает с основным генератором)
# =========================

API_KEY = os.getenv("FAST_GEN_API_KEY") or os.getenv("FASTGEN_API_KEY") or os.getenv("MEDIA_GEN_API_KEY") or ""
BASE_URL = os.getenv("FAST_GEN_API_BASE", "https://api.fast-gen.ai")

BASE_DIR = Path(os.getenv("BASE_FOLDER", "/Users/aleksandrtomilov/Desktop/ПТИЦЫ РЕАЛИСТИЧНЫЕ"))
PROMPTS_ROOT = BASE_DIR / "ПРОМПТЫ"
VISUAL_ROOT = BASE_DIR / "ВИЗУАЛ"

LOCALES = ["RU", "GE", "PL", "ES"]

PROMPT_FILES_BY_LOCALE: Dict[str, Path] = {
    "GE": PROMPTS_ROOT / "DE_немецкий" / "de_GE_prompts.txt",
    "ES": PROMPTS_ROOT / "ES_испанский" / "es_ES_prompts.txt",
    "PL": PROMPTS_ROOT / "PL_польский" / "pl_PL_prompts.txt",
    "RU": PROMPTS_ROOT / "RU_русский" / "ru_RU_prompts.txt",
}

IMAGE_ASPECT_RATIO = os.getenv("FAST_GEN_IMAGE_ASPECT_RATIO", "16:9")
VIDEO_ASPECT_RATIO = os.getenv("FAST_GEN_VIDEO_ASPECT_RATIO", "16:9")

REQUEST_TIMEOUT = int(os.getenv("FAST_GEN_REQUEST_TIMEOUT", "300"))
OPERATION_POLL_SEC = int(os.getenv("FAST_GEN_OPERATION_POLL_SEC", "10"))
RETRY_DELAY_SEC = int(os.getenv("FAST_GEN_RETRY_DELAY_SEC", "8"))

# Ретраи при ВРЕМЕННЫХ ошибках (сеть/429/5xx). Постоянные (safety/4xx) не повторяем.
MAX_ATTEMPTS = max(1, int(os.getenv("FAST_GEN_MAX_ATTEMPTS", "4")))

MAX_VIDEO_WORKERS = int(os.getenv("FAST_GEN_VIDEO_WORKERS", "4"))

# V6 endpoints.
V6_GENERATIONS_ENDPOINT = "/api/v6/generations"
V6_GENERATION_STATUS_ENDPOINT = "/api/v6/generations/{generation_id}"

OP_IMAGE_GENERATE = os.getenv("FAST_GEN_IMAGE_OPERATION", "nano_banana_pro_image_generate")
OP_VIDEO_FROM_IMAGE = os.getenv("FAST_GEN_VIDEO_OPERATION", "flow_video_from_ingredients")

VIDEO_MODEL = os.getenv("FAST_GEN_VIDEO_MODEL") or None

_SEED_ENV = os.getenv("FAST_GEN_SEED", "").strip()
GENERATION_SEED: Optional[int] = int(_SEED_ENV) if _SEED_ENV.lstrip("-").isdigit() else None

_DURATION_ENV = os.getenv("FAST_GEN_VIDEO_DURATION_SECONDS", "").strip()
VIDEO_DURATION_SECONDS: Optional[int] = int(_DURATION_ENV) if _DURATION_ENV.isdigit() else None

VIDEO_RESOLUTION = os.getenv("FAST_GEN_VIDEO_RESOLUTION") or None
VIDEO_ULTRA = os.getenv("FAST_GEN_VIDEO_ULTRA", "").strip().lower() in {"1", "true", "yes", "on"}

MAX_IMAGE_BYTES = 5 * 1024 * 1024

# Файл считается валидным видео, если он больше этого размера. Меньше — битый/пустой.
MIN_VIDEO_BYTES = int(os.getenv("FAST_GEN_MIN_VIDEO_BYTES", "1024"))

# Лимит стартов видео в час (защита от 429).
MAX_VIDEO_STARTS_PER_HOUR = int(os.getenv("FAST_GEN_MAX_VIDEO_STARTS_PER_HOUR", "150"))
RATE_WINDOW_SECONDS = 3600

# Модель для переписывания промптов (если доступен OpenAI).
PROMPT_MODEL = os.getenv("PROMPT_MODEL", "gpt-4o-mini")

# Уровни эскалации: 0 = оригинал; 1..(MAX-1) = переписанные; MAX = гарантированный безопасный.
MAX_REWRITE_LEVELS = int(os.getenv("FAST_GEN_MAX_REWRITE_LEVELS", "4"))
# Сколько дополнительных заходов сделать на гарантированном уровне, если мешает сеть.
EXTRA_GUARANTEED_ROUNDS = int(os.getenv("FAST_GEN_EXTRA_GUARANTEED_ROUNDS", "3"))

TIMECODE_RE = re.compile(
    r"^(?:(\d+)\s*\|\s*)?"
    r"(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*"
    r"(\d{2}:\d{2}:\d{2},\d{3})"
    r"(?:\s*\|.*)?$"
)

PERMANENT_ERROR_MARKERS = (
    "safety", "blocked", "moderation", "content policy",
    "prohibited", "not allowed", "violat",
)

# Слова, которые чаще всего валят модерацию, и их нейтральные замены.
RISKY_REPLACEMENTS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\bblood[a-z]*\b", re.I), ""),
    (re.compile(r"\bgore?\b", re.I), ""),
    (re.compile(r"\bgory\b", re.I), ""),
    (re.compile(r"\bkill(?:s|ing|ed|er|ers)?\b", re.I), "catching"),
    (re.compile(r"\bhunt(?:s|ing|ed|er|ers)?\b", re.I), "searching for food"),
    (re.compile(r"\bprey\b", re.I), "small fish"),
    (re.compile(r"\bpredator(?:s|y)?\b", re.I), "bird"),
    (re.compile(r"\battack(?:s|ing|ed)?\b", re.I), "approaching"),
    (re.compile(r"\bdead\b", re.I), "resting"),
    (re.compile(r"\bdeath\b", re.I), "stillness"),
    (re.compile(r"\bdying\b", re.I), "resting"),
    (re.compile(r"\bcarcass(?:es)?\b", re.I), "rocks"),
    (re.compile(r"\bcorpse(?:s)?\b", re.I), "rocks"),
    (re.compile(r"\bwound(?:s|ed|ing)?\b", re.I), ""),
    (re.compile(r"\bfight(?:s|ing)?\b", re.I), "gathering"),
    (re.compile(r"\baggressive(?:ly)?\b", re.I), "lively"),
    (re.compile(r"\bviolen(?:t|tly|ce)\b", re.I), "energetic"),
    (re.compile(r"\bstrike(?:s|ing)?\b", re.I), "gliding"),
    (re.compile(r"\bstab(?:s|bing|bed)?\b", re.I), ""),
    (re.compile(r"\bspear(?:s|ing|ed)?\b", re.I), "beak"),
    (re.compile(r"\btearing (?:flesh|meat|apart)\b", re.I), "resting"),
    (re.compile(r"\bflesh\b", re.I), ""),
    (re.compile(r"\bmeat\b", re.I), "food"),
    (re.compile(r"\bdrown(?:s|ing|ed)?\b", re.I), "swimming"),
    (re.compile(r"\bstarv(?:e|ing|ation)\b", re.I), "resting"),
    (re.compile(r"\bsuffer(?:s|ing|ed)?\b", re.I), ""),
    (re.compile(r"\bwar\b", re.I), "sky"),
    (re.compile(r"\bweapon(?:s)?\b", re.I), ""),
    (re.compile(r"\bgun(?:s)?\b", re.I), ""),
    (re.compile(r"\bblade(?:s)?\b", re.I), ""),
    (re.compile(r"\bnaked\b", re.I), ""),
    (re.compile(r"\bnude\b", re.I), ""),
]

# Гарантированно безопасные нейтральные сцены (природа/море/птица без конфликта).
SAFE_FALLBACK_SCENES = [
    "A peaceful wide landscape of a calm sea at golden hour, gentle waves, soft warm sunlight, "
    "drifting clouds, serene and beautiful, no people, no conflict",
    "A single seabird perched calmly on a rock by the shore, soft daylight, gentle breeze, "
    "tranquil natural scene, no people, no conflict",
    "A calm coastal cliff with soft green grass, gentle wind, distant calm ocean, warm sunset light, "
    "peaceful nature, no people, no conflict",
    "A serene sky with soft clouds at dawn, warm gentle light over a calm sea horizon, "
    "peaceful and quiet, no people, no conflict",
    "A quiet sandy beach at sunrise, soft pastel sky, calm shallow water, gentle reflections, "
    "peaceful and empty, no people, no conflict",
]


# =========================
# DATA MODELS / ERRORS
# =========================

@dataclass
class PromptBlock:
    index: int
    start_tc: str
    end_tc: str
    header: str          # исходная строка-заголовок (для точной перезаписи файла)
    prompt: str


class PermanentError(Exception):
    """safety-фильтр / 4xx — повторять бессмысленно."""


class TransientError(Exception):
    """сеть / 429 / 5xx — можно повторить ограниченно."""


def classify_error_message(message: str) -> bool:
    low = message.lower()
    return any(marker in low for marker in PERMANENT_ERROR_MARKERS)


# =========================
# BASIC HELPERS
# =========================

_print_lock = threading.Lock()


def log(msg: str) -> None:
    with _print_lock:
        print(msg, flush=True)


def clean_base_url(url: str) -> str:
    return url.rstrip("/")


def headers() -> Dict[str, str]:
    return {"X-API-Key": API_KEY, "Content-Type": "application/json"}


def ensure_api_ready() -> None:
    if not API_KEY:
        print("[ERROR] Не найден API ключ. export FAST_GEN_API_KEY=\"...\"")
        sys.exit(1)


def pretty_json(data: Any) -> str:
    try:
        return json.dumps(data, ensure_ascii=False, indent=2)
    except Exception:
        return str(data)


def safe_timecode(value: str) -> str:
    return value.replace(":", "-").replace(",", "-").replace(" ", "")


def output_name(block: PromptBlock) -> str:
    # Такое же имя, как в основном генераторе, чтобы сборщик увидел файл.
    return f"{block.index:04d}_{safe_timecode(block.start_tc)}__{safe_timecode(block.end_tc)}.mp4"


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


# =========================
# PROMPT FILE PARSING / WRITING
# =========================

def parse_prompt_blocks(path: Path) -> List[PromptBlock]:
    lines = path.read_text(encoding="utf-8").splitlines()
    blocks: List[PromptBlock] = []

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        m = TIMECODE_RE.match(line)
        if not m:
            i += 1
            continue

        raw_index, start_tc, end_tc = m.groups()
        parsed_index = int(raw_index) if raw_index else len(blocks) + 1
        header = lines[i].rstrip("\n")

        i += 1
        prompt_lines: List[str] = []
        while i < len(lines):
            current = lines[i].strip()
            if TIMECODE_RE.match(current):
                break
            if current:
                prompt_lines.append(current)
            i += 1

        prompt = " ".join(prompt_lines).strip()
        if prompt:
            blocks.append(PromptBlock(
                index=parsed_index,
                start_tc=start_tc,
                end_tc=end_tc,
                header=header,
                prompt=prompt,
            ))

    return blocks


def write_prompt_blocks(path: Path, blocks: List[PromptBlock], updates: Dict[int, str]) -> None:
    """Переписывает файл промптов, подставляя новые тексты для указанных индексов.
    Заголовки (тайминги) сохраняются дословно. Перед записью делается бэкап."""
    backup = path.with_suffix(path.suffix + f".bak_{time.strftime('%Y%m%d_%H%M%S')}")
    try:
        backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    except Exception as e:
        log(f"[WARN] не удалось сделать бэкап {path.name}: {e}")

    out_lines: List[str] = []
    for b in blocks:
        new_prompt = updates.get(b.index, b.prompt)
        out_lines.append(b.header)
        out_lines.append(new_prompt)
        out_lines.append("")
    path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    log(f"[PROMPTS] Обновлён файл промптов: {path} (бэкап: {backup.name})")


# =========================
# EXISTING VIDEOS / MISSING DETECTION
# =========================

def existing_video_indices(locale_dir: Path) -> Dict[int, Path]:
    """index -> путь к валидному (непустому) видео."""
    done: Dict[int, Path] = {}
    if not locale_dir.exists():
        return done
    for p in locale_dir.glob("*.mp4"):
        m = re.match(r"^(\d+)", p.name)
        if not m:
            continue
        try:
            if p.stat().st_size >= MIN_VIDEO_BYTES:
                done[int(m.group(1))] = p
        except OSError:
            continue
    return done


def find_missing_blocks(blocks: List[PromptBlock], done: Dict[int, Path]) -> List[PromptBlock]:
    return [b for b in blocks if b.index not in done]


# =========================
# DATA URI / RESULT HELPERS
# =========================

def split_data_uri(data_uri: str) -> Tuple[str, bytes]:
    if not isinstance(data_uri, str) or not data_uri.startswith("data:") or "," not in data_uri:
        raise ValueError(f"Ответ API не похож на data URI: {str(data_uri)[:120]}")
    header, b64 = data_uri.split(",", 1)
    mime = header.replace("data:", "").replace(";base64", "")
    return mime, base64.b64decode(b64)


def write_bytes_checked(raw: bytes, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    if not path.exists() or path.stat().st_size == 0:
        raise TransientError(f"Файл не сохранился или пустой: {path}")


def infer_mime_type(path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    return mime or "image/png"


def image_to_data_uri(image_path: Path) -> str:
    size = image_path.stat().st_size
    if size > MAX_IMAGE_BYTES:
        raise PermanentError(
            f"Стартовый кадр слишком большой для inline ({size/1024/1024:.2f} MB > 5 MB)."
        )
    raw = image_path.read_bytes()
    encoded = base64.b64encode(raw).decode("ascii")
    return f"data:{infer_mime_type(image_path)};base64,{encoded}"


def download_url_to_bytes(url: str) -> bytes:
    last_error: Optional[Exception] = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            resp = requests.get(url, headers={"X-API-Key": API_KEY}, timeout=REQUEST_TIMEOUT)
            if 400 <= resp.status_code < 500:
                raise PermanentError(f"HTTP {resp.status_code}: {resp.text[:200]}")
            if resp.status_code >= 500:
                raise TransientError(f"HTTP {resp.status_code}: {resp.text[:200]}")
            if not resp.content:
                raise TransientError("Пустой ответ при скачивании результата")
            return resp.content
        except PermanentError:
            raise
        except KeyboardInterrupt:
            raise
        except Exception as e:
            last_error = e
            if attempt < MAX_ATTEMPTS:
                time.sleep(RETRY_DELAY_SEC)
    raise TransientError(f"Не удалось скачать результат: {last_error}")


def save_result_item_to_file(result_item: Dict[str, Any], path: Path) -> None:
    inline = result_item.get("data")
    if isinstance(inline, str) and inline.startswith("data:"):
        _mime, raw = split_data_uri(inline)
        write_bytes_checked(raw, path)
        return
    download_url = result_item.get("download_url")
    if isinstance(download_url, str) and download_url:
        write_bytes_checked(download_url_to_bytes(download_url), path)
        return
    raise TransientError(f"Result item без data и download_url: {pretty_json(result_item)}")


# =========================
# HTTP HELPERS
# =========================

def request_post(url: str, payload: dict, *, label: str) -> dict:
    last_error: Optional[Exception] = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            resp = requests.post(url, headers=headers(), json=payload, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 429:
                raise TransientError(f"Rate limit 429: {resp.text[:300]}")
            if 400 <= resp.status_code < 500:
                raise PermanentError(f"HTTP {resp.status_code}: {resp.text[:300]}")
            if resp.status_code >= 500:
                raise TransientError(f"HTTP {resp.status_code}: {resp.text[:300]}")
            return resp.json()
        except PermanentError:
            raise
        except KeyboardInterrupt:
            raise
        except Exception as e:
            last_error = e
            if attempt < MAX_ATTEMPTS:
                time.sleep(RETRY_DELAY_SEC)
    raise TransientError(f"{label}: не удалось за {MAX_ATTEMPTS} попыток: {last_error}")


def request_get(url: str, *, label: str) -> dict:
    last_error: Optional[Exception] = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            resp = requests.get(url, headers={"X-API-Key": API_KEY}, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 429:
                raise TransientError(f"Rate limit 429: {resp.text[:300]}")
            if 400 <= resp.status_code < 500:
                raise PermanentError(f"HTTP {resp.status_code}: {resp.text[:300]}")
            if resp.status_code >= 500:
                raise TransientError(f"HTTP {resp.status_code}: {resp.text[:300]}")
            return resp.json()
        except PermanentError:
            raise
        except KeyboardInterrupt:
            raise
        except Exception as e:
            last_error = e
            if attempt < MAX_ATTEMPTS:
                time.sleep(RETRY_DELAY_SEC)
    raise TransientError(f"{label}: не удалось за {MAX_ATTEMPTS} попыток: {last_error}")


# =========================
# RATE LIMITER
# =========================

class HourlyRateLimiter:
    def __init__(self, max_events: int, window_seconds: int):
        self.max_events = max_events
        self.window_seconds = window_seconds
        self.events: deque = deque()
        self.lock = threading.Lock()

    def acquire(self, tag: str) -> None:
        while True:
            with self.lock:
                now = time.time()
                while self.events and (now - self.events[0] >= self.window_seconds):
                    self.events.popleft()
                if len(self.events) < self.max_events:
                    self.events.append(now)
                    return
                wait_time = max(1, int(self.window_seconds - (now - self.events[0])))
            log(f"[RATE-WAIT] {tag}: лимит {self.max_events}/час, жду {min(wait_time, 60)} сек")
            time.sleep(min(wait_time, 60))


video_rate_limiter = HourlyRateLimiter(MAX_VIDEO_STARTS_PER_HOUR, RATE_WINDOW_SECONDS)


# =========================
# V6 GENERATIONS API
# =========================

def create_generation(payload: Dict[str, Any], *, label: str) -> str:
    url = clean_base_url(BASE_URL) + V6_GENERATIONS_ENDPOINT
    data = request_post(url, payload, label=label)
    generation_id = data.get("id")
    if not generation_id:
        raise PermanentError(f"{label}: API не вернул generation id: {pretty_json(data)}")
    return generation_id


def start_flow_image(prompt: str) -> str:
    payload = {"operation": OP_IMAGE_GENERATE, "prompt": prompt, "aspect_ratio": IMAGE_ASPECT_RATIO}
    return create_generation(payload, label="IMAGE GENERATE")


# Аудио-требование для генерации видео СО ЗВУКОМ: только оригинальные звуки природы,
# как они звучат в реальности. Строго без человеческой речи, голоса за кадром и музыки.
AUDIO_PROMPT_SUFFIX = (
    "Audio: only authentic natural ambient sound captured on location, exactly as it sounds in the wild — "
    "wind, air, water, waves, rustling vegetation, and the real calls, cries and movement sounds of the "
    "animals visible on screen. Strictly NO human voice, NO speech, NO narration, NO voice-over, "
    "NO singing, NO music, NO soundtrack, NO score, NO added artificial sound effects."
)


def build_video_prompt(scene: str) -> str:
    base = (
        "Animate this exact photorealistic documentary shot with subtle, natural cinematic motion. "
        "Keep the same real subject, composition, colors, lighting, and photoreal nature-documentary look. "
        "Gentle natural movement, moving water and clouds, gentle telephoto camera push-in, real shallow depth of field. "
        "Stay photorealistic: no illustration, no painting, no cartoon, no 3D or CGI look. "
        "Do not add text, subtitles, labels, logos, or watermark. "
        f"Scene: {scene}"
    )
    # Не дублируем аудио-блок, если он уже пришёл из файла промптов.
    if "no music" not in scene.lower():
        base = f"{base} {AUDIO_PROMPT_SUFFIX}"
    return base


def build_video_payload(image_data_uri: str, scene: str) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "operation": OP_VIDEO_FROM_IMAGE,
        "prompt": build_video_prompt(scene),
        "inputs": [image_data_uri],
        "aspect_ratio": VIDEO_ASPECT_RATIO,
    }
    if VIDEO_MODEL:
        payload["model"] = VIDEO_MODEL
    if GENERATION_SEED is not None:
        payload["seed"] = GENERATION_SEED
    if VIDEO_DURATION_SECONDS is not None:
        payload["duration_seconds"] = VIDEO_DURATION_SECONDS
    if VIDEO_RESOLUTION:
        payload["resolution"] = VIDEO_RESOLUTION
    if VIDEO_ULTRA:
        payload["ultra"] = True
    return payload


def start_video_from_image(image_data_uri: str, scene: str) -> str:
    return create_generation(build_video_payload(image_data_uri, scene), label="VIDEO FROM INGREDIENTS")


def poll_generation(generation_id: str, *, label: str) -> Dict[str, Any]:
    url = clean_base_url(BASE_URL) + V6_GENERATION_STATUS_ENDPOINT.format(generation_id=generation_id)
    while True:
        data = request_get(url, label=f"{label} STATUS {generation_id}")
        status = str(data.get("status") or "").lower()

        if status in ("queued", "running"):
            time.sleep(OPERATION_POLL_SEC)
            continue
        if status in ("succeeded", "success", "completed", "done"):
            results = data.get("results") or []
            if not results:
                raise TransientError(f"{label}: results пустой: {pretty_json(data)}")
            return results[0]
        if status in ("failed", "error", "cancelled", "canceled"):
            error_str = str(data.get("error") or pretty_json(data))
            if classify_error_message(error_str):
                raise PermanentError(f"{label} заблокировано: {error_str}")
            raise TransientError(f"{label} ошибка: {error_str}")
        raise TransientError(f"Неизвестный статус {label}: {pretty_json(data)}")


# =========================
# PROMPT REWRITING
# =========================

_openai_lock = threading.Lock()
_openai_client_cache: List[Any] = []  # [client] или [None]


def get_openai_client():
    with _openai_lock:
        if _openai_client_cache:
            return _openai_client_cache[0]
        client = None
        if os.getenv("OPENAI_API_KEY"):
            try:
                from openai import OpenAI  # локальный импорт, чтобы не требовать пакет
                client = OpenAI()
            except Exception as e:
                log(f"[WARN] OpenAI недоступен ({e}); переписываю промпты эвристикой.")
                client = None
        _openai_client_cache.append(client)
        return client


def heuristic_sanitize(prompt: str, level: int) -> str:
    text = prompt
    for pattern, repl in RISKY_REPLACEMENTS:
        text = pattern.sub(repl, text)
    text = clean_text(text)
    if level >= 2:
        text = (
            "A peaceful, calm and serene photorealistic documentary nature scene. "
            f"{text}. No violence, no conflict, no distress, no gore, no injury."
        )
    if level >= 3:
        # Сильно обобщаем: оставляем только спокойное окружение.
        text = (
            "A tranquil, gentle and beautiful photorealistic nature documentary scene, "
            "soft natural light, calm atmosphere, peaceful mood, no people, no conflict, no distress."
        )
    return clean_text(text)


_LLM_INSTRUCTIONS = {
    1: "Rewrite this image/video generation prompt so it safely passes content moderation filters. "
       "Remove any violence, gore, blood, hunting, killing, death, injury, weapons, or distressing content. "
       "Keep it a photorealistic documentary nature scene, keep the same general subject and setting if they are safe. "
       "Return ONLY the rewritten prompt in English, one paragraph, no quotes, no explanations.",
    2: "Rewrite this prompt into a CALM, NEUTRAL, peaceful photorealistic documentary nature scene. "
       "Remove any animal conflict, predation, distress, injury or anything a strict safety filter could flag. "
       "Keep it visual and serene. Return ONLY the rewritten prompt in English, no quotes, no explanations.",
    3: "Produce a very neutral, peaceful, safe photorealistic nature documentary scene loosely inspired by this text: "
       "gentle landscape, soft light, calm mood, no people, no animals in conflict, nothing a content filter could object to. "
       "Return ONLY the prompt in English, no quotes, no explanations.",
}


def llm_rewrite(client, prompt: str, level: int) -> Optional[str]:
    instruction = _LLM_INSTRUCTIONS.get(level, _LLM_INSTRUCTIONS[3])
    try:
        response = client.chat.completions.create(
            model=PROMPT_MODEL,
            temperature=0.4,
            messages=[
                {"role": "system", "content": "You are a careful prompt editor for a photorealistic documentary. "
                                              "You make prompts safe for strict image/video content filters while keeping them visual."},
                {"role": "user", "content": f"{instruction}\n\nORIGINAL PROMPT:\n{prompt}"},
            ],
        )
        text = clean_text(response.choices[0].message.content or "")
        return text or None
    except Exception as e:
        log(f"[WARN] LLM-переписывание не удалось (level {level}): {e}")
        return None


def guaranteed_safe_scene(seed_index: int) -> str:
    return SAFE_FALLBACK_SCENES[seed_index % len(SAFE_FALLBACK_SCENES)]


def choose_prompt_for_level(original: str, level: int, seed_index: int) -> str:
    """level 0 = оригинал; 1..MAX-1 = переписанный; MAX = гарантированный безопасный."""
    if level <= 0:
        return original
    if level >= MAX_REWRITE_LEVELS:
        return guaranteed_safe_scene(seed_index)

    client = get_openai_client()
    if client is not None:
        rewritten = llm_rewrite(client, original, level)
        if rewritten:
            return rewritten
    # фолбэк на эвристику, если LLM недоступен или вернул пусто
    return heuristic_sanitize(original, level)


# =========================
# GENERATION (image -> video, Flow), with transient retries
# =========================

def generate_video_once(scene: str, out_path: Path, tmp_img_path: Path) -> Tuple[str, str]:
    # 1) стартовый кадр из текста
    image_generation_id = start_flow_image(scene)
    image_item = poll_generation(image_generation_id, label="IMAGE")
    save_result_item_to_file(image_item, tmp_img_path)

    # 2) inline data URI
    image_data_uri = image_to_data_uri(tmp_img_path)

    # 3) видео из картинки (Flow), с учётом часового лимита
    video_rate_limiter.acquire(out_path.name)
    video_generation_id = start_video_from_image(image_data_uri, scene)

    # 4) ждём результат и сохраняем
    video_item = poll_generation(video_generation_id, label="VIDEO")
    save_result_item_to_file(video_item, out_path)
    return image_generation_id, video_generation_id


def generate_video_with_transient_retries(scene: str, out_path: Path, tmp_img_path: Path) -> Tuple[str, str]:
    """Возвращает (image_gen_id, video_gen_id) при успехе.
    PermanentError пробрасывается сразу (эскалация уровня выше). TransientError — после ретраев."""
    last_error: Optional[Exception] = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return generate_video_once(scene, out_path, tmp_img_path)
        except PermanentError:
            raise
        except KeyboardInterrupt:
            raise
        except Exception as e:
            last_error = e
            log(f"[VIDEO ERROR] {out_path.name}: {e}")
            if attempt < MAX_ATTEMPTS:
                log(f"    retry in {RETRY_DELAY_SEC}s...")
                time.sleep(RETRY_DELAY_SEC)
    raise TransientError(str(last_error))


def repair_one(block: PromptBlock, locale: str, locale_dir: Path) -> dict:
    """Догенерирует одну пропущенную позицию, эскалируя переписывание до успеха."""
    out_path = locale_dir / output_name(block)
    tmp_dir = locale_dir / "_video_start_images_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_img_path = tmp_dir / f"start_{block.index:04d}.png"

    def cleanup() -> None:
        try:
            tmp_img_path.unlink(missing_ok=True)
            if tmp_dir.exists() and not any(tmp_dir.iterdir()):
                tmp_dir.rmdir()
        except Exception:
            pass

    log(f"[REPAIR] {locale} #{block.index:04d} -> {out_path.name}")

    # Эскалация: 0=оригинал, 1..MAX-1=переписанный, MAX=гарантированный.
    for level in range(0, MAX_REWRITE_LEVELS + 1):
        scene = choose_prompt_for_level(block.prompt, level, block.index)
        tag = "оригинал" if level == 0 else (f"переписан L{level}" if level < MAX_REWRITE_LEVELS else "гарантированный")
        log(f"    L{level} ({tag})")
        try:
            image_gen, video_gen = generate_video_with_transient_retries(scene, out_path, tmp_img_path)
            cleanup()
            log(f"    OK: {out_path.name} (level {level})")
            return {
                "locale": locale, "index": block.index, "output": str(out_path),
                "status": "success", "level": level, "rewritten": scene if level > 0 else None,
                "image_generation_id": image_gen, "generation_id": video_gen,
            }
        except PermanentError as e:
            log(f"    BLOCKED на L{level}: {e} — эскалирую")
            continue
        except TransientError as e:
            log(f"    TRANSIENT на L{level}: {e} — эскалирую")
            continue

    # Последний рубеж: несколько дополнительных заходов на гарантированных сценах.
    for extra in range(1, EXTRA_GUARANTEED_ROUNDS + 1):
        scene = guaranteed_safe_scene(block.index + extra)
        log(f"    гарантированный доп.заход {extra}")
        try:
            image_gen, video_gen = generate_video_with_transient_retries(scene, out_path, tmp_img_path)
            cleanup()
            log(f"    OK (гарант. доп {extra}): {out_path.name}")
            return {
                "locale": locale, "index": block.index, "output": str(out_path),
                "status": "success", "level": MAX_REWRITE_LEVELS, "rewritten": scene,
                "image_generation_id": image_gen, "generation_id": video_gen,
            }
        except Exception as e:
            log(f"    доп.заход {extra} не удался: {e}")
            continue

    cleanup()
    log(f"    FAILED окончательно: {locale} #{block.index:04d} (вероятно сеть/лимиты)")
    return {
        "locale": locale, "index": block.index, "output": str(out_path),
        "status": "failed", "level": None, "rewritten": None,
        "error": "не удалось даже на гарантированном уровне (проверь сеть/ключ/лимиты)",
    }


# =========================
# PER-LOCALE PROCESSING
# =========================

def process_locale(locale: str, dry_run: bool, update_prompts: bool) -> List[dict]:
    prompt_file = PROMPT_FILES_BY_LOCALE.get(locale)
    if not prompt_file or not prompt_file.exists():
        log(f"[SKIP] {locale}: нет файла промптов ({prompt_file})")
        return []

    locale_dir = VISUAL_ROOT / locale
    locale_dir.mkdir(parents=True, exist_ok=True)

    blocks = parse_prompt_blocks(prompt_file)
    if not blocks:
        log(f"[SKIP] {locale}: файл промптов пустой/не распознан")
        return []

    done = existing_video_indices(locale_dir)
    missing = find_missing_blocks(blocks, done)

    log(f"\n===== {locale} =====")
    log(f"  Всего позиций в промптах: {len(blocks)}")
    log(f"  Уже сгенерировано видео:  {len(done)}")
    log(f"  Пропущено (догенерить):   {len(missing)}")
    if missing:
        preview = ", ".join(f"{b.index:04d}" for b in missing[:30])
        more = "" if len(missing) <= 30 else f" … (+{len(missing) - 30})"
        log(f"  Индексы: {preview}{more}")

    if not missing:
        log(f"  {locale}: всё на месте, 100% ✅")
        return []

    if dry_run:
        log(f"  [DRY-RUN] {locale}: генерацию не запускаю")
        return []

    results: List[dict] = []
    workers = max(1, min(MAX_VIDEO_WORKERS, len(missing)))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {executor.submit(repair_one, b, locale, locale_dir): b for b in missing}
        for future in as_completed(future_map):
            results.append(future.result())

    # Обновляем файл промптов переписанными успешными вариантами.
    if update_prompts:
        updates = {
            r["index"]: r["rewritten"]
            for r in results
            if r.get("status") == "success" and r.get("rewritten")
        }
        if updates:
            write_prompt_blocks(prompt_file, blocks, updates)

    ok = sum(1 for r in results if r.get("status") == "success")
    fail = sum(1 for r in results if r.get("status") != "success")
    log(f"  {locale} итог: догенерировано {ok}/{len(missing)}, не удалось {fail}")
    return results


# =========================
# MAIN
# =========================

def main() -> None:
    global BASE_URL, VISUAL_ROOT

    parser = argparse.ArgumentParser(
        description="Догенерация недостающих видео до 100% (image->video Flow) с переписыванием заблокированных промптов."
    )
    parser.add_argument("--locales", nargs="*", default=LOCALES, help="Какие языки обрабатывать (по умолчанию все).")
    parser.add_argument("--visual-root", default=str(VISUAL_ROOT), help="Папка ВИЗУАЛ.")
    parser.add_argument("--api-base", default=BASE_URL, help="API base.")
    parser.add_argument("--dry-run", action="store_true", help="Только показать пропуски, не генерировать.")
    parser.add_argument("--no-update-prompts", action="store_true", help="Не переписывать файлы промптов.")
    args = parser.parse_args()

    BASE_URL = args.api_base
    VISUAL_ROOT = Path(args.visual_root).expanduser()

    ensure_api_ready()

    log("РЕМОНТНИК недостающих видео (image -> video, Flow)")
    log(f"BASE_URL: {BASE_URL}")
    log(f"Visual root: {VISUAL_ROOT}")
    log(f"Video operation: {OP_VIDEO_FROM_IMAGE} (модель Flow)")
    log(f"Image operation: {OP_IMAGE_GENERATE}")
    log(f"OpenAI переписывание: {'ДА' if os.getenv('OPENAI_API_KEY') else 'нет (эвристика)'}")
    log(f"Уровней эскалации: {MAX_REWRITE_LEVELS} (последний — гарантированно безопасный)")
    log(f"Video workers: {MAX_VIDEO_WORKERS}, max attempts (temp): {MAX_ATTEMPTS}")

    requested = [loc.upper() for loc in args.locales]
    all_results: List[dict] = []
    for locale in requested:
        if locale not in PROMPT_FILES_BY_LOCALE:
            log(f"[SKIP] Неизвестный язык: {locale}")
            continue
        all_results.extend(process_locale(locale, args.dry_run, not args.no_update_prompts))

    # Итоговый лог + проверка 100%.
    if not args.dry_run:
        report_path = VISUAL_ROOT / "repair_log.json"
        try:
            report_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    log("\n========== ИТОГ РЕМОНТА ==========")
    total_ok = sum(1 for r in all_results if r.get("status") == "success")
    total_fail = sum(1 for r in all_results if r.get("status") != "success")
    log(f"  Догенерировано: {total_ok}")
    log(f"  Не удалось:     {total_fail}")

    # Финальная перепроверка покрытия по каждому языку.
    log("\n  Финальное покрытие:")
    all_100 = True
    for locale in requested:
        prompt_file = PROMPT_FILES_BY_LOCALE.get(locale)
        if not prompt_file or not prompt_file.exists():
            continue
        blocks = parse_prompt_blocks(prompt_file)
        done = existing_video_indices(VISUAL_ROOT / locale)
        missing = find_missing_blocks(blocks, done)
        mark = "✅ 100%" if not missing else f"⚠️ не хватает {len(missing)}"
        if missing:
            all_100 = False
        log(f"    {locale}: {len(done)}/{len(blocks)} {mark}")

    if all_100:
        log("\n  ГОТОВО: во всех языках 100% видео. Можно запускать финальный сборщик.")
    else:
        log("\n  Остались пропуски (обычно из-за сети/лимитов/ключа). Просто запусти скрипт ещё раз.")


if __name__ == "__main__":
    main()
