#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FAST visuals generator для media_gen_api V6 (image из текста -> video).

ПЕРЕРАБОТАННАЯ ВЕРСИЯ. Что изменилось относительно старой Veo-версии и почему:

  1) Видео теперь генерируется операцией `flow_video_from_ingredients` (модель Flow),
     а НЕ `flower_video_from_image` (Veo 3.1). У Veo модерация заметно строже и
     половина промптов падала с "Request blocked by provider safety filters".
     Flow пропускает тот же контент значительно чаще.

  2) Стартовый кадр больше НЕ грузится в storage.fast-gen.ai. Он кодируется прямо
     в запрос как data:image/...;base64 и передаётся в inputs[] (V6MediaInput),
     ровно как в рабочем скрипте video_from_local_images_base64. Лимит 5 MB на
     inline-картинку учитывается.

  3) FAIL-FAST на модерации. Раньше generate_*_until_done крутили `while True` и
     повторяли ЛЮБУЮ ошибку бесконечно, включая постоянный safety-блок — отсюда
     `attempt 542`. Теперь:
        - постоянные ошибки (safety / blocked / moderation / 4xx) НЕ повторяются:
          промпт помечается failed, и скрипт идёт дальше;
        - временные ошибки (сеть, 429, 5xx) повторяются, но ОГРАНИЧЕННО
          (FAST_GEN_MAX_ATTEMPTS, по умолчанию 4).

  4) В конце печатается сводка: сколько успехов и какие именно промпты заблокированы.

  5) ПОЛНОСТЬЮ НА FLOW. Раньше стартовый кадр рисовался провайдером flower
     (flower_image_generate). Теперь и картинка идёт через flow: по умолчанию
     nano_banana_pro_image_generate (flow/nano-banana). Видео как и было — flow
     (flow_video_from_ingredients). Модель картинки переопределяется через
     FAST_GEN_IMAGE_OPERATION (альт.: nano_banana_2_image_generate,
     nano_banana_2_lite_image_generate).

Пайплайн для каждого промпта остался прежним по смыслу:
   текст промпта -> nano_banana_pro_image_generate (стартовый кадр, flow)
                 -> flow_video_from_ingredients (оживление кадра) -> mp4

Запуск:
   cd "/Users/aleksandrtomilov/Desktop/ПТИЦЫ РЕАЛИСТИЧНЫЕ"
   source venv/bin/activate
   python "/Users/aleksandrtomilov/Desktop/ПТИЦЫ РЕАЛИСТИЧНЫЕ/ВИЗУАЛ/flow_visual_batch_generator_realistic.py"

Ключ:
   echo 'export FAST_GEN_API_KEY="ТВОЙ_КЛЮЧ"' >> ~/.zshrc
   source ~/.zshrc

Полезные настройки:
   export FAST_GEN_MAX_ATTEMPTS="4"          # сколько раз пробовать при ВРЕМЕННЫХ ошибках (сеть/5xx)
   export FAST_GEN_WAIT_ON_RATE_LIMIT="1"    # 429/часовой лимит: ждать и повторять БЕСКОНЕЧНО (по умолч. вкл)
   export FAST_GEN_RATE_LIMIT_WAIT_SEC="60"  # пауза между повторами при 429, если нет Retry-After
   export FAST_GEN_ANIMATE_FIRST_N="2"
   export FAST_GEN_IMAGE_WORKERS="4"
   export FAST_GEN_VIDEO_WORKERS="1"
   export FAST_GEN_LOCALE_WORKERS="1"
   export FAST_GEN_VIDEO_OPERATION="flow_video_from_ingredients"
   export FAST_GEN_VIDEO_MODEL="flow-video-lite"   # если нужен конкретный shorthand-model
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import mimetypes
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

import requests


# =========================
# CONFIG
# =========================

API_KEY = os.getenv("FAST_GEN_API_KEY") or os.getenv("FASTGEN_API_KEY") or os.getenv("MEDIA_GEN_API_KEY") or ""
BASE_URL = os.getenv("FAST_GEN_API_BASE", "https://api.fast-gen.ai")

BASE_DIR = Path("/Users/aleksandrtomilov/Desktop/ПТИЦЫ РЕАЛИСТИЧНЫЕ")
PROMPTS_ROOT = BASE_DIR / "ПРОМПТЫ"
VISUAL_ROOT = BASE_DIR / "ВИЗУАЛ"

LOCALES = ["RU", "GE", "PL", "ES"]

PROMPT_FILES_BY_LOCALE: Dict[str, Path] = {
    "GE": Path("/Users/aleksandrtomilov/Desktop/ПТИЦЫ РЕАЛИСТИЧНЫЕ/ПРОМПТЫ/DE_немецкий/de_GE_prompts.txt"),
    "ES": Path("/Users/aleksandrtomilov/Desktop/ПТИЦЫ РЕАЛИСТИЧНЫЕ/ПРОМПТЫ/ES_испанский/es_ES_prompts.txt"),
    "PL": Path("/Users/aleksandrtomilov/Desktop/ПТИЦЫ РЕАЛИСТИЧНЫЕ/ПРОМПТЫ/PL_польский/pl_PL_prompts.txt"),
    "RU": Path("/Users/aleksandrtomilov/Desktop/ПТИЦЫ РЕАЛИСТИЧНЫЕ/ПРОМПТЫ/RU_русский/ru_RU_prompts.txt"),
}

# V6 требует aspect_ratio в форме n:n (^[1-9]\d*:[1-9]\d*$), например 16:9, 9:16, 1:1.
IMAGE_ASPECT_RATIO = os.getenv("FAST_GEN_IMAGE_ASPECT_RATIO", "16:9")
VIDEO_ASPECT_RATIO = os.getenv("FAST_GEN_VIDEO_ASPECT_RATIO", "16:9")

# ВЕСЬ РОЛИК СОСТОИТ ИЗ ВИДЕО.
ANIMATE_ALL = os.getenv("FAST_GEN_ANIMATE_ALL", "1").strip().lower() not in ("0", "false", "no", "нет")
ANIMATE_FIRST_N = int(os.getenv("FAST_GEN_ANIMATE_FIRST_N", "2"))

REQUEST_TIMEOUT = int(os.getenv("FAST_GEN_REQUEST_TIMEOUT", "300"))
OPERATION_POLL_SEC = int(os.getenv("FAST_GEN_OPERATION_POLL_SEC", "10"))
RETRY_DELAY_SEC = int(os.getenv("FAST_GEN_RETRY_DELAY_SEC", "8"))
SKIP_EXISTING = True

# Сколько раз пробовать при ВРЕМЕННЫХ ошибках (сеть/5xx). Постоянные ошибки
# (safety/blocked/4xx) не повторяются вовсе.
MAX_ATTEMPTS = max(1, int(os.getenv("FAST_GEN_MAX_ATTEMPTS", "4")))

# ЧАСОВОЙ ЛИМИТ (HTTP 429) — отдельный случай. По умолчанию скрипт НЕ сдаётся, а ЖДЁТ и
# повторяет БЕСКОНЕЧНО, пока окно лимита не сбросится (это НЕ тратит попытки MAX_ATTEMPTS).
# Так можно оставить генерацию надолго: упёрлись в часовой лимит -> подождали -> продолжили.
# Выключить (вернуть старое поведение «429 = обычная временная ошибка»): FAST_GEN_WAIT_ON_RATE_LIMIT=0.
WAIT_ON_RATE_LIMIT = os.getenv("FAST_GEN_WAIT_ON_RATE_LIMIT", "1").strip().lower() not in ("0", "false", "no", "нет")
# Пауза между повторами при 429, сек (если сервер не прислал Retry-After).
RATE_LIMIT_WAIT_SEC = max(5, int(os.getenv("FAST_GEN_RATE_LIMIT_WAIT_SEC", "60")))

SKIP_MISSING_PROMPTS = os.getenv("FAST_GEN_SKIP_MISSING_PROMPTS", "1").strip().lower() not in ("0", "false", "no", "нет")

MAX_IMAGE_WORKERS = int(os.getenv("FAST_GEN_IMAGE_WORKERS", "20"))
MAX_VIDEO_WORKERS = int(os.getenv("FAST_GEN_VIDEO_WORKERS", "20"))
MAX_LOCALE_WORKERS = int(os.getenv("FAST_GEN_LOCALE_WORKERS", "4"))

# V6 media_gen_api endpoints.
V6_GENERATIONS_ENDPOINT = "/api/v6/generations"
V6_GENERATION_STATUS_ENDPOINT = "/api/v6/generations/{generation_id}"

# Canonical V6 operation ids.
#   image из текста — nano_banana_pro_image_generate (стартовый кадр, flow/nano-banana);
#   видео из ингредиентов — flow_video_from_ingredients (модель Flow, мягкая модерация).
OP_IMAGE_GENERATE = os.getenv("FAST_GEN_IMAGE_OPERATION", "nano_banana_pro_image_generate")
OP_VIDEO_FROM_IMAGE = os.getenv("FAST_GEN_VIDEO_OPERATION", "flow_video_from_ingredients")

# Необязательный shorthand-model для видео (например flow-video-lite / flow-video-quality).
VIDEO_MODEL = os.getenv("FAST_GEN_VIDEO_MODEL") or None

# Опциональные параметры видео (отправляются только если заданы через env).
_SEED_ENV = os.getenv("FAST_GEN_SEED", "").strip()
GENERATION_SEED: Optional[int] = int(_SEED_ENV) if _SEED_ENV.lstrip("-").isdigit() else None

_DURATION_ENV = os.getenv("FAST_GEN_VIDEO_DURATION_SECONDS", "").strip()
VIDEO_DURATION_SECONDS: Optional[int] = int(_DURATION_ENV) if _DURATION_ENV.isdigit() else None

VIDEO_RESOLUTION = os.getenv("FAST_GEN_VIDEO_RESOLUTION") or None
VIDEO_ULTRA = os.getenv("FAST_GEN_VIDEO_ULTRA", "").strip().lower() in {"1", "true", "yes", "on"}

# По схеме V6MediaInput для inline data URI максимум 5 MB на одну картинку.
MAX_IMAGE_BYTES = 5 * 1024 * 1024

GLOBAL_LOG_FILE = VISUAL_ROOT / "generation_log.json"

TIMECODE_RE = re.compile(
    r"^(?:(\d+)\s*\|\s*)?"
    r"(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*"
    r"(\d{2}:\d{2}:\d{2},\d{3})"
    r"(?:\s*\|.*)?$"
)

# Маркеры ПОСТОЯННОЙ ошибки провайдера — такие повторять бессмысленно.
PERMANENT_ERROR_MARKERS = (
    "safety",
    "blocked",
    "moderation",
    "content policy",
    "prohibited",
    "not allowed",
    "violat",
)

# Маркеры ЛИМИТА запросов / часового лимита — их ждём (бесконечно), а не считаем обычной ошибкой.
# На случай, если лимит приходит не как HTTP 429, а как текст ошибки в теле/статусе.
RATE_LIMIT_MARKERS = (
    "429",
    "rate limit",
    "rate-limit",
    "too many requests",
    "quota",
    "per hour",
    "per-hour",
    "hourly",
    "hour limit",
)


# =========================
# DATA MODELS / ERRORS
# =========================

@dataclass
class PromptItem:
    index: int
    start_tc: str
    end_tc: str
    prompt: str


class PermanentError(Exception):
    """Ошибка, которую повторять бессмысленно (safety-фильтр, 4xx и т.п.)."""


class TransientError(Exception):
    """Временная ошибка (сеть, 429, 5xx) — можно повторить ограниченное число раз."""


def classify_error_message(message: str) -> bool:
    """True, если сообщение похоже на ПОСТОЯННУЮ ошибку (не повторяем)."""
    low = message.lower()
    return any(marker in low for marker in PERMANENT_ERROR_MARKERS)


def looks_like_rate_limit(message: Any) -> bool:
    """True, если ошибка похожа на лимит запросов / часовой лимит (ждём, не сдаёмся)."""
    low = str(message).lower()
    return any(marker in low for marker in RATE_LIMIT_MARKERS)


# =========================
# BASIC HELPERS
# =========================

def log(msg: str) -> None:
    print(msg, flush=True)


def clean_base_url(url: str) -> str:
    return url.rstrip("/")


def headers(json_content: bool = True) -> Dict[str, str]:
    h = {"X-API-Key": API_KEY}
    if json_content:
        h["Content-Type"] = "application/json"
    return h


def ensure_api_ready() -> None:
    if not API_KEY:
        print("[ERROR] Не найден API ключ.")
        print("Сохрани ключ так:")
        print("echo 'export FAST_GEN_API_KEY=\"ТВОЙ_КЛЮЧ\"' >> ~/.zshrc")
        print("source ~/.zshrc")
        print("echo $FAST_GEN_API_KEY")
        sys.exit(1)
    if not BASE_URL:
        print("[ERROR] Не указан BASE_URL. Нужно: https://api.fast-gen.ai")
        sys.exit(1)


def ensure_dirs() -> None:
    VISUAL_ROOT.mkdir(parents=True, exist_ok=True)
    for locale in LOCALES:
        (VISUAL_ROOT / locale).mkdir(parents=True, exist_ok=True)


def pretty_json(data: Any) -> str:
    try:
        return json.dumps(data, ensure_ascii=False, indent=2)
    except Exception:
        return str(data)


def safe_timecode(value: str) -> str:
    return value.replace(":", "-").replace(",", "-").replace(" ", "")


def output_name(item: PromptItem, is_video: bool) -> str:
    ext = ".mp4" if is_video else ".png"
    return f"{item.index:04d}_{safe_timecode(item.start_tc)}__{safe_timecode(item.end_tc)}{ext}"


def load_json(path: Path, default: Any) -> Any:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return default
    return default


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# =========================
# PROMPTS
# =========================

def find_locale_prompts_file(locale: str) -> Path:
    prompts_file = PROMPT_FILES_BY_LOCALE.get(locale)
    if prompts_file is None:
        available = ", ".join(sorted(PROMPT_FILES_BY_LOCALE))
        raise KeyError(f"Для языка {locale} не задан файл промптов. Доступно: {available}")

    prompts_file = prompts_file.expanduser()
    if not prompts_file.exists():
        raise FileNotFoundError(f"Для {locale} не найден файл промптов: {prompts_file}")
    if not prompts_file.is_file():
        raise IsADirectoryError(f"Путь для {locale} не является файлом: {prompts_file}")

    return prompts_file


def parse_prompts_file(path: Path) -> List[PromptItem]:
    if not path.exists():
        raise FileNotFoundError(f"Не найден файл промптов: {path}")

    lines = path.read_text(encoding="utf-8").splitlines()
    items: List[PromptItem] = []

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        m = TIMECODE_RE.match(line)
        if not m:
            i += 1
            continue

        raw_index, start_tc, end_tc = m.groups()
        parsed_index = int(raw_index) if raw_index else len(items) + 1

        i += 1
        prompt_lines: List[str] = []

        while i < len(lines):
            current_line = lines[i].strip()
            if TIMECODE_RE.match(current_line):
                break
            if current_line:
                prompt_lines.append(current_line)
            i += 1

        prompt = " ".join(prompt_lines).strip()
        if prompt:
            items.append(
                PromptItem(
                    index=parsed_index,
                    start_tc=start_tc,
                    end_tc=end_tc,
                    prompt=prompt,
                )
            )

    if not items:
        raise RuntimeError(
            f"Не нашёл промпты в файле {path}. "
            "Поддерживаемый формат: '001 | 00:00:00,000 --> 00:00:07,000 | duration: 7.00s', "
            "а следующей строкой — текст промпта."
        )

    return items


def load_prompts_by_locale() -> Dict[str, List[PromptItem]]:
    prompts_by_locale: Dict[str, List[PromptItem]] = {}
    skipped_locales: List[str] = []

    for locale in LOCALES:
        try:
            prompts_file = find_locale_prompts_file(locale)
            items = parse_prompts_file(prompts_file)
        except (FileNotFoundError, IsADirectoryError, RuntimeError) as e:
            if SKIP_MISSING_PROMPTS:
                skipped_locales.append(locale)
                log(f"[SKIP] {locale}: файл промптов не найден, пустой или не читается: {e}")
                continue
            raise

        prompts_by_locale[locale] = items
        log(f"[{locale}] Prompts file: {prompts_file}")
        log(f"[{locale}] Parsed prompts: {len(items)}")

    if skipped_locales:
        log(f"[INFO] Пропущены языки без доступных промптов: {', '.join(skipped_locales)}")

    if not prompts_by_locale:
        raise RuntimeError("Не найден ни один рабочий файл промптов. Генерацию запускать не из чего.")

    return prompts_by_locale


# =========================
# DATA URI / RESULT HELPERS
# =========================

def split_data_uri(data_uri: str) -> Tuple[str, bytes]:
    if not isinstance(data_uri, str) or not data_uri.startswith("data:"):
        raise ValueError(f"Ответ API не похож на data URI: {str(data_uri)[:120]}")
    if "," not in data_uri:
        raise ValueError("Некорректный data URI: нет запятой")
    header, b64 = data_uri.split(",", 1)
    mime = header.replace("data:", "").replace(";base64", "")
    return mime, base64.b64decode(b64)


def write_bytes_checked(raw: bytes, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"Файл не сохранился или пустой: {path}")


def infer_mime_type(path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    return mime or "image/png"


def image_to_data_uri(image_path: Path) -> str:
    """Кодирует локальную картинку в data:image/...;base64 для inputs[] (V6MediaInput)."""
    file_size = image_path.stat().st_size
    if file_size > MAX_IMAGE_BYTES:
        raise PermanentError(
            f"Файл слишком большой для inline data URI: {image_path.name} "
            f"({file_size / 1024 / 1024:.2f} MB > 5.00 MB). Уменьши aspect/размер стартового кадра."
        )
    mime_type = infer_mime_type(image_path)
    raw = image_path.read_bytes()
    encoded = base64.b64encode(raw).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def download_url_to_bytes(url: str) -> bytes:
    """Скачивает файл результата по download_url. 429 — ждём бесконечно, прочее — ограниченный ретрай."""
    last_error: Optional[Exception] = None
    rate_waited = 0
    attempts_used = 0
    while attempts_used < MAX_ATTEMPTS:
        try:
            resp = requests.get(url, headers={"X-API-Key": API_KEY}, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 429:
                if WAIT_ON_RATE_LIMIT:
                    rate_waited = wait_for_rate_limit(resp, label="download result", waited=rate_waited)
                    continue
                raise TransientError(f"Rate limit 429: {resp.text[:200]}")
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
            if WAIT_ON_RATE_LIMIT and looks_like_rate_limit(e):
                rate_waited = wait_for_rate_limit(None, label="download result", waited=rate_waited)
                continue
            attempts_used += 1
            last_error = e
            log(f"[WARN] download result попытка {attempts_used}/{MAX_ATTEMPTS} не удалась: {e}")
            if attempts_used < MAX_ATTEMPTS:
                time.sleep(RETRY_DELAY_SEC)
    raise TransientError(f"Не удалось скачать результат за {MAX_ATTEMPTS} попыток: {last_error}")


def save_result_item_to_file(result_item: Dict[str, Any], path: Path) -> None:
    inline = result_item.get("data")
    if isinstance(inline, str) and inline.startswith("data:"):
        _mime, raw = split_data_uri(inline)
        write_bytes_checked(raw, path)
        return

    download_url = result_item.get("download_url")
    if isinstance(download_url, str) and download_url:
        raw = download_url_to_bytes(download_url)
        write_bytes_checked(raw, path)
        return

    raise RuntimeError(f"Result item без data и download_url: {pretty_json(result_item)}")


# =========================
# HTTP HELPERS
# =========================

def wait_for_rate_limit(resp: Optional[requests.Response], *, label: str, waited: int) -> int:
    """Пауза при 429 / часовом лимите. Возвращает суммарное время ожидания.

    ВАЖНО: эта пауза НЕ тратит попытки MAX_ATTEMPTS — вызывающие циклы повторяют
    запрос БЕСКОНЕЧНО, пока окно часового лимита не сбросится. Прервать — Ctrl+C.
    """
    delay = RATE_LIMIT_WAIT_SEC
    if resp is not None:
        retry_after = resp.headers.get("Retry-After")
        if retry_after and str(retry_after).strip().isdigit():
            delay = max(5, int(str(retry_after).strip()))
    total = waited + delay
    mins = total // 60
    extra = f" ≈ {mins} мин" if mins else ""
    log(f"[RATE-LIMIT] {label}: похоже, исчерпан часовой лимит (429). "
        f"Жду {delay}s и повторяю без ограничения по попыткам (в ожидании уже ~{total}s{extra})...")
    time.sleep(delay)
    return total


def request_with_retries_post(url: str, payload: dict, *, label: str) -> dict:
    """POST c ретраем. 429/часовой лимит — ждём бесконечно; 4xx — постоянная ошибка, не повторяем."""
    last_error: Optional[Exception] = None
    rate_waited = 0
    attempts_used = 0
    while attempts_used < MAX_ATTEMPTS:
        try:
            resp = requests.post(url, headers=headers(), json=payload, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 429:
                if WAIT_ON_RATE_LIMIT:
                    rate_waited = wait_for_rate_limit(resp, label=label, waited=rate_waited)
                    continue  # не тратим попытку — ждём сброса лимита
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
            if WAIT_ON_RATE_LIMIT and looks_like_rate_limit(e):
                rate_waited = wait_for_rate_limit(None, label=label, waited=rate_waited)
                continue  # лимит пришёл текстом ошибки — тоже ждём, не тратим попытку
            attempts_used += 1
            last_error = e
            log(f"[WARN] {label}: попытка {attempts_used}/{MAX_ATTEMPTS} не удалась: {e}")
            if attempts_used < MAX_ATTEMPTS:
                time.sleep(RETRY_DELAY_SEC)
    raise TransientError(f"{label}: не удалось за {MAX_ATTEMPTS} попыток: {last_error}")


def request_get_with_retries(url: str, *, label: str, params: Optional[dict] = None) -> dict:
    """GET c ретраем. 429/часовой лимит — ждём бесконечно; 4xx — постоянная ошибка, не повторяем."""
    last_error: Optional[Exception] = None
    rate_waited = 0
    attempts_used = 0
    while attempts_used < MAX_ATTEMPTS:
        try:
            resp = requests.get(url, headers={"X-API-Key": API_KEY}, params=params, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 429:
                if WAIT_ON_RATE_LIMIT:
                    rate_waited = wait_for_rate_limit(resp, label=label, waited=rate_waited)
                    continue
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
            if WAIT_ON_RATE_LIMIT and looks_like_rate_limit(e):
                rate_waited = wait_for_rate_limit(None, label=label, waited=rate_waited)
                continue
            attempts_used += 1
            last_error = e
            log(f"[WARN] {label}: GET попытка {attempts_used}/{MAX_ATTEMPTS} не удалась: {e}")
            if attempts_used < MAX_ATTEMPTS:
                time.sleep(RETRY_DELAY_SEC)
    raise TransientError(f"{label}: не удалось за {MAX_ATTEMPTS} попыток: {last_error}")


# =========================
# V6 GENERATIONS API
# =========================

def create_generation(payload: Dict[str, Any], *, label: str) -> str:
    url = clean_base_url(BASE_URL) + V6_GENERATIONS_ENDPOINT
    data = request_with_retries_post(url, payload, label=label)
    generation_id = data.get("id")
    if not generation_id:
        raise PermanentError(f"{label}: API не вернул generation id: {pretty_json(data)}")
    return generation_id


def start_flow_image(prompt: str) -> str:
    payload: Dict[str, Any] = {
        "operation": OP_IMAGE_GENERATE,
        "prompt": prompt,
        "aspect_ratio": IMAGE_ASPECT_RATIO,
    }
    return create_generation(payload, label="FLOW IMAGE GENERATE")


def build_video_payload(image_data_uri: str, prompt: str) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "operation": OP_VIDEO_FROM_IMAGE,
        "prompt": build_video_prompt(prompt),
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


def start_video_from_image(image_data_uri: str, prompt: str) -> str:
    payload = build_video_payload(image_data_uri, prompt)
    return create_generation(payload, label="VIDEO FROM INGREDIENTS")


def poll_generation(generation_id: str, *, label: str) -> Dict[str, Any]:
    """Ждёт завершения генерации и возвращает первый result item.

    На status=failed кидает PermanentError (если это safety/модерация) или
    TransientError (иначе) — чтобы вызывающий код мог решить, повторять или нет.
    """
    url = clean_base_url(BASE_URL) + V6_GENERATION_STATUS_ENDPOINT.format(generation_id=generation_id)

    while True:
        data = request_get_with_retries(url, label=f"{label} STATUS {generation_id}")
        status = str(data.get("status") or "").lower()

        if status in ("queued", "running"):
            log(f"    {label.lower()} status: {status}")
            time.sleep(OPERATION_POLL_SEC)
            continue

        if status in ("succeeded", "success", "completed", "done"):
            results = data.get("results") or []
            if not results:
                raise TransientError(f"{label} завершилось, но results пустой: {pretty_json(data)}")
            return results[0]

        if status in ("failed", "error", "cancelled", "canceled"):
            error = data.get("error") or pretty_json(data)
            error_str = str(error)
            if classify_error_message(error_str):
                raise PermanentError(f"{label} заблокировано провайдером: {error_str}")
            raise TransientError(f"{label} закончилось ошибкой: {error_str}")

        raise TransientError(f"Неизвестный статус {label}: {pretty_json(data)}")


def build_video_prompt(prompt: str) -> str:
    return (
        "Animate this exact photorealistic wildlife shot with subtle, natural cinematic motion. "
        "Keep the same real bird, composition, colors, lighting, and photoreal nature-documentary look. "
        "Realistic feather and wing movement, natural flight or gliding, moving ocean water, waves and sea spray, "
        "drifting clouds, gentle telephoto camera push-in, real shallow depth of field. "
        "Stay photorealistic: no illustration, no painting, no cartoon, no 3D or CGI look. "
        "Do not add text, subtitles, labels, logos, or watermark. "
        f"Scene: {prompt}"
    )


# =========================
# GENERATION LOGIC
# =========================

def generate_image_to_file(prompt: str, path: Path) -> str:
    """Генерирует стартовую картинку и сохраняет в path, возвращает generation id."""
    generation_id = start_flow_image(prompt)
    result_item = poll_generation(generation_id, label="IMAGE")
    save_result_item_to_file(result_item, path)
    return generation_id


def generate_image_until_done(item: PromptItem, out_path: Path) -> dict:
    """Генерирует картинку. Постоянные ошибки не повторяет — помечает failed."""
    last_error: Optional[str] = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            log(f"[IMAGE/FLOW] {out_path.name} | attempt {attempt}/{MAX_ATTEMPTS}")
            generation_id = start_flow_image(item.prompt)
            log(f"    image generation_id: {generation_id}")
            result_item = poll_generation(generation_id, label="IMAGE")
            save_result_item_to_file(result_item, out_path)
            log(f"    saved: {out_path}")
            return {
                "index": item.index,
                "type": "image",
                "output": str(out_path),
                "status": "success",
                "attempts": attempt,
                "generation_id": generation_id,
            }
        except KeyboardInterrupt:
            raise
        except PermanentError as e:
            log(f"[IMAGE BLOCKED] {out_path.name}: {e} — пропускаю без повтора")
            return {
                "index": item.index,
                "type": "image",
                "output": str(out_path),
                "status": "failed",
                "reason": "permanent",
                "attempts": attempt,
                "error": str(e),
            }
        except Exception as e:
            last_error = str(e)
            log(f"[IMAGE ERROR] {out_path.name}: {e}")
            if attempt < MAX_ATTEMPTS:
                log(f"    retry in {RETRY_DELAY_SEC}s...")
                time.sleep(RETRY_DELAY_SEC)

    return {
        "index": item.index,
        "type": "image",
        "output": str(out_path),
        "status": "failed",
        "reason": "transient_exhausted",
        "attempts": MAX_ATTEMPTS,
        "error": last_error,
    }


def generate_video_until_done(item: PromptItem, out_path: Path, locale_dir: Path) -> dict:
    """Стартовый кадр -> inline base64 -> flow_video_from_ingredients.

    Постоянные ошибки (safety) не повторяет — помечает failed и идёт дальше.
    """
    tmp_dir = locale_dir / "_video_start_images_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_img_path = tmp_dir / f"start_{item.index:04d}.png"

    def cleanup_tmp() -> None:
        try:
            tmp_img_path.unlink(missing_ok=True)
            if tmp_dir.exists() and not any(tmp_dir.iterdir()):
                tmp_dir.rmdir()
        except Exception:
            pass

    last_error: Optional[str] = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            log(f"[VIDEO/FLOW] {out_path.name} | attempt {attempt}/{MAX_ATTEMPTS}")

            # 1) Стартовый кадр из текста.
            image_generation_id = generate_image_to_file(item.prompt, tmp_img_path)

            # 2) Кодируем кадр inline (без storage upload) и передаём в inputs[].
            image_data_uri = image_to_data_uri(tmp_img_path)

            # 3) Оживляем через flow_video_from_ingredients (модель Flow).
            video_generation_id = start_video_from_image(image_data_uri, item.prompt)
            log(f"    video generation_id: {video_generation_id}")

            # 4) Ждём результат.
            video_result_item = poll_generation(video_generation_id, label="VIDEO")
            save_result_item_to_file(video_result_item, out_path)

            cleanup_tmp()
            log(f"    saved: {out_path}")
            return {
                "index": item.index,
                "type": "video",
                "output": str(out_path),
                "status": "success",
                "attempts": attempt,
                "image_generation_id": image_generation_id,
                "generation_id": video_generation_id,
            }
        except KeyboardInterrupt:
            raise
        except PermanentError as e:
            log(f"[VIDEO BLOCKED] {out_path.name}: {e} — пропускаю без повтора")
            cleanup_tmp()
            return {
                "index": item.index,
                "type": "video",
                "output": str(out_path),
                "status": "failed",
                "reason": "permanent",
                "attempts": attempt,
                "error": str(e),
            }
        except Exception as e:
            last_error = str(e)
            log(f"[VIDEO ERROR] {out_path.name}: {e}")
            if attempt < MAX_ATTEMPTS:
                log(f"    retry in {RETRY_DELAY_SEC}s...")
                time.sleep(RETRY_DELAY_SEC)

    cleanup_tmp()
    return {
        "index": item.index,
        "type": "video",
        "output": str(out_path),
        "status": "failed",
        "reason": "transient_exhausted",
        "attempts": MAX_ATTEMPTS,
        "error": last_error,
    }


def write_manifest(locale_dir: Path, rows: List[dict]) -> None:
    manifest_path = locale_dir / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["index", "type", "start_tc", "end_tc", "filename", "prompt"],
        )
        writer.writeheader()
        writer.writerows(rows)


def process_locale(locale: str, items: List[PromptItem], global_log: List[dict], log_lock: Lock) -> None:
    locale_dir = VISUAL_ROOT / locale
    locale_dir.mkdir(parents=True, exist_ok=True)

    log(f"\n===== START {locale} =====")

    manifest_rows: List[dict] = []
    video_jobs: List[tuple[PromptItem, Path]] = []
    image_jobs: List[tuple[PromptItem, Path]] = []

    for item in items:
        is_video = True if ANIMATE_ALL else (item.index <= ANIMATE_FIRST_N)
        filename = output_name(item, is_video=is_video)
        out_path = locale_dir / filename

        manifest_rows.append(
            {
                "index": item.index,
                "type": "video" if is_video else "image",
                "start_tc": item.start_tc,
                "end_tc": item.end_tc,
                "filename": filename,
                "prompt": item.prompt,
            }
        )

        if SKIP_EXISTING and out_path.exists() and out_path.stat().st_size > 0:
            log(f"[SKIP] {locale}/{filename} уже существует")
            with log_lock:
                global_log.append(
                    {
                        "locale": locale,
                        "index": item.index,
                        "type": "video" if is_video else "image",
                        "output": str(out_path),
                        "status": "skipped_existing",
                    }
                )
                save_json(GLOBAL_LOG_FILE, global_log)
            continue

        if is_video:
            video_jobs.append((item, out_path))
        else:
            image_jobs.append((item, out_path))

    write_manifest(locale_dir, manifest_rows)

    log(f"[{locale}] Нужно создать: video={len(video_jobs)}, image={len(image_jobs)}")
    log(f"[{locale}] Потоки: video={MAX_VIDEO_WORKERS}, image={MAX_IMAGE_WORKERS}")

    def save_result(result: dict, item: PromptItem) -> None:
        result["locale"] = locale
        result["start_tc"] = item.start_tc
        result["end_tc"] = item.end_tc
        result["prompt"] = item.prompt
        with log_lock:
            global_log.append(result)
            save_json(GLOBAL_LOG_FILE, global_log)

    if video_jobs:
        workers = max(1, min(MAX_VIDEO_WORKERS, len(video_jobs)))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_job = {
                executor.submit(generate_video_until_done, item, out_path, locale_dir): (item, out_path)
                for item, out_path in video_jobs
            }
            for future in as_completed(future_to_job):
                item, _out_path = future_to_job[future]
                result = future.result()
                save_result(result, item)

    if image_jobs:
        workers = max(1, min(MAX_IMAGE_WORKERS, len(image_jobs)))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_job = {
                executor.submit(generate_image_until_done, item, out_path): (item, out_path)
                for item, out_path in image_jobs
            }
            for future in as_completed(future_to_job):
                item, _out_path = future_to_job[future]
                result = future.result()
                save_result(result, item)

    log(f"===== DONE {locale} =====")


def process_all_locales(prompts_by_locale: Dict[str, List[PromptItem]], global_log: List[dict]) -> None:
    log_lock = Lock()
    active_locales = [locale for locale in LOCALES if locale in prompts_by_locale]

    if not active_locales:
        raise RuntimeError("Нет активных языков с промптами. Генерация остановлена.")

    log(f"[INFO] Активные языки для генерации: {', '.join(active_locales)}")

    if MAX_LOCALE_WORKERS <= 1:
        for locale in active_locales:
            process_locale(locale, prompts_by_locale[locale], global_log, log_lock)
        return

    workers = max(1, min(MAX_LOCALE_WORKERS, len(active_locales)))
    log(f"\n[FAST MODE] Языки генерируются одновременно, locale workers={workers}")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(process_locale, locale, prompts_by_locale[locale], global_log, log_lock): locale
            for locale in active_locales
        }
        for future in as_completed(futures):
            future.result()


def print_summary(global_log: List[dict]) -> None:
    success = [r for r in global_log if r.get("status") == "success"]
    skipped = [r for r in global_log if r.get("status") == "skipped_existing"]
    blocked = [r for r in global_log if r.get("status") == "failed" and r.get("reason") == "permanent"]
    failed_other = [
        r for r in global_log
        if r.get("status") == "failed" and r.get("reason") != "permanent"
    ]

    log("\n========== СВОДКА ==========")
    log(f"  Успешно:               {len(success)}")
    log(f"  Пропущено (существуют): {len(skipped)}")
    log(f"  Заблокировано фильтром: {len(blocked)}")
    log(f"  Прочие ошибки:          {len(failed_other)}")

    if blocked:
        log("\n  Заблокированные провайдером (safety) — стоит переформулировать промпт:")
        for r in blocked:
            name = Path(str(r.get("output", ""))).name
            log(f"    - [{r.get('locale')}] {name}")

    if failed_other:
        log("\n  Не удалось из-за временных ошибок (сеть/лимиты) — перезапусти скрипт:")
        for r in failed_other:
            name = Path(str(r.get("output", ""))).name
            log(f"    - [{r.get('locale')}] {name}: {r.get('error')}")


# =========================
# MAIN
# =========================

def main() -> None:
    global BASE_URL, PROMPTS_ROOT, VISUAL_ROOT, SKIP_EXISTING, GLOBAL_LOG_FILE

    parser = argparse.ArgumentParser(
        description="Generate RU/GE/PL/ES visuals from exact prompt files via media_gen_api V6 (Flow image + Flow video)."
    )
    parser.add_argument(
        "--prompts-root",
        default=str(PROMPTS_ROOT),
        help="Сохранён только для совместимости. Промпты берутся из PROMPT_FILES_BY_LOCALE в CONFIG.",
    )
    parser.add_argument("--visual-root", default=str(VISUAL_ROOT), help="Output root folder")
    parser.add_argument("--api-base", default=BASE_URL, help="API base, default https://api.fast-gen.ai")
    parser.add_argument("--no-skip-existing", action="store_true", help="Regenerate existing files too")
    args = parser.parse_args()

    BASE_URL = args.api_base
    PROMPTS_ROOT = Path(args.prompts_root).expanduser()
    VISUAL_ROOT = Path(args.visual_root).expanduser()
    GLOBAL_LOG_FILE = VISUAL_ROOT / "generation_log.json"
    SKIP_EXISTING = not args.no_skip_existing

    ensure_api_ready()
    ensure_dirs()

    log("media_gen_api V6 image+video visual generator (Flow video, fail-fast)")
    log(f"BASE_URL: {BASE_URL}")
    log(f"Prompts root: {PROMPTS_ROOT}")
    log("Prompt files:")
    for _locale in LOCALES:
        log(f"  {_locale}: {PROMPT_FILES_BY_LOCALE[_locale]}")
    log(f"Visual root: {VISUAL_ROOT}")
    log(f"Locales: {', '.join(LOCALES)}")
    log(f"Animate all prompts as video: {ANIMATE_ALL} (fallback first-N if disabled: {ANIMATE_FIRST_N})")
    log(f"Create endpoint: POST {V6_GENERATIONS_ENDPOINT}")
    log(f"Status endpoint: GET {V6_GENERATION_STATUS_ENDPOINT}")
    log(f"Image operation: {OP_IMAGE_GENERATE}")
    log(f"Video operation: {OP_VIDEO_FROM_IMAGE}  (модель Flow, мягкая модерация)")
    if VIDEO_MODEL:
        log(f"Video model: {VIDEO_MODEL}")
    log("Start image: inline base64 data URI в inputs[] (без storage upload)")
    log(f"Aspect ratio: image={IMAGE_ASPECT_RATIO}, video={VIDEO_ASPECT_RATIO}")
    log(f"Max attempts (сеть/5xx): {MAX_ATTEMPTS}; safety-блок НЕ повторяется")
    if WAIT_ON_RATE_LIMIT:
        log(f"Часовой лимит (429): ЖДУ и повторяю БЕСКОНЕЧНО (пауза {RATE_LIMIT_WAIT_SEC}s или Retry-After)")
    else:
        log("Часовой лимит (429): выключено ожидание — считается обычной временной ошибкой")
    log(f"Speed: image_workers={MAX_IMAGE_WORKERS}, video_workers={MAX_VIDEO_WORKERS}, locale_workers={MAX_LOCALE_WORKERS}")

    prompts_by_locale = load_prompts_by_locale()

    global_log = load_json(GLOBAL_LOG_FILE, [])
    if not isinstance(global_log, list):
        global_log = []

    process_all_locales(prompts_by_locale, global_log)

    save_json(GLOBAL_LOG_FILE, global_log)
    print_summary(global_log)
    log("\nALL DONE")
    log(f"Output: {VISUAL_ROOT}")


if __name__ == "__main__":
    main()
