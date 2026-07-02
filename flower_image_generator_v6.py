#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DE/PL/RU IMAGE generator — Fast-Gen V6, БЕЗ привязки к персонажу.

Переписано строго под актуальную OpenAPI-документацию (media_gen_api / V6)
и оптимизировано под МАКСИМАЛЬНОЕ число параллельных потоков для скорости.

Что нового по сравнению со старой версией:
  • Тела запросов/ответов приведены точно к схемам OpenAPI:
      - POST /api/v6/generations         -> GenerationAcceptedResponse (id, operation, ...)
      - GET  /api/v6/generations/{id}    -> GenerationStatusResponse (status, results[], ...)
      - status enum: queued | running | succeeded | failed
      - results[] -> GenerationResultItem (index, type, download_url, data, text, mime_type, metadata)
  • Убрана неверная проверка поля "success" — V6 его не возвращает.
  • МАКСИМУМ ПОТОКОВ:
      - реальный лимит потоков берётся автоматически из GET /api/v6/usage
        (account_limits.img_generation_threads_allowed);
      - все языки (DE/PL/RU) обрабатываются ОДНИМ общим пулом потоков,
        поэтому потоки не простаивают между языками и всегда загружены под лимит.
  • Аутентификация через заголовок X-API-Key (securityScheme ApiKeyHeader).
  • Остановка при FatalApiError (400/401/403/404/422), чтобы не спамить сотни задач.
  • 429 (rate limit) — не фатально, ретраится с бэкоффом.

Читает промпты из:
  ПРОМПТЫ/prompts_de.txt / prompts_pl.txt / prompts_ru.txt

Сохраняет картинки в:
  ВИЗУАЛ/DE/001.png  ВИЗУАЛ/PL/001.png  ВИЗУАЛ/RU/001.png

Быстрый запуск (максимум потоков автоматически):
  python3 flower_image_generator_v6.py

Флаги:
  --workers auto|N   число потоков. auto = лимит аккаунта из /api/v6/usage. По умолчанию auto.
  --lang DE PL RU    обработать только нужные языки
  --operation flower_image_generate|nano_banana_2_image_generate|
              nano_banana_pro_image_generate|grok_image_generate|openai_image_generate
  --quality speed|quality   режим для grok_image_generate
  --seed N                  фиксированный seed, -1 = случайный
  --upscale                 2x апскейл для nano_banana_* операций
  --no-skip          перегенерировать уже существующие
  --dry-run          только показать промпты, API не вызывать
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import random
import re
import sys
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from threading import Event, Lock
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests


# =========================
# CONFIG
# =========================

# Ключ можно оставить здесь, но безопаснее задавать через:
#   export FAST_GEN_API_KEY='твой_ключ'
API_KEY_HERE = "veo_589a296b7f7e4eb9f81b3549d533454c4ae77e8c868c0f94"

API_KEY = (
    os.getenv("FAST_GEN_API_KEY", "").strip()
    or os.getenv("FASTGEN_API_KEY", "").strip()
    or API_KEY_HERE.strip()
)

BASE_URL = os.getenv("FAST_GEN_API_BASE", "https://api.fast-gen.ai")

PROJECT_ROOT = Path("/Users/aleksandrtomilov/Desktop/ПСИХОЛОГИЯ ГЕРМАНИЯ ПОЛЬША")

PROMPTS_DIR = PROJECT_ROOT / "ПРОМПТЫ"
VISUAL_ROOT = PROJECT_ROOT / "ВИЗУАЛ"

PROMPTS_FILE_BY_LANG: Dict[str, Path] = {
    "DE": PROMPTS_DIR / "prompts_de.txt",
    "PL": PROMPTS_DIR / "prompts_pl.txt",
    "RU": PROMPTS_DIR / "prompts_ru.txt",
}

LOCALES = ["DE", "PL", "RU"]

# V6 endpoints (OpenAPI).
V6_GENERATIONS_ENDPOINT = "/api/v6/generations"
V6_GENERATION_STATUS_ENDPOINT = "/api/v6/generations/{generation_id}"
V6_USAGE_ENDPOINT = "/api/v6/usage"

# Операция по умолчанию: flower_image_generate — flower/flower-image, 1 credit.
V6_OPERATION = (
    os.getenv("FAST_GEN_IMAGE_OPERATION", "").strip()
    or os.getenv("FAST_GEN_V6_OPERATION", "").strip()
    or "flower_image_generate"
)

# Совместимость со старыми переменными/флагами.
IMAGE_ENGINE = os.getenv("FAST_GEN_IMAGE_ENGINE", "").strip().lower()
FLOW_MODEL = os.getenv("FAST_GEN_FLOW_MODEL", "").strip()

ASPECT_RATIO = os.getenv("FAST_GEN_IMAGE_ASPECT_RATIO", "16:9")
# n:n по схеме GenerationCreateRequest.aspect_ratio (pattern ^[1-9]\d*:[1-9]\d*$).
ASPECT_RATIO_RE = re.compile(r"^[1-9]\d*:[1-9]\d*$")

SEED_MAX = 2147483647
_SEED_RAW = os.getenv("FAST_GEN_SEED", "").strip()
IMAGE_SEED: Optional[int] = (
    min(int(_SEED_RAW), SEED_MAX)
    if _SEED_RAW.lstrip("-").isdigit() and int(_SEED_RAW) >= 0
    else None
)

FLOW_UPSCALE_2X = os.getenv("FAST_GEN_FLOW_UPSCALE_2X", "").strip().lower() in {"1", "true", "yes", "on"}

REQUEST_TIMEOUT = int(os.getenv("FAST_GEN_REQUEST_TIMEOUT", "300"))
OPERATION_POLL_SEC = int(os.getenv("FAST_GEN_OPERATION_POLL_SEC", "4"))
OPERATION_TIMEOUT_SEC = int(os.getenv("FAST_GEN_OPERATION_TIMEOUT_SEC", "1800"))
RETRY_DELAY_SEC = int(os.getenv("FAST_GEN_RETRY_DELAY_SEC", "8"))
MAX_RETRIES = int(os.getenv("FAST_GEN_MAX_RETRIES", "0"))  # 0 = бесконечно для НЕфатальных ошибок

# 429 concurrency: короткий джиттер-бэкофф, чтобы воркеры не долбили синхронно.
RATE_LIMIT_RETRY_MIN = float(os.getenv("FAST_GEN_RATE_LIMIT_MIN", "2"))
RATE_LIMIT_RETRY_MAX = float(os.getenv("FAST_GEN_RATE_LIMIT_MAX", "7"))

# "auto" => взять лимит из /api/v6/usage. Иначе фиксированное число.
_WORKERS_RAW = os.getenv("FAST_GEN_IMAGE_WORKERS", "").strip()
MAX_IMAGE_WORKERS: str | int = int(_WORKERS_RAW) if _WORKERS_RAW.isdigit() and int(_WORKERS_RAW) > 0 else "auto"

# Запас по слотам: держим workers НИЖЕ лимита конкурентности, иначе на submit
# постоянно ловим 429 из-за гонки "слот освобождён, но ещё не разрегистрирован".
CONCURRENCY_MARGIN = int(os.getenv("FAST_GEN_CONCURRENCY_MARGIN", "2"))
# Верхний потолок на случай, если API вернёт странно большое число потоков.
WORKERS_HARD_CAP = int(os.getenv("FAST_GEN_WORKERS_HARD_CAP", "64"))
# Фолбэк, если usage недоступен или вернул 0.
WORKERS_FALLBACK = int(os.getenv("FAST_GEN_WORKERS_FALLBACK", "4"))
# ВАЖНО: дешёвые провайдеры (flower) физически не тянут десятки одновременных
# генераций и возвращают "Generation failed" на всё. Поэтому режим 'auto'
# держит СКРОМНУЮ конкурентность, которая реально генерит. Полный лимит аккаунта
# включается явно через --workers max (на свой риск).
AUTO_WORKERS = int(os.getenv("FAST_GEN_AUTO_WORKERS", "6"))
# Сколько раз повторять генерацию, упавшую на стороне сервера ("try again later"),
# прежде чем сдаться по этому промпту (0 = бесконечно).
GEN_FAIL_MAX_RETRIES = int(os.getenv("FAST_GEN_GEN_FAIL_MAX_RETRIES", "6"))

SKIP_EXISTING = True

# Сигнал общей остановки при фатальной ошибке — общий для всего пула.
STOP_EVENT = Event()


# =========================
# DATA
# =========================

@dataclass(frozen=True)
class PromptItem:
    index: int
    prompt: str


@dataclass(frozen=True)
class Job:
    locale: str
    item: PromptItem
    out_path: Path


class FatalApiError(RuntimeError):
    """Фатальная ошибка API: ключ/доступ/endpoint/payload. Повторять сотни задач бессмысленно."""


# =========================
# BASIC HELPERS
# =========================

def log(msg: str) -> None:
    print(msg, flush=True)


def headers(json_content: bool = True) -> Dict[str, str]:
    h = {"X-API-Key": API_KEY}
    if json_content:
        h["Content-Type"] = "application/json"
    return h


def ensure_api_ready() -> None:
    if not API_KEY:
        print("[ERROR] Не найден API ключ Fast-Gen.", file=sys.stderr)
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


def output_path(locale: str, item: PromptItem) -> Path:
    return VISUAL_ROOT / locale / f"{item.index:03d}.png"


def file_ok(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size > 0


def resolve_existing_path(path: Path) -> Path:
    """Ищет файл с учётом NFC/NFD Unicode-нормализации на macOS."""
    path = path.expanduser()
    if path.exists():
        return path
    for form in ("NFC", "NFD"):
        c = Path(unicodedata.normalize(form, str(path))).expanduser()
        if c.exists():
            return c
    parent = path.parent
    if not parent.exists():
        for form in ("NFC", "NFD"):
            cp = Path(unicodedata.normalize(form, str(parent))).expanduser()
            if cp.exists():
                parent = cp
                break
    if parent.exists():
        target_names = {path.name, unicodedata.normalize("NFC", path.name), unicodedata.normalize("NFD", path.name)}
        for child in parent.iterdir():
            child_names = {child.name, unicodedata.normalize("NFC", child.name), unicodedata.normalize("NFD", child.name)}
            if target_names & child_names:
                return child
    return path


# =========================
# PROMPT PARSER
# =========================

def clean_prompt_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return re.sub(r"\s+", " ", text).strip()


PSYCH_HEADER_RE = re.compile(
    r"^\s*(\d{1,5})\s*\|\s*\d{2}:\d{2}:\d{2}[,\.]?\d*\s*-->"
)


def parse_psych_pipeline_format(lines: List[str]) -> List[PromptItem]:
    """Берёт image prompt после строки TEXT: в формате psych_prompt_pipeline."""
    items: List[PromptItem] = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        m = PSYCH_HEADER_RE.match(line)
        if not m:
            i += 1
            continue
        index = int(m.group(1))
        i += 1

        while i < len(lines) and lines[i].strip().upper().startswith("TEXT:"):
            i += 1

        prompt_lines: List[str] = []
        while i < len(lines):
            cur = lines[i].strip()
            if not cur:
                i += 1
                break
            if PSYCH_HEADER_RE.match(cur):
                break
            prompt_lines.append(cur)
            i += 1

        prompt = clean_prompt_text(" ".join(prompt_lines))
        if prompt:
            items.append(PromptItem(index=index, prompt=prompt))
    return items


NUMBERED_RE = re.compile(r"^\s*(\d{1,5})\s*[\.)\]:\-–—]\s*(.+)$")


def parse_numbered_style(lines: List[str]) -> List[PromptItem]:
    items: List[PromptItem] = []
    current_idx: Optional[int] = None
    current_lines: List[str] = []

    def flush() -> None:
        nonlocal current_idx, current_lines
        if current_idx is None:
            return
        prompt = clean_prompt_text(" ".join(current_lines))
        if prompt:
            items.append(PromptItem(index=current_idx, prompt=prompt))
        current_idx = None
        current_lines = []

    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        m = NUMBERED_RE.match(line)
        if m:
            flush()
            current_idx = int(m.group(1))
            rest = m.group(2).strip()
            current_lines = [rest] if rest else []
        elif current_idx is not None:
            current_lines.append(line)

    flush()
    return items


def parse_blank_block_style(text: str) -> List[PromptItem]:
    blocks = [clean_prompt_text(b) for b in re.split(r"\n\s*\n+", text) if clean_prompt_text(b)]
    if len(blocks) <= 1:
        blocks = [clean_prompt_text(x) for x in text.splitlines() if clean_prompt_text(x)]
    return [PromptItem(index=i + 1, prompt=p) for i, p in enumerate(blocks) if p]


def deduplicate_indexes(items: List[PromptItem]) -> List[PromptItem]:
    result: List[PromptItem] = []
    used: set[int] = set()
    next_free = 1
    for item in items:
        idx = item.index
        if idx in used or idx <= 0:
            while next_free in used:
                next_free += 1
            idx = next_free
        used.add(idx)
        result.append(PromptItem(index=idx, prompt=item.prompt))
    return sorted(result, key=lambda x: x.index)


def load_prompts(path: Path) -> List[PromptItem]:
    path = resolve_existing_path(path)
    if not path.exists():
        raise FileNotFoundError(f"Не найден файл промптов: {path}")

    text = path.read_text(encoding="utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
    lines = text.splitlines()

    items = parse_psych_pipeline_format(lines)
    if not items:
        items = parse_numbered_style(lines)
    if not items:
        items = parse_blank_block_style(text)
    if not items:
        raise RuntimeError(f"Не нашёл промпты в файле {path}")

    return deduplicate_indexes(items)


def find_prompts_file(lang: str) -> Optional[Path]:
    ll = lang.lower()
    candidates = [
        PROMPTS_DIR / f"prompts_{ll}.txt",
        PROMPTS_DIR / lang / f"{ll}_{ll}_prompts.txt",
        PROMPTS_DIR / lang.upper() / f"prompts_{ll}.txt",
    ]
    explicit = PROMPTS_FILE_BY_LANG.get(lang.upper())
    if explicit:
        candidates.insert(0, explicit)

    for p in candidates:
        rp = resolve_existing_path(p)
        if rp.exists() and rp.stat().st_size > 0:
            return rp

    if PROMPTS_DIR.exists():
        for p in PROMPTS_DIR.rglob(f"*prompts*{ll}*.txt"):
            if p.stat().st_size > 0:
                return p
    return None


def write_manifest(locale: str, items: Sequence[PromptItem]) -> None:
    locale_dir = VISUAL_ROOT / locale
    locale_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = locale_dir / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["index", "filename", "operation", "prompt"])
        writer.writeheader()
        for item in items:
            writer.writerow({
                "index": item.index,
                "filename": f"{item.index:03d}.png",
                "operation": V6_OPERATION,
                "prompt": item.prompt,
            })


# =========================
# HTTP HELPERS — Fast-Gen V6
# =========================

def normalize_api_base(base: str) -> str:
    """
    Пути в документации уже начинаются с /api/v6/...
    Поэтому BASE_URL должен быть только https://api.fast-gen.ai, без /api/v6.
    """
    base = (base or "https://api.fast-gen.ai").strip().rstrip("/")
    for suffix in ("/api/v6", "/api/v4", "/api"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    return base.rstrip("/")


FATAL_STATUSES = {400, 401, 403, 404, 422}


def should_retry(attempt: int) -> bool:
    return MAX_RETRIES <= 0 or attempt < MAX_RETRIES


def retry_delay() -> float:
    """Джиттер для обычных ретраев, чтобы воркеры не били синхронно."""
    return RETRY_DELAY_SEC + random.uniform(0, min(4.0, float(RETRY_DELAY_SEC)))


def rate_limit_delay(attempt: int) -> float:
    """Короткий бэкофф с джиттером для 429 concurrency."""
    hi = min(RATE_LIMIT_RETRY_MAX, RATE_LIMIT_RETRY_MIN * (1.0 + 0.5 * (attempt - 1)))
    return random.uniform(RATE_LIMIT_RETRY_MIN, max(RATE_LIMIT_RETRY_MIN, hi))


def short_429(resp: requests.Response) -> str:
    try:
        j = resp.json()
        return str(j.get("error") or j.get("code") or "rate_limit")
    except Exception:
        return "rate_limit"


def _safe_payload_for_log(payload: dict) -> dict:
    safe = dict(payload)
    if isinstance(safe.get("prompt"), str) and len(safe["prompt"]) > 220:
        safe["prompt"] = safe["prompt"][:220] + "...[truncated]"
    return safe


def _as_json_object(resp: requests.Response, *, label: str) -> dict:
    try:
        data = resp.json()
    except Exception as e:
        raise RuntimeError(f"{label}: ответ не JSON: {e}: {resp.text[:300]}")
    if not isinstance(data, dict):
        raise RuntimeError(f"{label}: API вернул не JSON-объект: {pretty_json(data)[:300]}")
    return data


def post_json(endpoint: str, payload: dict, *, label: str) -> dict:
    url = normalize_api_base(BASE_URL) + endpoint
    attempt = 0
    while True:
        if STOP_EVENT.is_set():
            raise FatalApiError(f"{label}: остановлено (STOP_EVENT)")
        attempt += 1
        try:
            resp = requests.post(url, headers=headers(), json=payload, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 429:
                if not should_retry(attempt):
                    raise RuntimeError(f"{label}: HTTP 429: {short_429(resp)}")
                delay = rate_limit_delay(attempt)
                if attempt == 1 or attempt % 10 == 0:
                    log(f"[WAIT] {label}: 429 ({short_429(resp)}) — жду слот, retry #{attempt} in {delay:.1f}s")
                time.sleep(delay)
                continue
            if resp.status_code in FATAL_STATUSES:
                raise FatalApiError(
                    f"{label}: HTTP {resp.status_code}: {resp.text}\n"
                    f"URL: {url}\n"
                    f"PAYLOAD: {pretty_json(_safe_payload_for_log(payload))}"
                )
            if resp.status_code >= 400:
                raise RuntimeError(f"{label}: HTTP {resp.status_code}: {resp.text}\nURL: {url}")
            return _as_json_object(resp, label=label)
        except (KeyboardInterrupt, FatalApiError):
            raise
        except Exception as e:
            if not should_retry(attempt):
                raise
            delay = retry_delay()
            log(f"[WARN] {label}: attempt {attempt} failed: {e}, retry in {delay:.1f}s...")
            time.sleep(delay)


def get_json(endpoint: str, *, label: str, params: Optional[dict] = None) -> dict:
    url = normalize_api_base(BASE_URL) + endpoint
    attempt = 0
    while True:
        if STOP_EVENT.is_set():
            raise FatalApiError(f"{label}: остановлено (STOP_EVENT)")
        attempt += 1
        try:
            resp = requests.get(url, headers=headers(json_content=False), params=params, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 429:
                if not should_retry(attempt):
                    raise RuntimeError(f"{label}: HTTP 429: {short_429(resp)}")
                time.sleep(rate_limit_delay(attempt))
                continue
            if resp.status_code in FATAL_STATUSES:
                raise FatalApiError(f"{label}: HTTP {resp.status_code}: {resp.text}\nURL: {url}")
            if resp.status_code >= 400:
                raise RuntimeError(f"{label}: HTTP {resp.status_code}: {resp.text}\nURL: {url}")
            return _as_json_object(resp, label=label)
        except (KeyboardInterrupt, FatalApiError):
            raise
        except Exception as e:
            if not should_retry(attempt):
                raise
            delay = retry_delay()
            log(f"[WARN] {label}: attempt {attempt} failed: {e}, retry in {delay:.1f}s...")
            time.sleep(delay)


def download_file(url: str, path: Path, *, label: str, use_api_key: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    attempt = 0
    while True:
        if STOP_EVENT.is_set():
            raise FatalApiError(f"{label}: остановлено (STOP_EVENT)")
        attempt += 1
        try:
            h = headers(json_content=False) if use_api_key else {}
            resp = requests.get(url, headers=h, timeout=REQUEST_TIMEOUT)
            # download_url может быть подписанным и не требовать ключ.
            if resp.status_code in {401, 403} and use_api_key:
                resp = requests.get(url, timeout=REQUEST_TIMEOUT)
            if resp.status_code in {400, 401, 403, 404}:
                raise FatalApiError(f"{label}: HTTP {resp.status_code}\nURL: {url}")
            if resp.status_code >= 400:
                raise RuntimeError(f"{label}: HTTP {resp.status_code}\nURL: {url}")
            path.write_bytes(resp.content)
            if not file_ok(path):
                raise RuntimeError(f"{label}: пустой файл: {path}")
            return
        except (KeyboardInterrupt, FatalApiError):
            raise
        except Exception as e:
            if not should_retry(attempt):
                raise
            log(f"[WARN] {label}: download attempt {attempt} failed: {e}, retry in {RETRY_DELAY_SEC}s...")
            time.sleep(RETRY_DELAY_SEC)


# =========================
# ACCOUNT LIMITS / THREADS
# =========================

def fetch_image_thread_limit() -> Optional[int]:
    """
    GET /api/v6/usage -> UsageResponse.account_limits.img_generation_threads_allowed
    Возвращает разрешённое число одновременных image-потоков, либо None.
    """
    try:
        data = get_json(V6_USAGE_ENDPOINT, label="V6 USAGE")
    except FatalApiError as e:
        log(f"[WARN] usage endpoint недоступен: {e}")
        return None
    except Exception as e:
        log(f"[WARN] не удалось получить usage: {e}")
        return None

    limits = data.get("account_limits") or {}
    threads = limits.get("img_generation_threads_allowed")
    hourly = limits.get("img_gen_per_hour_limit")
    try:
        threads = int(threads)
    except (TypeError, ValueError):
        return None
    log(f"    account_limits: img_threads_allowed={threads}, img_per_hour={hourly}")
    return threads if threads > 0 else None


def resolve_workers(requested: str | int) -> int:
    """Определяет итоговое число потоков с учётом лимитов аккаунта."""
    if isinstance(requested, int) and requested > 0:
        workers = requested
        log(f"[THREADS] задано вручную: {workers}")
    else:
        limit = fetch_image_thread_limit()
        # Полный лимит аккаунта минус запас на гонку освобождения слота.
        full = max(1, (limit - CONCURRENCY_MARGIN)) if limit else WORKERS_FALLBACK
        if requested == "max":
            workers = full
            log(f"[THREADS] MAX: limit={limit}, margin={CONCURRENCY_MARGIN} -> workers={workers} "
                f"(осторожно: дешёвые модели могут массово падать)")
        else:  # 'auto' — скромно и надёжно
            workers = min(full, AUTO_WORKERS)
            log(f"[THREADS] AUTO: limit={limit} -> workers={workers} "
                f"(для полного лимита используй --workers max)")
    workers = max(1, min(workers, WORKERS_HARD_CAP))
    return workers


# =========================
# RESULT DECODING — GenerationResultItem
# =========================

def split_data_uri(data_uri: str) -> Tuple[str, bytes]:
    if not data_uri.startswith("data:") or "," not in data_uri:
        raise ValueError(f"Некорректный data URI: {data_uri[:80]}")
    header, b64 = data_uri.split(",", 1)
    mime = header.replace("data:", "").replace(";base64", "")
    return mime, base64.b64decode(b64)


def save_data_uri(data_uri: str, path: Path) -> None:
    _, raw = split_data_uri(data_uri)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    if not file_ok(path):
        raise RuntimeError(f"Файл не сохранился: {path}")


def save_v6_results(results: Any, out_path: Path) -> None:
    """
    GenerationStatusResponse.results: [GenerationResultItem]
      { index, type, download_url, data, text, mime_type, metadata }
    Берём первый image-result: приоритет download_url, затем inline data.
    """
    if not isinstance(results, list) or not results:
        raise RuntimeError(f"V6 result пустой или неожиданный: {pretty_json(results)}")

    item = None
    for r in results:
        if isinstance(r, dict) and (r.get("type") == "image" or r.get("download_url") or r.get("data")):
            item = r
            break
    if not isinstance(item, dict):
        raise RuntimeError(f"Не нашёл image-result в V6 results: {pretty_json(results)}")

    download_url = item.get("download_url")
    data_uri = item.get("data")

    # Storage-backed медиа: приоритетно скачиваем по download_url.
    if isinstance(download_url, str) and download_url:
        download_file(download_url, out_path, label="V6 RESULT download_url", use_api_key=True)
        return

    if isinstance(data_uri, str) and data_uri.startswith("data:"):
        save_data_uri(data_uri, out_path)
        return

    # запасной вариант: raw base64 без data: prefix
    if isinstance(data_uri, str) and len(data_uri) > 100:
        mime = item.get("mime_type") or "image/png"
        save_data_uri(f"data:{mime};base64,{data_uri}", out_path)
        return

    raise RuntimeError(f"Неизвестный формат V6 image result: {pretty_json(item)}")


# =========================
# V6 TEXT-TO-IMAGE, БЕЗ ПЕРСОНАЖА
# =========================

ALLOWED_V6_IMAGE_OPERATIONS = {
    "flower_image_generate",
    "nano_banana_2_image_generate",
    "nano_banana_pro_image_generate",
    "grok_image_generate",
    "openai_image_generate",
}


def _resolve_aspect_ratio() -> str:
    if ASPECT_RATIO_RE.match(ASPECT_RATIO or ""):
        return ASPECT_RATIO
    log(f"[WARN] aspect_ratio {ASPECT_RATIO!r} не соответствует n:n — использую '16:9'.")
    return "16:9"


def legacy_engine_to_operation(engine: str, flow_model: str = "") -> str:
    """Поддержка старого --engine."""
    e = (engine or "").strip().lower()
    fm = (flow_model or "").strip().upper()
    if e == "flower":
        return "flower_image_generate"
    if e == "flow":
        if fm == "GEM_PIX_2":
            return "nano_banana_pro_image_generate"
        if fm == "IMAGEN_3_5":
            return "openai_image_generate"
        return "nano_banana_2_image_generate"
    if e in {"nano2", "nano_banana_2"}:
        return "nano_banana_2_image_generate"
    if e in {"nanopro", "nano_banana_pro"}:
        return "nano_banana_pro_image_generate"
    if e == "grok":
        return "grok_image_generate"
    if e == "openai":
        return "openai_image_generate"
    return V6_OPERATION


def build_payload(prompt: str) -> Tuple[str, Dict[str, Any], str]:
    """Возвращает endpoint, payload и label для V6 text-to-image (GenerationCreateRequest)."""
    cleaned = clean_prompt_text(prompt)

    operation = V6_OPERATION
    if operation not in ALLOWED_V6_IMAGE_OPERATIONS:
        raise FatalApiError(
            f"Недопустимая image operation: {operation!r}. "
            f"Разрешено: {', '.join(sorted(ALLOWED_V6_IMAGE_OPERATIONS))}"
        )

    payload: Dict[str, Any] = {
        "operation": operation,
        "prompt": cleaned,
        "aspect_ratio": _resolve_aspect_ratio(),
    }

    if IMAGE_SEED is not None:
        payload["seed"] = min(max(0, IMAGE_SEED), SEED_MAX)

    if operation == "grok_image_generate":
        q = os.getenv("FAST_GEN_IMAGE_QUALITY", "").strip().lower()
        if q in {"speed", "quality"}:
            payload["quality"] = q

    if operation in {"nano_banana_2_image_generate", "nano_banana_pro_image_generate"} and FLOW_UPSCALE_2X:
        payload["generation_config"] = {"upscale": {"type": "2x"}}

    return V6_GENERATIONS_ENDPOINT, payload, f"V6 {operation}"


def submit_generate(prompt: str) -> Tuple[str, str]:
    endpoint, payload, label = build_payload(prompt)
    data = post_json(endpoint, payload, label=label)

    generation_id = data.get("id")
    if not generation_id:
        raise RuntimeError(f"API не вернул id генерации: {pretty_json(data)}")

    op = str(data.get("operation") or payload.get("operation") or "")
    return str(generation_id), op


def poll_operation(generation_id: str) -> dict:
    endpoint = V6_GENERATION_STATUS_ENDPOINT.format(generation_id=generation_id)
    started = time.time()
    while True:
        data = get_json(endpoint, label=f"V6 GENERATION {generation_id}")
        state = data.get("status")

        if state in {"queued", "running"}:
            if time.time() - started > OPERATION_TIMEOUT_SEC:
                raise RuntimeError(f"Timeout {generation_id} after {OPERATION_TIMEOUT_SEC}s")
            time.sleep(OPERATION_POLL_SEC)
            continue

        if state == "succeeded":
            if not data.get("results"):
                raise RuntimeError(f"generation {generation_id}: succeeded, но results пустой")
            for w in (data.get("warnings") or []):
                log(f"    [WARN] generation {generation_id}: {w}")
            return data

        # failed или неожиданный статус
        err = data.get("error") or pretty_json(data)
        translations = data.get("translations") or {}
        if isinstance(translations, dict) and translations:
            tr = translations.get("ru") or next(iter(translations.values()), None)
            if tr:
                err = f"{err} | {tr}"
        raise RuntimeError(f"generation {generation_id}: {state}: {err}")


def generate_image(job: Job) -> dict:
    item, out_path, locale = job.item, job.out_path, job.locale
    attempt = 0
    while True:
        if STOP_EVENT.is_set():
            raise FatalApiError("остановлено (STOP_EVENT)")
        attempt += 1
        generation_id = None
        try:
            log(f"[{locale}] {out_path.name} (#{item.index}) attempt {attempt}")
            generation_id, operation = submit_generate(item.prompt)
            log(f"    generation_id: {generation_id}" + (f" ({operation})" if operation else ""))
            op_data = poll_operation(generation_id)
            save_v6_results(op_data.get("results"), out_path)
            log(f"    saved: {out_path}")
            return {
                "index": item.index,
                "locale": locale,
                "status": "success",
                "output": str(out_path),
                "generation_id": generation_id,
                "operation": operation,
                "attempts": attempt,
            }
        except (KeyboardInterrupt, FatalApiError):
            raise
        except Exception as e:
            # Транзиентные server-side падения ("Generation failed, please try
            # again later") повторяем, но не бесконечно — иначе один битый промпт
            # держит поток вечно.
            if GEN_FAIL_MAX_RETRIES > 0 and attempt >= GEN_FAIL_MAX_RETRIES:
                log(f"[GIVE-UP] [{locale}] #{item.index}: {e} (после {attempt} попыток)")
                raise
            delay = min(60.0, retry_delay() * (1.5 ** (attempt - 1)))
            log(f"[ERROR] [{locale}] #{item.index}: {e}, retry {attempt} in {delay:.1f}s...")
            time.sleep(delay)


# =========================
# SMOKE TEST — какая модель реально работает
# =========================

def probe_operation(operation: str, prompt: str, *, timeout: int = 120) -> Tuple[bool, str]:
    """Одна генерация через указанную operation, БЕЗ бесконечных ретраев. (ok, detail)."""
    url = normalize_api_base(BASE_URL) + V6_GENERATIONS_ENDPOINT
    payload: Dict[str, Any] = {
        "operation": operation,
        "prompt": clean_prompt_text(prompt),
        "aspect_ratio": _resolve_aspect_ratio(),
    }
    if operation == "grok_image_generate":
        payload["quality"] = os.getenv("FAST_GEN_IMAGE_QUALITY", "speed").strip().lower() or "speed"
    try:
        r = requests.post(url, headers=headers(), json=payload, timeout=REQUEST_TIMEOUT)
    except Exception as e:
        return False, f"POST error: {e}"
    if r.status_code != 200:
        return False, f"HTTP {r.status_code}: {r.text[:200]}"
    try:
        gid = r.json().get("id")
    except Exception:
        return False, f"нет id в ответе: {r.text[:200]}"
    if not gid:
        return False, f"нет id в ответе: {r.text[:200]}"

    status_url = normalize_api_base(BASE_URL) + V6_GENERATION_STATUS_ENDPOINT.format(generation_id=gid)
    started = time.time()
    while time.time() - started < timeout:
        time.sleep(3)
        try:
            s = requests.get(status_url, headers=headers(json_content=False), timeout=REQUEST_TIMEOUT)
        except Exception as e:
            return False, f"GET error: {e}"
        if s.status_code == 429:
            continue
        if s.status_code != 200:
            return False, f"status HTTP {s.status_code}: {s.text[:200]}"
        body = s.json()
        st = body.get("status")
        if st == "succeeded":
            n = len(body.get("results") or [])
            return True, f"succeeded, results={n}"
        if st == "failed":
            return False, f"failed: {body.get('error') or body}"
    return False, f"timeout {timeout}s (последний статус не succeeded/failed)"


def run_smoke_test(sample_prompt: str) -> None:
    log("\n" + "=" * 50)
    log("SMOKE TEST — по одной генерации на каждую модель")
    log(f"Промпт: {sample_prompt[:80]}")
    log("=" * 50)
    ops = [
        "flower_image_generate",
        "nano_banana_2_image_generate",
        "nano_banana_pro_image_generate",
        "grok_image_generate",
        "openai_image_generate",
    ]
    results: List[Tuple[str, bool, str]] = []
    for op in ops:
        log(f"\n-> {op} ...")
        ok, detail = probe_operation(op, sample_prompt)
        mark = "✅ OK" if ok else "❌ FAIL"
        log(f"   {mark}: {detail}")
        results.append((op, ok, detail))
    log("\n" + "=" * 50)
    log("ИТОГ:")
    working = [op for op, ok, _ in results if ok]
    for op, ok, detail in results:
        log(f"  {'✅' if ok else '❌'} {op}")
    if working:
        log(f"\nРабочие модели: {', '.join(working)}")
        log(f"Запускай так:  python3 flower_image_generator_v6.py --operation {working[0]}")
    else:
        log("\nНи одна модель не отдала картинку — проблема на стороне аккаунта/сервиса, "
            "а не в скрипте. Проверь баланс кредитов и статус Fast-Gen.")
    log("=" * 50)


# =========================
# GLOBAL SCHEDULING — один пул на все языки для максимума потоков
# =========================

def build_jobs(jobs_by_lang: List[Tuple[str, List[PromptItem]]]) -> List[Job]:
    """Собирает плоский список заданий по всем языкам, пропуская уже готовые."""
    jobs: List[Job] = []
    for locale, items in jobs_by_lang:
        pending = 0
        for item in items:
            p = output_path(locale, item)
            if SKIP_EXISTING and file_ok(p):
                continue
            jobs.append(Job(locale=locale, item=item, out_path=p))
            pending += 1
        log(f"  {locale}: к генерации {pending} из {len(items)}")
    # Чередуем языки, чтобы прогресс шёл равномерно по DE/PL/RU.
    jobs.sort(key=lambda j: (j.item.index, j.locale))
    return jobs


def run_all(jobs: List[Job], workers: int, global_log: List[dict], log_lock: Lock) -> None:
    if not jobs:
        log("  [OK] все картинки уже существуют")
        return

    workers = max(1, min(workers, len(jobs)))
    log(f"\n{'=' * 42}")
    log(f"  ЕДИНЫЙ ПУЛ: {len(jobs)} заданий | workers={workers}")
    log(f"{'=' * 42}")

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(generate_image, job): job for job in jobs}
        try:
            for future in as_completed(futures):
                job = futures[future]
                item, locale = job.item, job.locale
                try:
                    result = future.result()
                    result["prompt"] = item.prompt
                    with log_lock:
                        global_log.append(result)
                        save_json(VISUAL_ROOT / "generation_log.json", global_log)
                except FatalApiError as e:
                    STOP_EVENT.set()
                    for f in futures:
                        f.cancel()
                    log(f"  [FATAL] [{locale}] #{item.index}: {e}")
                    with log_lock:
                        global_log.append({
                            "index": item.index,
                            "locale": locale,
                            "status": "fatal_error",
                            "error": str(e),
                        })
                        save_json(VISUAL_ROOT / "generation_log.json", global_log)
                    raise
                except Exception as e:
                    log(f"  [FAILED] [{locale}] #{item.index}: {e}")
                    with log_lock:
                        global_log.append({
                            "index": item.index,
                            "locale": locale,
                            "status": "error",
                            "error": str(e),
                        })
                        save_json(VISUAL_ROOT / "generation_log.json", global_log)
        except (FatalApiError, KeyboardInterrupt):
            STOP_EVENT.set()
            raise


# =========================
# MAIN
# =========================

def _parse_workers_arg(value: str) -> str | int:
    v = (value or "").strip().lower()
    if v in {"auto", ""}:
        return "auto"
    if v in {"max", "full"}:
        return "max"
    if v.isdigit() and int(v) > 0:
        return int(v)
    raise argparse.ArgumentTypeError("workers должно быть 'auto', 'max' или положительным числом")


def main() -> None:
    global BASE_URL, SKIP_EXISTING, ASPECT_RATIO, MAX_IMAGE_WORKERS
    global OPERATION_POLL_SEC, PROMPTS_DIR, VISUAL_ROOT
    global V6_OPERATION, IMAGE_ENGINE, FLOW_MODEL, IMAGE_SEED, FLOW_UPSCALE_2X

    parser = argparse.ArgumentParser(
        description="Генерирует DE/PL/RU картинки из текстовых промптов через Fast-Gen V6 /api/v6/generations, "
                    "один общий пул потоков на максимальной скорости."
    )
    parser.add_argument("--lang", nargs="+", choices=LOCALES, default=None,
                        help="Языки для обработки. По умолчанию — все три.")
    parser.add_argument("--prompts-dir", default=str(PROMPTS_DIR),
                        help="Папка с prompts_de.txt / prompts_pl.txt / prompts_ru.txt")
    parser.add_argument("--visual-root", default=str(VISUAL_ROOT), help="Папка ВИЗУАЛ")
    parser.add_argument("--operation", choices=sorted(ALLOWED_V6_IMAGE_OPERATIONS), default=V6_OPERATION,
                        help="V6 operation. По умолчанию flower_image_generate = 1 credit.")
    parser.add_argument("--engine", choices=["flower", "flow", "nano2", "nanopro", "grok", "openai"], default=None,
                        help="Старый совместимый флаг. flower -> flower_image_generate; flow -> nano_banana_2/pro.")
    parser.add_argument("--flow-model", default=FLOW_MODEL or "NARWHAL",
                        choices=["NARWHAL", "GEM_PIX_2", "IMAGEN_3_5"],
                        help="Только для совместимости со старым --engine flow.")
    parser.add_argument("--seed", type=int, default=(IMAGE_SEED if IMAGE_SEED is not None else -1),
                        help=f"Фиксированный seed, 0..{SEED_MAX}. -1 = случайный.")
    parser.add_argument("--quality", choices=["speed", "quality"],
                        default=os.getenv("FAST_GEN_IMAGE_QUALITY", "").strip().lower() or None,
                        help="Качество для grok_image_generate: speed или quality.")
    parser.add_argument("--upscale", action="store_true", default=FLOW_UPSCALE_2X,
                        help="2x upscale для nano_banana_* операций. Удваивает кредиты.")
    parser.add_argument("--api-base", default=BASE_URL)
    parser.add_argument("--aspect-ratio", default=ASPECT_RATIO,
                        help="Соотношение сторон n:n, например 16:9, 9:16, 1:1, 4:3, 3:4.")
    parser.add_argument("--workers", type=_parse_workers_arg, default=MAX_IMAGE_WORKERS,
                        help="Потоки: 'auto' = скромно и надёжно (flower реально генерит), "
                             "'max' = полный лимит аккаунта (дешёвые модели могут массово падать), "
                             "или число. По умолчанию auto.")
    parser.add_argument("--poll-sec", type=int, default=OPERATION_POLL_SEC)
    parser.add_argument("--no-skip", action="store_true", help="Перегенерировать уже существующие")
    parser.add_argument("--dry-run", action="store_true", help="Показать промпты без API-вызовов")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Прогнать по 1 генерации на каждую модель и показать, какая реально работает.")
    args = parser.parse_args()

    BASE_URL = normalize_api_base(args.api_base)
    PROMPTS_DIR = Path(args.prompts_dir).expanduser()
    VISUAL_ROOT = Path(args.visual_root).expanduser()
    ASPECT_RATIO = args.aspect_ratio
    MAX_IMAGE_WORKERS = args.workers
    OPERATION_POLL_SEC = max(1, args.poll_sec)
    SKIP_EXISTING = not args.no_skip
    IMAGE_ENGINE = args.engine or IMAGE_ENGINE
    FLOW_MODEL = args.flow_model
    V6_OPERATION = legacy_engine_to_operation(args.engine, args.flow_model) if args.engine else args.operation
    IMAGE_SEED = min(args.seed, SEED_MAX) if args.seed is not None and args.seed >= 0 else None
    FLOW_UPSCALE_2X = bool(args.upscale)
    if args.quality:
        os.environ["FAST_GEN_IMAGE_QUALITY"] = args.quality

    langs_to_process = args.lang or LOCALES

    ensure_dirs()

    print("\n=== План работ ===")
    jobs_by_lang: List[Tuple[str, List[PromptItem]]] = []
    for lang in langs_to_process:
        pf = find_prompts_file(lang)
        ok = pf is not None and pf.exists()
        if ok:
            try:
                items = load_prompts(pf)
                jobs_by_lang.append((lang, items))
                print(f"  {lang}: {len(items)} промптов из {pf.name} -> {VISUAL_ROOT / lang}/")
            except Exception as e:
                print(f"  {lang}: ❌ ошибка загрузки промптов: {e}")
        else:
            print(f"  {lang}: ❌ файл промптов не найден в {PROMPTS_DIR}")
    print("==================\n")

    if not jobs_by_lang:
        print("❌ Не найдено ни одного файла промптов.", file=sys.stderr)
        print(f"Ожидаемые файлы: prompts_de.txt / prompts_pl.txt / prompts_ru.txt в {PROMPTS_DIR}", file=sys.stderr)
        sys.exit(1)

    seed_txt = IMAGE_SEED if IMAGE_SEED is not None else "random"
    extra_parts = [f"operation={V6_OPERATION}", f"seed={seed_txt}"]
    if FLOW_UPSCALE_2X:
        extra_parts.append("upscale=2x")
    q = os.getenv("FAST_GEN_IMAGE_QUALITY", "").strip()
    if q:
        extra_parts.append(f"quality={q}")
    extra = " | " + ", ".join(extra_parts)

    if args.dry_run:
        log("\nDRY RUN — API не вызывается")
        log(f"Fast-Gen V6{extra} | text-to-image, без персонажа | aspect_ratio={_resolve_aspect_ratio()}")
        for lang, items in jobs_by_lang:
            log(f"\n[{lang}] первые 3 промпта:")
            for item in items[:3]:
                log(f"  {item.index:03d}.png <- {clean_prompt_text(item.prompt)[:160]}...")
        return

    ensure_api_ready()

    if args.smoke_test:
        sample = next((it.prompt for _, items in jobs_by_lang for it in items),
                      "a simple flat cartoon of a calm young man in an olive green hoodie")
        run_smoke_test(sample)
        return

    log(f"Fast-Gen V6{extra} | text-to-image, без персонажа | aspect_ratio={_resolve_aspect_ratio()}")

    # Максимум потоков.
    workers = resolve_workers(MAX_IMAGE_WORKERS)

    for lang, items in jobs_by_lang:
        write_manifest(lang, items)

    jobs = build_jobs(jobs_by_lang)

    global_log = load_json(VISUAL_ROOT / "generation_log.json", [])
    if not isinstance(global_log, list):
        global_log = []
    log_lock = Lock()

    try:
        run_all(jobs, workers, global_log, log_lock)
    except FatalApiError as e:
        print("\n" + "=" * 42, file=sys.stderr)
        print("FATAL API ERROR — запуск остановлен", file=sys.stderr)
        print(str(e), file=sys.stderr)
        print("Проверь API-ключ, FAST_GEN_API_BASE и доступ к V6 endpoints.", file=sys.stderr)
        print("=" * 42, file=sys.stderr)
        sys.exit(2)

    print(f"\n{'=' * 42}")
    print("ALL DONE")
    for lang, _ in jobs_by_lang:
        count = len(list((VISUAL_ROOT / lang).glob("*.png")))
        print(f"  {lang}: {count} картинок в {VISUAL_ROOT / lang}/")
    print(f"{'=' * 42}")


if __name__ == "__main__":
    main()
