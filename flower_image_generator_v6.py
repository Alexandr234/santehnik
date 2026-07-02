#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DE/PL/RU IMAGE generator — Fast-Gen V6, ЧИСТО ПО ТЕКСТУ (без персонажа/референса), максимум потоков.

Важно про модели (по OpenAPI V6, api1_11):
  • flower НЕ генерит из текста — это модель РЕДАКТИРОВАНИЯ картинки
    (flower_image_generate из текста => "Generation failed"). Поэтому flower тут НЕ используется.
  • Из текста умеют:
      grok_image_generate            — grok/grok-image, 1 кредит (speed)
      openai_image_generate          — openai/openai-image, 2 кредита
      nano_banana_2_image_generate   — flow/nano-banana-2, 4 кредита
      nano_banana_pro_image_generate — flow/nano-banana-pro, 4 кредита

Чтобы не гадать, какая модель жива на аккаунте, скрипт при старте делает по 1 пробной
генерации (автоподбор) и дальше гонит всё на первой рабочей. Можно жёстко задать --operation.

Строго по V6:
  • POST /api/v6/generations       (operation, prompt, aspect_ratio, seed, quality, generation_config)
  • GET  /api/v6/generations/{id}  (status queued|running|succeeded|failed, results[])
      results[i]: {type, download_url, data, mime_type}

Скорость: все языки (DE/PL/RU) в одном общем пуле потоков; число потоков из лимита
аккаунта (GET /api/v6/usage -> img_generation_threads_allowed).

Быстрый запуск:
  python3 flower_image_generator_v6.py

Флаги:
  --operation grok_image_generate|openai_image_generate|nano_banana_2_image_generate|nano_banana_pro_image_generate
              жёстко выбрать модель (по умолчанию — автоподбор рабочей)
  --workers auto|max|N     потоки. auto = лимит аккаунта − запас (по умолчанию)
  --lang DE PL RU          только нужные языки
  --quality speed|quality  для grok
  --seed N                 фикс. seed (nano/openai), -1 = случайный
  --upscale                nano_banana_*: 2x-апскейл (удваивает кредиты)
  --aspect-ratio 16:9|9:16|1:1|4:3|3:4
  --no-skip                перегенерировать существующие
  --dry-run                показать промпты, API не вызывать
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

V6_GENERATIONS_ENDPOINT = "/api/v6/generations"
V6_GENERATION_STATUS_ENDPOINT = "/api/v6/generations/{generation_id}"
V6_USAGE_ENDPOINT = "/api/v6/usage"

# Модели, умеющие text-to-image. flower исключён (он только edit).
TEXT2IMAGE_OPERATIONS = [
    "grok_image_generate",            # 1 кредит (speed)
    "openai_image_generate",          # 2 кредита
    "nano_banana_2_image_generate",   # 4 кредита
    "nano_banana_pro_image_generate", # 4 кредита
]

# Текущая рабочая операция (выбирается автоподбором или флагом --operation).
V6_OPERATION = os.getenv("FAST_GEN_IMAGE_OPERATION", "").strip()

ASPECT_RATIO = os.getenv("FAST_GEN_IMAGE_ASPECT_RATIO", "16:9")
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
OPERATION_POLL_SEC = int(os.getenv("FAST_GEN_OPERATION_POLL_SEC", "5"))
OPERATION_TIMEOUT_SEC = int(os.getenv("FAST_GEN_OPERATION_TIMEOUT_SEC", "1800"))
RETRY_DELAY_SEC = int(os.getenv("FAST_GEN_RETRY_DELAY_SEC", "8"))
MAX_RETRIES = int(os.getenv("FAST_GEN_MAX_RETRIES", "0"))

RATE_LIMIT_RETRY_MIN = float(os.getenv("FAST_GEN_RATE_LIMIT_MIN", "2"))
RATE_LIMIT_RETRY_MAX = float(os.getenv("FAST_GEN_RATE_LIMIT_MAX", "7"))

_WORKERS_RAW = os.getenv("FAST_GEN_IMAGE_WORKERS", "").strip()
MAX_IMAGE_WORKERS: str | int = int(_WORKERS_RAW) if _WORKERS_RAW.isdigit() and int(_WORKERS_RAW) > 0 else "auto"
CONCURRENCY_MARGIN = int(os.getenv("FAST_GEN_CONCURRENCY_MARGIN", "2"))
WORKERS_HARD_CAP = int(os.getenv("FAST_GEN_WORKERS_HARD_CAP", "64"))
WORKERS_FALLBACK = int(os.getenv("FAST_GEN_WORKERS_FALLBACK", "6"))

GEN_FAIL_MAX_RETRIES = int(os.getenv("FAST_GEN_GEN_FAIL_MAX_RETRIES", "6"))

SKIP_EXISTING = True
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
    """Фатальная ошибка API: ключ/доступ/endpoint/payload — повторять бессмысленно."""


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

FATAL_STATUSES = {400, 401, 403, 404, 422}


def _as_json_object(resp: requests.Response, *, label: str) -> dict:
    try:
        data = resp.json()
    except Exception as e:
        raise RuntimeError(f"{label}: ответ не JSON: {e}: {resp.text[:300]}")
    if not isinstance(data, dict):
        raise RuntimeError(f"{label}: API вернул не JSON-объект: {pretty_json(data)[:300]}")
    if data.get("success") is False:
        raise RuntimeError(f"{label}: API success=false: {data.get('error') or pretty_json(data)}")
    return data


def should_retry(attempt: int) -> bool:
    return MAX_RETRIES <= 0 or attempt < MAX_RETRIES


def retry_delay() -> float:
    return RETRY_DELAY_SEC + random.uniform(0, min(4.0, float(RETRY_DELAY_SEC)))


def rate_limit_delay(attempt: int) -> float:
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
    if isinstance(safe.get("prompt"), str) and len(safe["prompt"]) > 200:
        safe["prompt"] = safe["prompt"][:200] + "...[truncated]"
    return safe


def post_json(endpoint: str, payload: dict, *, label: str) -> dict:
    url = BASE_URL.rstrip("/") + endpoint
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
                    f"URL: {url}\nPAYLOAD: {pretty_json(_safe_payload_for_log(payload))}"
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
    url = BASE_URL.rstrip("/") + endpoint
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
            delay = retry_delay()
            log(f"[WARN] {label}: download attempt {attempt} failed: {e}, retry in {delay:.1f}s...")
            time.sleep(delay)


# =========================
# ACCOUNT LIMITS / THREADS
# =========================

def fetch_image_thread_limit() -> Optional[int]:
    try:
        data = get_json(V6_USAGE_ENDPOINT, label="USAGE")
    except Exception as e:
        log(f"[WARN] usage недоступен: {e}")
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
    if isinstance(requested, int) and requested > 0:
        workers = requested
        log(f"[THREADS] задано вручную: {workers}")
    else:
        limit = fetch_image_thread_limit()
        full = max(1, (limit - CONCURRENCY_MARGIN)) if limit else WORKERS_FALLBACK
        if requested == "max":
            workers = limit if limit else WORKERS_FALLBACK
            log(f"[THREADS] MAX: limit={limit} -> workers={workers}")
        else:
            workers = full
            log(f"[THREADS] AUTO: limit={limit}, margin={CONCURRENCY_MARGIN} -> workers={workers}")
    return max(1, min(workers, WORKERS_HARD_CAP))


# =========================
# RESULT DECODING
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
    if isinstance(download_url, str) and download_url:
        download_file(download_url, out_path, label="V6 RESULT download_url", use_api_key=True)
        return
    if isinstance(data_uri, str) and data_uri.startswith("data:"):
        save_data_uri(data_uri, out_path)
        return
    if isinstance(data_uri, str) and len(data_uri) > 100:
        mime = item.get("mime_type") or "image/png"
        save_data_uri(f"data:{mime};base64,{data_uri}", out_path)
        return
    raise RuntimeError(f"Неизвестный формат V6 image result: {pretty_json(item)}")


# =========================
# V6 TEXT-TO-IMAGE
# =========================

def _resolve_aspect_ratio() -> str:
    if ASPECT_RATIO_RE.match(ASPECT_RATIO or ""):
        return ASPECT_RATIO
    log(f"[WARN] aspect_ratio {ASPECT_RATIO!r} не n:n — использую '16:9'.")
    return "16:9"


def build_payload(prompt: str, operation: str) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "operation": operation,
        "prompt": clean_prompt_text(prompt),
        "aspect_ratio": _resolve_aspect_ratio(),
    }
    if operation == "grok_image_generate":
        q = os.getenv("FAST_GEN_IMAGE_QUALITY", "speed").strip().lower() or "speed"
        if q in {"speed", "quality"}:
            payload["quality"] = q
    if IMAGE_SEED is not None and operation != "grok_image_generate":
        payload["seed"] = min(max(0, IMAGE_SEED), SEED_MAX)
    if operation in {"nano_banana_2_image_generate", "nano_banana_pro_image_generate"} and FLOW_UPSCALE_2X:
        payload["generation_config"] = {"upscale": {"type": "2x"}}
    return payload


def submit_generate(prompt: str) -> Tuple[str, str]:
    payload = build_payload(prompt, V6_OPERATION)
    data = post_json(V6_GENERATIONS_ENDPOINT, payload, label=f"V6 {V6_OPERATION}")
    generation_id = data.get("id")
    if not generation_id:
        raise RuntimeError(f"API не вернул id генерации: {pretty_json(data)}")
    op = str(data.get("operation") or payload.get("operation") or "")
    return str(generation_id), op


def poll_operation(generation_id: str) -> dict:
    endpoint = V6_GENERATION_STATUS_ENDPOINT.format(generation_id=generation_id)
    started = time.time()
    while True:
        data = get_json(endpoint, label=f"V6 GEN {generation_id}")
        state = data.get("status")
        if state in {"queued", "running"}:
            if time.time() - started > OPERATION_TIMEOUT_SEC:
                raise RuntimeError(f"Timeout {generation_id} after {OPERATION_TIMEOUT_SEC}s")
            time.sleep(OPERATION_POLL_SEC)
            continue
        if state == "succeeded":
            if not data.get("results"):
                raise RuntimeError(f"gen {generation_id}: succeeded, но results пустой")
            for w in (data.get("warnings") or []):
                log(f"    [WARN] gen {generation_id}: {w}")
            return data
        err = data.get("error") or pretty_json(data)
        translations = data.get("translations") or {}
        if isinstance(translations, dict) and translations:
            tr = translations.get("ru") or next(iter(translations.values()), None)
            if tr:
                err = f"{err} | {tr}"
        raise RuntimeError(f"gen {generation_id}: {state}: {err}")


def generate_image(job: Job) -> dict:
    item, out_path, locale = job.item, job.out_path, job.locale
    attempt = 0
    while True:
        if STOP_EVENT.is_set():
            raise FatalApiError("остановлено (STOP_EVENT)")
        attempt += 1
        try:
            log(f"[{locale}] {out_path.name} (#{item.index}) attempt {attempt}")
            generation_id, operation = submit_generate(item.prompt)
            log(f"    generation_id: {generation_id}" + (f" ({operation})" if operation else ""))
            op_data = poll_operation(generation_id)
            save_v6_results(op_data.get("results"), out_path)
            log(f"    saved: {out_path}")
            return {
                "index": item.index, "locale": locale, "status": "success",
                "output": str(out_path), "generation_id": generation_id,
                "operation": operation, "attempts": attempt,
            }
        except (KeyboardInterrupt, FatalApiError):
            raise
        except Exception as e:
            if GEN_FAIL_MAX_RETRIES > 0 and attempt >= GEN_FAIL_MAX_RETRIES:
                log(f"[GIVE-UP] [{locale}] #{item.index}: {e} (после {attempt} попыток)")
                raise
            delay = min(60.0, retry_delay() * (1.5 ** (attempt - 1)))
            log(f"[ERROR] [{locale}] #{item.index}: {e}, retry {attempt} in {delay:.1f}s...")
            time.sleep(delay)


# =========================
# АВТОПОДБОР РАБОЧЕЙ МОДЕЛИ
# =========================

def probe_operation(operation: str, prompt: str, *, timeout: int = 120) -> Tuple[bool, str]:
    """Одна генерация через operation, без бесконечных ретраев. (ok, detail)."""
    url = BASE_URL.rstrip("/") + V6_GENERATIONS_ENDPOINT
    payload = build_payload(prompt, operation)
    try:
        r = requests.post(url, headers=headers(), json=payload, timeout=REQUEST_TIMEOUT)
    except Exception as e:
        return False, f"POST error: {e}"
    if r.status_code != 200:
        return False, f"HTTP {r.status_code}: {r.text[:160]}"
    try:
        gid = r.json().get("id")
    except Exception:
        gid = None
    if not gid:
        return False, f"нет id: {r.text[:160]}"
    status_url = BASE_URL.rstrip("/") + V6_GENERATION_STATUS_ENDPOINT.format(generation_id=gid)
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
            return False, f"status HTTP {s.status_code}: {s.text[:160]}"
        body = s.json()
        st = body.get("status")
        if st == "succeeded":
            return True, f"ok, results={len(body.get('results') or [])}"
        if st == "failed":
            return False, f"failed: {body.get('error') or body}"
    return False, f"timeout {timeout}s"


def autoselect_operation(sample_prompt: str) -> Optional[str]:
    log("\n" + "=" * 50)
    log("АВТОПОДБОР рабочей модели (по 1 пробной картинке)")
    log("=" * 50)
    for op in TEXT2IMAGE_OPERATIONS:
        log(f"-> проверяю {op} ...")
        ok, detail = probe_operation(op, sample_prompt)
        log(f"   {'✅ OK' if ok else '❌'} {op}: {detail}")
        if ok:
            log(f"\n[ВЫБРАНА МОДЕЛЬ] {op}")
            log("=" * 50)
            return op
    log("\n❌ Ни одна text-to-image модель не сработала — проблема на стороне аккаунта/сервиса.")
    log("=" * 50)
    return None


# =========================
# GLOBAL SCHEDULING
# =========================

def build_jobs(jobs_by_lang: List[Tuple[str, List[PromptItem]]]) -> List[Job]:
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
                        global_log.append({"index": item.index, "locale": locale,
                                           "status": "fatal_error", "error": str(e)})
                        save_json(VISUAL_ROOT / "generation_log.json", global_log)
                    raise
                except Exception as e:
                    log(f"  [FAILED] [{locale}] #{item.index}: {e}")
                    with log_lock:
                        global_log.append({"index": item.index, "locale": locale,
                                           "status": "error", "error": str(e)})
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
    global V6_OPERATION, IMAGE_SEED, FLOW_UPSCALE_2X

    parser = argparse.ArgumentParser(
        description="Генерирует DE/PL/RU картинки ИЗ ТЕКСТА через Fast-Gen V6, без персонажа, "
                    "один общий пул потоков; при старте автоподбор рабочей модели."
    )
    parser.add_argument("--lang", nargs="+", choices=LOCALES, default=None)
    parser.add_argument("--prompts-dir", default=str(PROMPTS_DIR))
    parser.add_argument("--visual-root", default=str(VISUAL_ROOT))
    parser.add_argument("--operation", choices=TEXT2IMAGE_OPERATIONS, default=(V6_OPERATION or None),
                        help="Жёстко выбрать модель. По умолчанию — автоподбор рабочей.")
    parser.add_argument("--seed", type=int, default=(IMAGE_SEED if IMAGE_SEED is not None else -1),
                        help=f"seed (nano/openai), 0..{SEED_MAX}. -1 = случайный.")
    parser.add_argument("--quality", choices=["speed", "quality"],
                        default=os.getenv("FAST_GEN_IMAGE_QUALITY", "").strip().lower() or None,
                        help="Качество для grok.")
    parser.add_argument("--upscale", action="store_true", default=FLOW_UPSCALE_2X,
                        help="2x upscale для nano_banana_*. Удваивает кредиты.")
    parser.add_argument("--api-base", default=BASE_URL)
    parser.add_argument("--aspect-ratio", default=ASPECT_RATIO,
                        choices=["16:9", "9:16", "1:1", "4:3", "3:4"])
    parser.add_argument("--workers", type=_parse_workers_arg, default=MAX_IMAGE_WORKERS,
                        help="Потоки: 'auto' (по умолчанию), 'max', либо число.")
    parser.add_argument("--poll-sec", type=int, default=OPERATION_POLL_SEC)
    parser.add_argument("--no-skip", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    BASE_URL = args.api_base.rstrip("/")
    PROMPTS_DIR = Path(args.prompts_dir).expanduser()
    VISUAL_ROOT = Path(args.visual_root).expanduser()
    ASPECT_RATIO = args.aspect_ratio
    MAX_IMAGE_WORKERS = args.workers
    OPERATION_POLL_SEC = max(1, args.poll_sec)
    SKIP_EXISTING = not args.no_skip
    V6_OPERATION = args.operation or ""
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
        if pf is not None and pf.exists():
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

    if args.dry_run:
        log("\nDRY RUN — API не вызывается")
        log(f"Fast-Gen V6 | text-to-image, без персонажа | aspect_ratio={_resolve_aspect_ratio()}")
        for lang, items in jobs_by_lang:
            log(f"\n[{lang}] первые 3 промпта:")
            for item in items[:3]:
                log(f"  {item.index:03d}.png <- {clean_prompt_text(item.prompt)[:160]}...")
        return

    ensure_api_ready()

    sample_prompt = next((it.prompt for _, items in jobs_by_lang for it in items), "a calm minimalist illustration")
    if not V6_OPERATION:
        chosen = autoselect_operation(sample_prompt)
        if not chosen:
            print("\n❌ Рабочая text-to-image модель не найдена. Проверь баланс кредитов и статус Fast-Gen.",
                  file=sys.stderr)
            sys.exit(2)
        V6_OPERATION = chosen
    else:
        log(f"[МОДЕЛЬ] задана вручную: {V6_OPERATION}")

    log(f"Fast-Gen V6 | operation={V6_OPERATION} | text-to-image, без персонажа | aspect_ratio={_resolve_aspect_ratio()}")

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
