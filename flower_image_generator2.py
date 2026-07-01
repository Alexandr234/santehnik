#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DE/PL/RU IMAGE generator — Fast-Gen V6, БЕЗ привязки к персонажу.

Генерирует картинки ЧИСТО ИЗ ТЕКСТОВЫХ ПРОМПТОВ (text-to-image). Никакого
reference-изображения персонажа и «фиксации лица» — каждый кадр рисуется свободно
по своему промпту (эпичный живописный стиль из psych_prompt_pipeline).

ОБНОВЛЕНО ПОД V6 (media_gen_api, openapi 1.11). Старые пути /api/v4/... удалены —
поэтому раньше был HTTP 404. Теперь всё идёт через единый эндпоинт:

  • POST /api/v6/generations           (GenerationCreateRequest)
      prompt (обяз.), operation (canonical id, напр. "flower_image_generate"),
      aspect_ratio (любой n:n, дефолт "16:9"), seed (0..2147483647),
      quality ("speed"|"quality" — для grok/openai),
      generation_config.upscale.type="2x" (для flow/nano image, удваивает кредиты).
      Ответ: GenerationAcceptedResponse -> поле "id" (id генерации).
  • GET  /api/v6/generations/{id}      (GenerationStatusResponse)
      status: queued | running | succeeded | failed,
      results[]: { type, download_url, data (inline data URI), mime_type, metadata },
      error, warnings, translations.

Операции для картинок (text-to-image) и их цена:
  flower_image_generate         — flower/flower-image      — 1 кредит      (дефолт, дёшево)
  grok_image_generate           — grok/grok-image          — 1 (speed) / 3 (quality)
  openai_image_generate         — openai/openai-image      — 2 кредита
  nano_banana_2_image_generate  — flow/nano-banana-2        — 4 (base) / 8 (2x)
  nano_banana_pro_image_generate— flow/nano-banana-pro      — 4 (base) / 8 (2x)

Читает промпты из ТРЁХ файлов psych_prompt_pipeline:
  ПРОМПТЫ/prompts_de.txt · prompts_pl.txt · prompts_ru.txt
Кладёт картинки в:
  ВИЗУАЛ/DE/001.png · ВИЗУАЛ/PL/001.png · ВИЗУАЛ/RU/001.png
(имена 001.png соответствуют image_filename из image_times_{lang}.json).

Быстрый запуск:
  cd "/Users/aleksandrtomilov/Desktop/ПСИХОЛОГИЯ ГЕРМАНИЯ ПОЛЬША/ВИЗУАЛ"
  python3 flower_image_generator2.py --workers 2

Флаги:
  --workers N            параллельных задач (начни с 1-2, снизь если 429)
  --lang DE PL RU        обработать только нужные языки
  --engine flower|nano2|nano-pro|grok|openai   (какой движок/операция; дефолт flower = 1 кредит)
  --operation <id>       напрямую задать canonical V6 operation id (перекрывает --engine)
  --quality speed|quality   для grok/openai
  --seed N               фиксированный seed, -1 = случайный
  --upscale              2x-апскейл (только nano/flow, удваивает кредиты)
  --aspect-ratio 16:9    любой n:n
  --no-skip              перегенерировать уже существующие
  --dry-run              только показать промпты, API не вызывать
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import re
import sys
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
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

# Папки по умолчанию
PROMPTS_DIR  = PROJECT_ROOT / "ПРОМПТЫ"
VISUAL_ROOT  = PROJECT_ROOT / "ВИЗУАЛ"

# Имена файлов промптов — генерирует psych_prompt_pipeline
PROMPTS_FILE_BY_LANG: Dict[str, Path] = {
    "DE": PROMPTS_DIR / "prompts_de.txt",
    "PL": PROMPTS_DIR / "prompts_pl.txt",
    "RU": PROMPTS_DIR / "prompts_ru.txt",
}

LOCALES = ["DE", "PL", "RU"]

ASPECT_RATIO = os.getenv("FAST_GEN_IMAGE_ASPECT_RATIO", "16:9")
# V6 принимает любой n:n. Проверяем базово, чтобы не ловить 422.
ASPECT_RATIO_RE = re.compile(r"^[1-9]\d*:[1-9]\d*$")

# Дружелюбные имена движков -> canonical V6 operation id для text-to-image.
ENGINE_TO_OPERATION: Dict[str, str] = {
    "flower":   "flower_image_generate",           # 1 кредит (дёшево, дефолт)
    "grok":     "grok_image_generate",             # 1/3 кредита
    "openai":   "openai_image_generate",           # 2 кредита
    "nano2":    "nano_banana_2_image_generate",    # 4/8 кредитов
    "nano-pro": "nano_banana_pro_image_generate",  # 4/8 кредитов
}
# Операции, которые поддерживают 2x-апскейл (flow image модели).
UPSCALE_CAPABLE = {"nano_banana_2_image_generate", "nano_banana_pro_image_generate"}

# Движок/операция по умолчанию — самый дешёвый text-to-image.
IMAGE_ENGINE   = os.getenv("FAST_GEN_IMAGE_ENGINE", "flower").strip().lower() or "flower"
IMAGE_OPERATION = os.getenv("FAST_GEN_IMAGE_OPERATION", "").strip()  # если задано — перекрывает engine
IMAGE_QUALITY  = os.getenv("FAST_GEN_IMAGE_QUALITY", "").strip()      # speed|quality (grok/openai)

# Допустимый диапазон seed: 0 .. 2147483647.
SEED_MAX = 2147483647

# Seed нужен только для ВОСПРОИЗВОДИМОСТИ конкретного кадра. По умолчанию не фиксируем.
_SEED_RAW = os.getenv("FAST_GEN_SEED", "").strip()
IMAGE_SEED: Optional[int] = (
    min(int(_SEED_RAW), SEED_MAX)
    if _SEED_RAW.lstrip("-").isdigit() and int(_SEED_RAW) >= 0
    else None
)

# 2x-апскейл (generation_config.upscale.type="2x"). Удваивает кредиты и rate-limit.
IMAGE_UPSCALE_2X = os.getenv("FAST_GEN_UPSCALE_2X", "").strip().lower() in {"1", "true", "yes", "on"}

REQUEST_TIMEOUT       = int(os.getenv("FAST_GEN_REQUEST_TIMEOUT", "300"))
OPERATION_POLL_SEC    = int(os.getenv("FAST_GEN_OPERATION_POLL_SEC", "6"))
OPERATION_TIMEOUT_SEC = int(os.getenv("FAST_GEN_OPERATION_TIMEOUT_SEC", "1800"))
RETRY_DELAY_SEC       = int(os.getenv("FAST_GEN_RETRY_DELAY_SEC", "8"))
MAX_RETRIES           = int(os.getenv("FAST_GEN_MAX_RETRIES", "0"))   # 0 = бесконечно
MAX_IMAGE_WORKERS     = int(os.getenv("FAST_GEN_IMAGE_WORKERS", "20"))

SKIP_EXISTING = True

# V6 эндпоинты.
V6_GENERATIONS_ENDPOINT = "/api/v6/generations"
V6_GENERATION_STATUS    = "/api/v6/generations/{generation_id}"


# =========================
# DATA
# =========================

@dataclass(frozen=True)
class PromptItem:
    index: int
    prompt: str


class FatalApiError(RuntimeError):
    pass


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
        print("[ERROR] Не найден API ключ Fast-Gen.")
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
    # Формат 001.png соответствует image_filename из image_times_{lang}.json
    return VISUAL_ROOT / locale / f"{item.index:03d}.png"


def file_ok(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size > 0


def resolve_existing_path(path: Path) -> Path:
    """Ищет файл с учётом разных Unicode-нормализаций (NFD/NFC — частая проблема macOS)."""
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


# Формат psych_prompt_pipeline:
# 001 | 00:00:00,000 --> 00:00:02,350 | 2.35s | scene_type
# TEXT: spoken text here
# image prompt here
#
# 002 | ...
PSYCH_HEADER_RE = re.compile(
    r"^\s*(\d{1,5})\s*\|\s*\d{2}:\d{2}:\d{2}[,\.]?\d*\s*-->"
)


def parse_psych_pipeline_format(lines: List[str]) -> List[PromptItem]:
    """Парсит формат psych_prompt_pipeline: берёт строку(и) image_prompt после TEXT:."""
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


# Запасной: нумерованные строки вида "1. prompt" / "1) prompt"
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
    """Ищет файл промптов для языка. Проверяет несколько возможных расположений."""
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
                "operation": current_operation(),
                "prompt": item.prompt,
            })


# =========================
# HTTP HELPERS
# =========================

def should_retry(attempt: int) -> bool:
    return MAX_RETRIES <= 0 or attempt < MAX_RETRIES


def post_json(endpoint: str, payload: dict, *, label: str) -> dict:
    url = BASE_URL.rstrip("/") + endpoint
    attempt = 0
    while True:
        attempt += 1
        try:
            resp = requests.post(url, headers=headers(), json=payload, timeout=REQUEST_TIMEOUT)
            if resp.status_code in {400, 401, 403, 404, 422}:
                raise FatalApiError(f"{label}: HTTP {resp.status_code}: {resp.text}")
            if resp.status_code >= 400:
                raise RuntimeError(f"{label}: HTTP {resp.status_code}: {resp.text}")
            return resp.json()
        except (KeyboardInterrupt, FatalApiError):
            raise
        except Exception as e:
            if not should_retry(attempt):
                raise
            log(f"[WARN] {label}: attempt {attempt} failed: {e}, retry in {RETRY_DELAY_SEC}s...")
            time.sleep(RETRY_DELAY_SEC)


def get_json(endpoint: str, *, label: str, params: Optional[dict] = None) -> dict:
    url = BASE_URL.rstrip("/") + endpoint
    attempt = 0
    while True:
        attempt += 1
        try:
            resp = requests.get(url, headers=headers(json_content=False), params=params, timeout=REQUEST_TIMEOUT)
            if resp.status_code in {400, 401, 403, 404, 422}:
                raise FatalApiError(f"{label}: HTTP {resp.status_code}: {resp.text}")
            if resp.status_code >= 400:
                raise RuntimeError(f"{label}: HTTP {resp.status_code}: {resp.text}")
            return resp.json()
        except (KeyboardInterrupt, FatalApiError):
            raise
        except Exception as e:
            if not should_retry(attempt):
                raise
            log(f"[WARN] {label}: attempt {attempt} failed: {e}, retry in {RETRY_DELAY_SEC}s...")
            time.sleep(RETRY_DELAY_SEC)


def download_file(url: str, path: Path, *, label: str, use_api_key: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    attempt = 0
    while True:
        attempt += 1
        try:
            h = {"X-API-Key": API_KEY} if use_api_key else {}
            resp = requests.get(url, headers=h, timeout=REQUEST_TIMEOUT)
            if resp.status_code in {400, 401, 403, 404}:
                raise FatalApiError(f"{label}: HTTP {resp.status_code}")
            if resp.status_code >= 400:
                raise RuntimeError(f"{label}: HTTP {resp.status_code}")
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
# RESULT DECODING (V6: download_url | inline data)
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


def save_generation_results(results: Any, out_path: Path) -> None:
    """
    Сохраняет первый image-результат из GenerationStatusResponse.results.
    V6 отдаёт либо download_url (готовый URL файла), либо inline data (data URI).
    """
    if not isinstance(results, list) or not results:
        raise RuntimeError(f"Пустой results: {pretty_json(results)}")

    # Берём первый элемент типа image (или просто первый).
    item = next((r for r in results if isinstance(r, dict) and r.get("type") == "image"), None)
    if item is None:
        item = results[0] if isinstance(results[0], dict) else {}

    data = item.get("data")
    url = item.get("download_url")

    if isinstance(data, str) and data.startswith("data:"):
        save_data_uri(data, out_path)
        return
    if isinstance(url, str) and url:
        # download_url обычно самодостаточен; если вернёт 401/403 — пробуем с API-ключом.
        try:
            download_file(url, out_path, label="RESULT URL")
        except FatalApiError:
            download_file(url, out_path, label="RESULT URL (auth)", use_api_key=True)
        return
    raise RuntimeError(f"В результате нет ни download_url, ни data: {pretty_json(item)}")


# =========================
# V6 IMAGE GENERATION (text-to-image, без персонажа)
# =========================

def current_operation() -> str:
    """Возвращает canonical V6 operation id: явный --operation или маппинг из --engine."""
    if IMAGE_OPERATION:
        return IMAGE_OPERATION
    return ENGINE_TO_OPERATION.get(IMAGE_ENGINE, "flower_image_generate")


def resolve_aspect_ratio() -> str:
    if ASPECT_RATIO_RE.match(ASPECT_RATIO):
        return ASPECT_RATIO
    log(f"[WARN] aspect_ratio {ASPECT_RATIO!r} не в формате n:n — использую 16:9.")
    return "16:9"


def build_payload(prompt: str) -> Dict[str, Any]:
    """Тело запроса GenerationCreateRequest для text-to-image (без inputs/референсов)."""
    op = current_operation()
    payload: Dict[str, Any] = {
        "prompt": clean_prompt_text(prompt),
        "operation": op,
        "aspect_ratio": resolve_aspect_ratio(),
    }
    if IMAGE_SEED is not None:
        payload["seed"] = min(max(0, IMAGE_SEED), SEED_MAX)
    if IMAGE_QUALITY:
        payload["quality"] = IMAGE_QUALITY
    if IMAGE_UPSCALE_2X and op in UPSCALE_CAPABLE:
        payload["generation_config"] = {"upscale": {"type": "2x"}}
    return payload


def submit_generate(prompt: str) -> str:
    """POST /api/v6/generations -> возвращает id генерации."""
    data = post_json(V6_GENERATIONS_ENDPOINT, build_payload(prompt), label="V6 GENERATION CREATE")
    gen_id = data.get("id")
    if not gen_id:
        raise RuntimeError(f"API не вернул id генерации: {pretty_json(data)}")
    return str(gen_id)


def poll_generation(gen_id: str) -> dict:
    """GET /api/v6/generations/{id} до succeeded/failed."""
    endpoint = V6_GENERATION_STATUS.format(generation_id=gen_id)
    started = time.time()
    while True:
        data = get_json(endpoint, label=f"V6 GEN {gen_id}")
        state = data.get("status")
        if state in {"queued", "running"}:
            log(f"    gen {gen_id}: {state}")
            if time.time() - started > OPERATION_TIMEOUT_SEC:
                raise RuntimeError(f"Timeout {gen_id} after {OPERATION_TIMEOUT_SEC}s")
            time.sleep(OPERATION_POLL_SEC)
            continue
        if state == "succeeded":
            if not data.get("results"):
                raise RuntimeError(f"gen {gen_id}: succeeded, но results пустой")
            for w in (data.get("warnings") or []):
                log(f"    [WARN] gen {gen_id}: {w}")
            return data
        # failed / неизвестно: подтягиваем перевод ошибки, если есть.
        err = data.get("error") or pretty_json(data)
        translations = data.get("translations") or {}
        if isinstance(translations, dict) and translations:
            tr = translations.get("ru") or next(iter(translations.values()), None)
            if tr:
                err = f"{err} | {tr}"
        raise RuntimeError(f"gen {gen_id}: {state}: {err}")


def generate_image(item: PromptItem, out_path: Path, locale: str) -> dict:
    attempt = 0
    while True:
        attempt += 1
        gen_id = None
        try:
            log(f"[{locale}] {out_path.name} (#{item.index}) attempt {attempt}")
            gen_id = submit_generate(item.prompt)
            log(f"    gen_id: {gen_id}")
            gen_data = poll_generation(gen_id)
            save_generation_results(gen_data.get("results"), out_path)
            log(f"    saved: {out_path}")
            return {"index": item.index, "locale": locale, "status": "success",
                    "output": str(out_path), "gen_id": gen_id, "attempts": attempt}
        except (KeyboardInterrupt, FatalApiError):
            raise
        except Exception as e:
            if not should_retry(attempt):
                raise
            log(f"[ERROR] [{locale}] #{item.index}: {e}, retry in {RETRY_DELAY_SEC}s...")
            time.sleep(RETRY_DELAY_SEC)


# =========================
# GENERATION LOGIC
# =========================

def process_locale(
    locale: str,
    items: List[PromptItem],
    global_log: List[dict],
    log_lock: Lock,
    workers: int,
) -> None:
    """Генерирует все картинки для одного языка параллельно."""
    log(f"\n{'=' * 42}")
    log(f"  {locale}: {len(items)} промптов | workers={workers}")
    log(f"  Промпты: {find_prompts_file(locale)}")
    log(f"  Выход:   {VISUAL_ROOT / locale}/")
    log(f"{'=' * 42}")

    pending = []
    for item in items:
        p = output_path(locale, item)
        if SKIP_EXISTING and file_ok(p):
            log(f"  [SKIP] {locale}/{p.name}")
            continue
        pending.append(item)

    if not pending:
        log(f"  [OK] {locale}: все картинки уже существуют")
        return

    log(f"  {locale}: к генерации {len(pending)} из {len(items)}")

    actual_workers = max(1, min(workers, len(pending)))
    with ThreadPoolExecutor(max_workers=actual_workers) as ex:
        futures = {ex.submit(generate_image, item, output_path(locale, item), locale): item
                   for item in pending}
        for future in as_completed(futures):
            try:
                result = future.result()
                result["prompt"] = futures[future].prompt
                with log_lock:
                    global_log.append(result)
                    save_json(VISUAL_ROOT / "generation_log.json", global_log)
            except Exception as e:
                item = futures[future]
                log(f"  [FAILED] [{locale}] #{item.index}: {e}")
                with log_lock:
                    global_log.append({"index": item.index, "locale": locale,
                                       "status": "error", "error": str(e)})
                    save_json(VISUAL_ROOT / "generation_log.json", global_log)


# =========================
# MAIN
# =========================

def main() -> None:
    global BASE_URL, SKIP_EXISTING, ASPECT_RATIO, MAX_IMAGE_WORKERS
    global OPERATION_POLL_SEC, PROMPTS_DIR, VISUAL_ROOT
    global IMAGE_ENGINE, IMAGE_OPERATION, IMAGE_QUALITY, IMAGE_SEED, IMAGE_UPSCALE_2X

    parser = argparse.ArgumentParser(
        description="Генерирует DE/PL/RU картинки из текстовых промптов через Fast-Gen V6, без персонажа"
    )
    parser.add_argument("--lang", nargs="+", choices=LOCALES, default=None,
                        help="Языки для обработки. По умолчанию — все три.")
    parser.add_argument("--prompts-dir", default=str(PROMPTS_DIR),
                        help="Папка с файлами prompts_de.txt / prompts_pl.txt / prompts_ru.txt")
    parser.add_argument("--visual-root", default=str(VISUAL_ROOT), help="Папка ВИЗУАЛ")
    parser.add_argument("--engine", choices=list(ENGINE_TO_OPERATION.keys()), default=IMAGE_ENGINE,
                        help="Движок/операция: flower=1кр (дефолт), grok=1/3кр, openai=2кр, nano2/nano-pro=4/8кр.")
    parser.add_argument("--operation", default=IMAGE_OPERATION or None,
                        help="Напрямую canonical V6 operation id (перекрывает --engine), напр. flower_image_generate.")
    parser.add_argument("--quality", default=(IMAGE_QUALITY or None), choices=["speed", "quality"],
                        help="Режим качества для grok/openai.")
    parser.add_argument("--seed", type=int, default=(IMAGE_SEED if IMAGE_SEED is not None else -1),
                        help=f"Фиксированный seed для воспроизводимости кадра, 0..{SEED_MAX}. -1 = случайный.")
    parser.add_argument("--upscale", action="store_true", default=IMAGE_UPSCALE_2X,
                        help="2x-апскейл (только nano2/nano-pro). Удваивает кредиты.")
    parser.add_argument("--api-base", default=BASE_URL)
    parser.add_argument("--aspect-ratio", default=ASPECT_RATIO,
                        help="Соотношение сторон в формате n:n (напр. 16:9, 9:16, 1:1, 4:3).")
    parser.add_argument("--workers", type=int, default=MAX_IMAGE_WORKERS,
                        help="Параллельных задач на язык (начни с 1-2, снизь при 429)")
    parser.add_argument("--poll-sec", type=int, default=OPERATION_POLL_SEC)
    parser.add_argument("--no-skip", action="store_true", help="Перегенерировать уже существующие")
    parser.add_argument("--dry-run", action="store_true", help="Показать промпты без API-вызовов")
    args = parser.parse_args()

    BASE_URL             = args.api_base.rstrip("/")
    PROMPTS_DIR          = Path(args.prompts_dir).expanduser()
    VISUAL_ROOT          = Path(args.visual_root).expanduser()
    ASPECT_RATIO         = args.aspect_ratio
    MAX_IMAGE_WORKERS    = max(1, args.workers)
    OPERATION_POLL_SEC   = max(1, args.poll_sec)
    SKIP_EXISTING        = not args.no_skip
    IMAGE_ENGINE         = args.engine
    IMAGE_OPERATION      = (args.operation or "").strip()
    IMAGE_QUALITY        = (args.quality or "").strip()
    IMAGE_SEED           = min(args.seed, SEED_MAX) if args.seed is not None and args.seed >= 0 else None
    IMAGE_UPSCALE_2X     = bool(args.upscale)

    langs_to_process = args.lang or LOCALES

    ensure_dirs()

    print("\n=== План работ ===")
    jobs: List[Tuple[str, List[PromptItem]]] = []
    for lang in langs_to_process:
        pf = find_prompts_file(lang)
        ok = pf is not None and pf.exists()
        if ok:
            try:
                items = load_prompts(pf)
                jobs.append((lang, items))
                print(f"  {lang}: {len(items)} промптов из {pf.name} -> {VISUAL_ROOT / lang}/")
            except Exception as e:
                print(f"  {lang}: ❌ ошибка загрузки промптов: {e}")
        else:
            print(f"  {lang}: ❌ файл промптов не найден в {PROMPTS_DIR}")
    print("==================\n")

    if not jobs:
        print("❌ Не найдено ни одного файла промптов.", file=sys.stderr)
        print(f"Ожидаемые файлы: prompts_de.txt / prompts_pl.txt / prompts_ru.txt в {PROMPTS_DIR}")
        sys.exit(1)

    op = current_operation()
    seed_txt = IMAGE_SEED if IMAGE_SEED is not None else "random"
    extra = f", seed={seed_txt}"
    if IMAGE_QUALITY:
        extra += f", quality={IMAGE_QUALITY}"
    if IMAGE_UPSCALE_2X and op in UPSCALE_CAPABLE:
        extra += ", upscale=2x"

    if args.dry_run:
        log("\nDRY RUN — API не вызывается")
        log(f"Operation: {op} | aspect_ratio={resolve_aspect_ratio()}{extra}")
        for lang, items in jobs:
            log(f"\n[{lang}] первые 3 промпта:")
            for item in items[:3]:
                log(f"  {item.index:03d}.png <- {clean_prompt_text(item.prompt)[:160]}...")
        return

    ensure_api_ready()
    log(f"Operation: {op} | text-to-image, без персонажа | aspect_ratio={resolve_aspect_ratio()}{extra}")

    global_log = load_json(VISUAL_ROOT / "generation_log.json", [])
    if not isinstance(global_log, list):
        global_log = []
    log_lock = Lock()

    for lang, items in jobs:
        write_manifest(lang, items)
        process_locale(lang, items, global_log, log_lock, MAX_IMAGE_WORKERS)

    print(f"\n{'=' * 42}")
    print("ALL DONE")
    for lang, _ in jobs:
        count = len(list((VISUAL_ROOT / lang).glob("*.png")))
        print(f"  {lang}: {count} картинок в {VISUAL_ROOT / lang}/")
    print(f"{'=' * 42}")


if __name__ == "__main__":
    main()
