#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DE/PL/RU IMAGE generator — Fast-Gen V4 (Flower / Flow), БЕЗ привязки к персонажу.

Эта версия генерирует картинки ЧИСТО ИЗ ТЕКСТОВЫХ ПРОМПТОВ (text-to-image).
Никакого reference-изображения персонажа и никакой «фиксации лица» больше нет —
каждый кадр рисуется свободно по своему промпту (эпичный живописный стиль,
анонимные силуэты / чистые пейзажи из psych_prompt_pipeline).

Эндпоинты (media_gen_api, openapi 3.1.0):
  • /api/v4/flower/image/generate  (FlowerGenerateImageRequest)
      prompt, aspect_ratio (16:9 | 9:16 | 1:1). Без reference_image это обычная
      text-to-image генерация — 1 кредит, aspect_ratio соблюдается.
  • /api/v4/flow/image/generate    (FlowGenerateImageRequest)
      prompt, aspect_ratio (16:9 | 4:3 | 1:1 | 3:4 | 9:16),
      model (GEM_PIX_2 | IMAGEN_3_5 | NARWHAL), seed (0..2147483647),
      generation_config.upscale.type="2x" (апскейл x2, удваивает кредиты). 4 кредита/картинка.
  • Опрос:    /api/v4/operations/{operation_id}?result_format=ref|data_uri
      OperationStatusResponse: status, result (СПИСОК строк), warnings, error, translations.

Читает промпты из ТРЁХ отдельных файлов (генерирует psych_prompt_pipeline):
  ПРОМПТЫ/prompts_de.txt
  ПРОМПТЫ/prompts_pl.txt
  ПРОМПТЫ/prompts_ru.txt

Для каждого языка генерирует СВОИ картинки:
  ВИЗУАЛ/DE/001.png, 002.png ...
  ВИЗУАЛ/PL/001.png, 002.png ...
  ВИЗУАЛ/RU/001.png, 002.png ...

Имена файлов (001.png) соответствуют полю image_filename из image_times_{lang}.json.

Быстрый запуск:
  cd "/Users/aleksandrtomilov/Desktop/ПСИХОЛОГИЯ ГЕРМАНИЯ ПОЛЬША/ВИЗУАЛ"
  python3 flower_image_generator2.py --workers 2

Флаги:
  --workers N        параллельных задач (начни с 1-2, снизь если 429)
  --lang DE PL RU    обработать только нужные языки
  --engine flow|flower
  --flow-model NARWHAL|GEM_PIX_2|IMAGEN_3_5
  --seed N           фиксированный seed (flow), -1 = случайный
  --upscale          flow: вернуть 2x-апскейл (удваивает кредиты)
  --no-skip          перегенерировать уже существующие
  --dry-run          только показать промпты, API не вызывать
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import re
import shutil
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

# Допустимые aspect_ratio по эндпоинтам (из новой OpenAPI):
FLOW_ASPECT_RATIOS   = {"16:9", "4:3", "1:1", "3:4", "9:16"}   # FlowGenerateImageRequest
FLOWER_ASPECT_RATIOS = {"16:9", "9:16", "1:1"}                 # FlowerGenerateImageRequest

# Движок генерации (обе опции теперь работают как чистый text-to-image, без референса):
#   "flower" -> /api/v4/flower/image/generate — text-to-image, 1 кредит/картинка.
#   "flow"   -> /api/v4/flow/image/generate   — text-to-image, модель на выбор,
#               seed для воспроизводимости, опциональный апскейл x2. 4 кредита/картинка.
IMAGE_ENGINE = os.getenv("FAST_GEN_IMAGE_ENGINE", "flower").strip().lower() or "flower"

# Модель для flow: NARWHAL (Nano Banana 2), GEM_PIX_2 (Nano Pro), IMAGEN_3_5 (Imagen 4)
FLOW_MODEL = os.getenv("FAST_GEN_FLOW_MODEL", "NARWHAL").strip() or "NARWHAL"

# Допустимый диапазон seed по новой схеме: 0 .. 2147483647.
SEED_MAX = 2147483647

# Seed нужен только для ВОСПРОИЗВОДИМОСТИ конкретного кадра (перегенерировать один и тот же
# результат). Персонажа больше нет, поэтому по умолчанию seed НЕ фиксируем — каждый кадр
# получает свободную композицию. Пусто/<0 = случайный seed.
_SEED_RAW = os.getenv("FAST_GEN_SEED", "").strip()
REFERENCE_SEED: Optional[int] = (
    min(int(_SEED_RAW), SEED_MAX)
    if _SEED_RAW.lstrip("-").isdigit() and int(_SEED_RAW) >= 0
    else None
)

# flow: апскейл x2 (generation_config.upscale.type="2x"). Удваивает кредиты и rate-limit.
FLOW_UPSCALE_2X = os.getenv("FAST_GEN_FLOW_UPSCALE_2X", "").strip().lower() in {"1", "true", "yes", "on"}

REQUEST_TIMEOUT       = int(os.getenv("FAST_GEN_REQUEST_TIMEOUT", "300"))
OPERATION_POLL_SEC    = int(os.getenv("FAST_GEN_OPERATION_POLL_SEC", "6"))
OPERATION_TIMEOUT_SEC = int(os.getenv("FAST_GEN_OPERATION_TIMEOUT_SEC", "1800"))
RETRY_DELAY_SEC       = int(os.getenv("FAST_GEN_RETRY_DELAY_SEC", "8"))
MAX_RETRIES           = int(os.getenv("FAST_GEN_MAX_RETRIES", "0"))   # 0 = бесконечно
MAX_IMAGE_WORKERS     = int(os.getenv("FAST_GEN_IMAGE_WORKERS", "20"))

SKIP_EXISTING = True

FLOWER_IMAGE_ENDPOINT = "/api/v4/flower/image/generate"
FLOW_IMAGE_ENDPOINT   = "/api/v4/flow/image/generate"
V4_OPERATION_ENDPOINT = "/api/v4/operations/{operation_id}"

# Хранилище нужно только для СКАЧИВАНИЯ результата, когда API возвращает file:<hash>.
STORAGE_GET_URL = "https://storage.fast-gen.ai/file/{file_hash}"


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
    """
    Парсит формат psych_prompt_pipeline:
      001 | 00:00:00,000 --> 00:00:02,350 | 2.35s | scene_type
      TEXT: озвучка
      image prompt текст
      (пустая строка)
    Берёт строку(и) image_prompt — всё после строки TEXT:.
    """
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
        # Пропускаем строку TEXT:
        while i < len(lines) and lines[i].strip().upper().startswith("TEXT:"):
            i += 1
        # Собираем строки промпта до следующего заголовка или пустой строки
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

    # Пробуем форматы по убыванию специфичности
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
    # Явно заданный путь
    explicit = PROMPTS_FILE_BY_LANG.get(lang.upper())
    if explicit:
        candidates.insert(0, explicit)

    for p in candidates:
        rp = resolve_existing_path(p)
        if rp.exists() and rp.stat().st_size > 0:
            return rp

    # Рекурсивный поиск
    for p in PROMPTS_DIR.rglob(f"*prompts*{ll}*.txt"):
        if p.stat().st_size > 0:
            return p
    return None


def write_manifest(locale: str, items: Sequence[PromptItem]) -> None:
    locale_dir = VISUAL_ROOT / locale
    locale_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = locale_dir / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["index", "filename", "prompt"])
        writer.writeheader()
        for item in items:
            writer.writerow({
                "index": item.index,
                "filename": f"{item.index:03d}.png",
                "prompt": item.prompt,
            })


# =========================
# HTTP HELPERS
# =========================

def _check_api_json(data: dict, *, label: str) -> dict:
    if not isinstance(data, dict):
        raise RuntimeError(f"{label}: API вернул не JSON-объект")
    # OperationResponse (submit) содержит success=true. OperationStatusResponse (poll)
    # поля success не имеет — поэтому проверяем только явный success=false.
    if data.get("success") is False:
        raise RuntimeError(f"{label}: API success=false: {data.get('error') or pretty_json(data)}")
    return data


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
            return _check_api_json(resp.json(), label=label)
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
            return _check_api_json(resp.json(), label=label)
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
            h = headers(json_content=False) if use_api_key else {}
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
# RESULT DECODING (data URI / file ref / url)
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


def resolve_file_ref(file_ref: str) -> str:
    """Скачивает результат по file:<hash> из хранилища и возвращает его как data URI."""
    fh = file_ref.replace("file:", "", 1)
    url = STORAGE_GET_URL.format(file_hash=fh)
    attempt = 0
    while True:
        attempt += 1
        try:
            resp = requests.get(url, headers={"X-API-Key": API_KEY}, timeout=REQUEST_TIMEOUT)
            if resp.status_code in {400, 401, 403, 404}:
                raise FatalApiError(f"STORAGE GET: HTTP {resp.status_code}")
            if resp.status_code >= 400:
                raise RuntimeError(f"STORAGE GET: HTTP {resp.status_code}")
            ct = (resp.headers.get("content-type") or "").lower()
            text = resp.text.strip()
            if text.startswith("data:"):
                return text
            if "application/json" in ct or text.startswith("{"):
                data = resp.json()
                for key in ["data_uri", "result", "file", "content"]:
                    v = data.get(key)
                    if isinstance(v, str) and v.startswith("data:"):
                        return v
            return "data:image/png;base64," + base64.b64encode(resp.content).decode()
        except (KeyboardInterrupt, FatalApiError):
            raise
        except Exception as e:
            if not should_retry(attempt):
                raise
            log(f"[WARN] storage get attempt {attempt} failed: {e}, retry in {RETRY_DELAY_SEC}s...")
            time.sleep(RETRY_DELAY_SEC)


def save_operation_result(result: Any, out_path: Path) -> None:
    # OperationStatusResponse.result — это СПИСОК строк (media: data URI или file:ref).
    if isinstance(result, list):
        result = result[0] if result else None
    if not isinstance(result, str):
        raise RuntimeError(f"Неожиданный result: {pretty_json(result)}")
    if result.startswith("data:"):
        save_data_uri(result, out_path)
    elif result.startswith("file:"):
        save_data_uri(resolve_file_ref(result), out_path)
    elif result.startswith("http"):
        download_file(result, out_path, label="RESULT URL")
    else:
        raise RuntimeError(f"Неизвестный формат result: {result[:200]}")


# =========================
# FLOWER / FLOW IMAGE API (text-to-image, без персонажа)
# =========================

def build_api_prompt(scene_prompt: str, locale: str) -> str:
    """
    Промпт для чистой text-to-image генерации. Персонажа нет, поэтому промпт из
    pipeline (эпичная живописная сцена + негативы) уходит в модель практически как есть.
    Никаких инструкций «сохрани лицо/того же человека» больше не добавляем.
    """
    return clean_prompt_text(scene_prompt)


def _resolve_aspect_ratio(engine: str) -> str:
    """Подбирает корректный aspect_ratio под конкретный эндпоинт новой схемы."""
    allowed = FLOW_ASPECT_RATIOS if engine == "flow" else FLOWER_ASPECT_RATIOS
    if ASPECT_RATIO in allowed:
        return ASPECT_RATIO
    fallback = "16:9" if "16:9" in allowed else sorted(allowed)[0]
    log(f"[WARN] aspect_ratio {ASPECT_RATIO!r} недопустим для {engine} — использую {fallback!r}.")
    return fallback


def submit_generate(prompt: str, locale: str) -> Tuple[str, str]:
    api_prompt = build_api_prompt(prompt, locale)

    if IMAGE_ENGINE == "flow":
        # flow text-to-image: модель на выбор, соблюдает aspect_ratio, seed для
        # воспроизводимости конкретного кадра, опциональный апскейл x2.
        payload: Dict[str, Any] = {
            "prompt": api_prompt,
            "aspect_ratio": _resolve_aspect_ratio("flow"),
            "model": FLOW_MODEL,
        }
        if REFERENCE_SEED is not None:
            payload["seed"] = min(max(0, REFERENCE_SEED), SEED_MAX)
        if FLOW_UPSCALE_2X:
            # generation_config.upscale.type="2x" — опция апскейла x2.
            payload["generation_config"] = {"upscale": {"type": "2x"}}
        data = post_json(FLOW_IMAGE_ENDPOINT, payload, label="FLOW IMAGE GENERATE")
    else:
        # flower text-to-image: 1 кредит, соблюдает aspect_ratio (референса нет).
        payload = {
            "prompt": api_prompt,
            "aspect_ratio": _resolve_aspect_ratio("flower"),
        }
        data = post_json(FLOWER_IMAGE_ENDPOINT, payload, label="FLOWER IMAGE GENERATE")

    op_id = data.get("operation_id")
    if not op_id:
        raise RuntimeError(f"API не вернул operation_id: {pretty_json(data)}")
    op_type = str(data.get("operation_type") or "")
    return str(op_id), op_type


def poll_operation(op_id: str) -> dict:
    endpoint = V4_OPERATION_ENDPOINT.format(operation_id=op_id)
    started = time.time()
    while True:
        # result_format=ref — получаем file:ref (стримим из хранилища), экономит трафик.
        data = get_json(endpoint, label=f"V4 OP {op_id}", params={"result_format": "ref"})
        state = data.get("status")
        if state in {"pending", "processing"}:
            log(f"    op {op_id}: {state}")
            if time.time() - started > OPERATION_TIMEOUT_SEC:
                raise RuntimeError(f"Timeout {op_id} after {OPERATION_TIMEOUT_SEC}s")
            time.sleep(OPERATION_POLL_SEC)
            continue
        if state == "success":
            if not data.get("result"):
                raise RuntimeError(f"op {op_id}: success но result пустой")
            for w in (data.get("warnings") or []):
                log(f"    [WARN] op {op_id}: {w}")
            return data
        # state == "error" (или неизвестно): подтягиваем перевод ошибки, если есть.
        err = data.get("error") or pretty_json(data)
        translations = data.get("translations") or {}
        if isinstance(translations, dict) and translations:
            tr = translations.get("ru") or next(iter(translations.values()), None)
            if tr:
                err = f"{err} | {tr}"
        raise RuntimeError(f"op {op_id}: {state}: {err}")


def generate_image(item: PromptItem, out_path: Path, locale: str) -> dict:
    attempt = 0
    while True:
        attempt += 1
        op_id = None
        try:
            log(f"[{locale}] {out_path.name} (#{item.index}) attempt {attempt}")
            op_id, op_type = submit_generate(item.prompt, locale)
            log(f"    op_id: {op_id}" + (f" ({op_type})" if op_type else ""))
            op_data = poll_operation(op_id)
            save_operation_result(op_data.get("result"), out_path)
            log(f"    saved: {out_path}")
            return {"index": item.index, "locale": locale, "status": "success",
                    "output": str(out_path), "op_id": op_id, "op_type": op_type,
                    "attempts": attempt}
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
    global IMAGE_ENGINE, FLOW_MODEL, REFERENCE_SEED, FLOW_UPSCALE_2X

    parser = argparse.ArgumentParser(
        description="Генерирует DE/PL/RU картинки из текстовых промптов через Fast-Gen (flow/flower), без персонажа"
    )
    parser.add_argument("--lang", nargs="+", choices=LOCALES, default=None,
                        help="Языки для обработки. По умолчанию — все три.")
    parser.add_argument("--prompts-dir", default=str(PROMPTS_DIR),
                        help="Папка с файлами prompts_de.txt / prompts_pl.txt / prompts_ru.txt")
    parser.add_argument("--visual-root", default=str(VISUAL_ROOT), help="Папка ВИЗУАЛ")
    parser.add_argument("--engine", choices=["flow", "flower"], default=IMAGE_ENGINE,
                        help="flower = text-to-image (1 кредит). flow = text-to-image с выбором модели/seed/апскейла (4 кредита).")
    parser.add_argument("--flow-model", default=FLOW_MODEL,
                        choices=["NARWHAL", "GEM_PIX_2", "IMAGEN_3_5"],
                        help="Модель для flow: NARWHAL=Nano Banana 2 (реком.), GEM_PIX_2=Nano Pro, IMAGEN_3_5=Imagen 4")
    parser.add_argument("--seed", type=int, default=(REFERENCE_SEED if REFERENCE_SEED is not None else -1),
                        help=f"Фиксированный seed для воспроизводимости кадра (flow), 0..{SEED_MAX}. -1 = случайный.")
    parser.add_argument("--upscale", action="store_true", default=FLOW_UPSCALE_2X,
                        help="flow: вернуть 2x-апскейл (generation_config.upscale.type=2x). Удваивает кредиты.")
    parser.add_argument("--api-base", default=BASE_URL)
    parser.add_argument("--aspect-ratio", default=ASPECT_RATIO,
                        choices=["16:9", "9:16", "1:1", "4:3", "3:4"],
                        help="flow: 16:9/4:3/1:1/3:4/9:16; flower: 16:9/9:16/1:1.")
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
    FLOW_MODEL           = args.flow_model
    REFERENCE_SEED       = min(args.seed, SEED_MAX) if args.seed is not None and args.seed >= 0 else None
    FLOW_UPSCALE_2X      = bool(args.upscale)

    langs_to_process = args.lang or LOCALES

    ensure_dirs()

    # Показываем план
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

    if args.dry_run:
        log("\nDRY RUN — API не вызывается")
        for lang, items in jobs:
            log(f"\n[{lang}] первые 3 промпта:")
            for item in items[:3]:
                preview = build_api_prompt(item.prompt, lang)
                log(f"  {item.index:03d}.png <- {preview[:160]}...")
        return

    ensure_api_ready()
    seed_txt = REFERENCE_SEED if REFERENCE_SEED is not None else "random"
    extra = ""
    if IMAGE_ENGINE == "flow":
        extra = f" (model={FLOW_MODEL}, seed={seed_txt}" + (", upscale=2x" if FLOW_UPSCALE_2X else "") + ")"
    log(f"Engine: {IMAGE_ENGINE}{extra} | text-to-image, без персонажа")

    global_log = load_json(VISUAL_ROOT / "generation_log.json", [])
    if not isinstance(global_log, list):
        global_log = []
    log_lock = Lock()

    # Генерируем каждый язык последовательно (параллелизм внутри языка)
    for lang, items in jobs:
        # Пишем manifest
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
