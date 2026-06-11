#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DE/PL/RU reference-image IMAGE generator — Fast-Gen V4 Flower.

Читает промпты из ТРЁХ отдельных файлов (генерирует psych_prompt_pipeline.py):
  ПРОМПТЫ/prompts_de.txt
  ПРОМПТЫ/prompts_pl.txt
  ПРОМПТЫ/prompts_ru.txt

Для каждого языка генерирует СВОИ картинки:
  ВИЗУАЛ/DE/001.png, 002.png ...
  ВИЗУАЛ/PL/001.png, 002.png ...
  ВИЗУАЛ/RU/001.png, 001.png ...

Имена файлов (001.png) соответствуют полю image_filename из image_times_{lang}.json,
то есть монтажный скрипт video_creator_timecoded.py подхватит их автоматически.

Быстрый запуск:
  cd "/Users/aleksandrtomilov/Desktop/ПСИХОЛОГИЯ ГЕРМАНИЯ ПОЛЬША/ВИЗУАЛ"
  python3 flower_image_generator.py --workers 2

Флаги:
  --workers N      параллельных задач (начни с 1-2, снизь если 429)
  --lang DE PL RU  обработать только нужные языки
  --no-skip        перегенерировать уже существующие
  --dry-run        только показать промпты, API не вызывать
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

# Имена файлов промптов — генерирует psych_prompt_pipeline.py
PROMPTS_FILE_BY_LANG: Dict[str, Path] = {
    "DE": PROMPTS_DIR / "prompts_de.txt",
    "PL": PROMPTS_DIR / "prompts_pl.txt",
    "RU": PROMPTS_DIR / "prompts_ru.txt",
}

CHARACTER_IMAGE_PATH = PROJECT_ROOT / "ВИЗУАЛ" / "МОЙ ПЕРСОНАЖ.png"

REFERENCE_NAME = os.getenv("FLOWER_REFERENCE_NAME", "Alex").strip() or "Alex"

LOCALES = ["DE", "PL", "RU"]

ASPECT_RATIO = os.getenv("FAST_GEN_IMAGE_ASPECT_RATIO", "16:9")

# Движок генерации:
#   "flow"   -> /api/v4/flow/image/generate  (Nano Banana 2 = NARWHAL),
#               reference_images + seed, держит лицо и 16:9. 4 кредита/картинка.
#   "flower" -> /api/v4/flower/image/generate (Nano Banana 2 via Flower),
#               img2img edit с reference_image, 1 кредит/картинка.
IMAGE_ENGINE = os.getenv("FAST_GEN_IMAGE_ENGINE", "flower").strip().lower() or "flower"

# Модель для flow: NARWHAL (Nano Banana 2), GEM_PIX_2 (Nano Pro), IMAGEN_3_5 (Imagen 4)
FLOW_MODEL = os.getenv("FAST_GEN_FLOW_MODEL", "NARWHAL").strip() or "NARWHAL"

# Фиксированный seed -> лицо персонажа воспроизводится стабильно от кадра к кадру.
# Пусто/<0 = не передавать seed (каждый раз случайный).
_SEED_RAW = os.getenv("FAST_GEN_SEED", "777").strip()
REFERENCE_SEED: Optional[int] = int(_SEED_RAW) if _SEED_RAW.lstrip("-").isdigit() and int(_SEED_RAW) >= 0 else None

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

STORAGE_UPLOAD_URL = "https://storage.fast-gen.ai/upload"
STORAGE_GET_URL    = "https://storage.fast-gen.ai/file/{file_hash}"

IMAGE_INPUT_MODE       = os.getenv("FAST_GEN_IMAGE_INPUT_MODE", "auto").strip().lower() or "auto"
MAX_INLINE_IMAGE_BYTES = int(os.getenv("FAST_GEN_MAX_INLINE_IMAGE_BYTES", str(4_800_000)))


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


def ensure_reference_name_ready(name: str) -> None:
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,40}", name):
        print(f"[ERROR] REFERENCE_NAME должен быть латиницей: {name!r}")
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


def copy_image(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    if not file_ok(dst):
        raise RuntimeError(f"Копия не сохранилась: {dst}")


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


# Формат psych_prompt_pipeline.py:
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
    Парсит формат psych_prompt_pipeline.py:
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


def write_manifest(locale: str, items: Sequence[PromptItem], reference_image: Path) -> None:
    locale_dir = VISUAL_ROOT / locale
    locale_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = locale_dir / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["index", "filename", "reference_name", "reference_image", "prompt"])
        writer.writeheader()
        for item in items:
            writer.writerow({
                "index": item.index,
                "filename": f"{item.index:03d}.png",
                "reference_name": REFERENCE_NAME,
                "reference_image": str(reference_image),
                "prompt": item.prompt,
            })


# =========================
# HTTP HELPERS
# =========================

def _check_api_json(data: dict, *, label: str) -> dict:
    if not isinstance(data, dict):
        raise RuntimeError(f"{label}: API вернул не JSON-объект")
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
# DATA URI / STORAGE
# =========================

def guess_mime(path: Path) -> str:
    s = path.suffix.lower()
    return {"jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}.get(s, "image/jpeg")


def to_data_uri(path: Path) -> str:
    return f"data:{guess_mime(path)};base64," + base64.b64encode(path.read_bytes()).decode()


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


def upload_to_storage(path: Path) -> str:
    mime = guess_mime(path)
    log(f"    upload to storage: {path.name} ({path.stat().st_size / 1024 / 1024:.2f} MB)")
    attempt = 0
    while True:
        attempt += 1
        try:
            with path.open("rb") as f:
                resp = requests.post(STORAGE_UPLOAD_URL, headers={"X-API-Key": API_KEY},
                                     files={"file": (path.name, f, mime)}, timeout=REQUEST_TIMEOUT)
            if resp.status_code in {400, 401, 403, 404, 422}:
                raise FatalApiError(f"STORAGE UPLOAD: HTTP {resp.status_code}: {resp.text}")
            if resp.status_code >= 400:
                raise RuntimeError(f"STORAGE UPLOAD: HTTP {resp.status_code}: {resp.text}")
            data = resp.json()
            fh = data.get("file_hash") or (data.get("result") or {}).get("file_hash")
            if not fh:
                raise RuntimeError(f"STORAGE UPLOAD: file_hash не найден: {pretty_json(data)}")
            fh = str(fh).strip()
            return fh if fh.startswith("file:") else f"file:{fh}"
        except (KeyboardInterrupt, FatalApiError):
            raise
        except Exception as e:
            if not should_retry(attempt):
                raise
            log(f"[WARN] storage upload attempt {attempt} failed: {e}, retry in {RETRY_DELAY_SEC}s...")
            time.sleep(RETRY_DELAY_SEC)


def resolve_file_ref(file_ref: str) -> str:
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
# REFERENCE IMAGE
# =========================

# Жёсткий лимит API на reference_image — 5 MB (см. ImageInput в OpenAPI).
# Берём запас, т.к. base64 раздувает размер примерно на +33% только при inline,
# но сам файл-референс тоже не должен превышать 5 MB. Держим исходник <= лимита.
REFERENCE_MAX_BYTES = int(os.getenv("FAST_GEN_REFERENCE_MAX_BYTES", str(4_500_000)))


def prepare_reference_file(src: Path) -> Path:
    """
    Гарантирует, что файл персонажа <= REFERENCE_MAX_BYTES, иначе API его отвергнет
    (HTTP 422) и персонаж в результате "поплывёт". Если файл больше — ужимаем через
    Pillow (без потери пропорций), сохраняя один и тот же кадр для всех генераций.
    Если Pillow нет, а файл велик — громко предупреждаем.
    """
    size = src.stat().st_size
    if size <= REFERENCE_MAX_BYTES:
        return src

    log(f"[WARN] Референс {size / 1024 / 1024:.2f} MB > лимита API 5 MB — ужимаю.")
    try:
        from PIL import Image  # type: ignore
    except Exception:
        log("[ERROR] Pillow не установлен (pip install pillow), не могу ужать референс.")
        log("[ERROR] API может отклонить картинку >5MB и персонаж не сохранится.")
        return src

    img = Image.open(src)
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")

    out = src.with_name(f".{src.stem}_ref_resized.jpg")
    # JPEG даёт стабильно маленький размер; персонаж от этого не меняется.
    rgb = img.convert("RGB")
    max_side = max(rgb.size)
    quality = 92
    while True:
        work = rgb
        if max_side > 1536:
            scale = 1536 / max_side
            work = rgb.resize((int(rgb.width * scale), int(rgb.height * scale)), Image.LANCZOS)
        work.save(out, format="JPEG", quality=quality, optimize=True)
        if out.stat().st_size <= REFERENCE_MAX_BYTES or quality <= 60:
            break
        quality -= 8
        max_side = int(max_side * 0.9)
    log(f"    ужал референс -> {out.name} ({out.stat().st_size / 1024 / 1024:.2f} MB, q={quality})")
    return out


def _encode_one_reference(image_path: Path) -> Tuple[str, str, str]:
    """Возвращает (image_input, mode, resolved_path_str) для одного файла-референса."""
    resolved = resolve_existing_path(image_path)
    if not resolved.exists():
        raise FileNotFoundError(
            f"Не найден файл персонажа:\n  {image_path}\n"
            "Проверь имя файла. На macOS буква Й бывает в разной Unicode-форме."
        )
    resolved = prepare_reference_file(resolved)
    size = resolved.stat().st_size
    # По умолчанию шлём inline (data_uri): нет TTL хранилища (1 час), а значит
    # один и тот же референс гарантированно доступен всю длинную генерацию.
    if IMAGE_INPUT_MODE == "upload" or (IMAGE_INPUT_MODE == "auto" and size > MAX_INLINE_IMAGE_BYTES):
        return upload_to_storage(resolved), "file_ref", str(resolved)
    return to_data_uri(resolved), "data_uri", str(resolved)


def load_reference_image(image_paths: Sequence[Path]) -> dict:
    """
    Кодирует один или несколько референсов персонажа.
    Несколько ракурсов (лицо/в полный рост/профиль) сильно повышают стабильность лица.
    flow принимает до 10 reference_images; flower использует только первый.
    """
    if isinstance(image_paths, (str, Path)):
        image_paths = [Path(image_paths)]
    images: List[str] = []
    paths: List[str] = []
    mode = "data_uri"
    for p in image_paths:
        img, mode, rp = _encode_one_reference(Path(p))
        images.append(img)
        paths.append(rp)
    if IMAGE_ENGINE == "flower" and len(images) > 1:
        log(f"[WARN] flower использует только 1 референс — беру первый из {len(images)}.")
    return {
        "name": REFERENCE_NAME,
        "images": images,           # список для flow.reference_images
        "image": images[0],         # обратная совместимость
        "mode": mode,
        "paths": paths,
        "path": paths[0],
    }


# =========================
# FLOWER IMAGE API
# =========================

# Намеренно НЕ просим "свежую композицию" — это раньше заставляло модель
# перерисовывать персонажа. Язык на идентичность лица не влияет, поэтому
# подсказки нейтральные.
LANG_HINTS = {
    "DE": "",
    "PL": "",
    "RU": "",
}


def build_api_prompt(scene_prompt: str, reference_name: str, locale: str) -> str:
    """
    Промпт для режима img2img / character-consistency (Nano Banana 2 via Flower).

    Когда передан reference_image, эндпоинт /flower/image/generate работает как
    flower_image_edit: промпт — это ИНСТРУКЦИЯ ПО РЕДАКТИРОВАНИЮ исходной картинки,
    а не описание новой сцены с нуля. Поэтому персонажа фиксируем максимально жёстко,
    а сцену описываем как "помести того же человека в это окружение".
    """
    scene = re.sub(r"\bthis character\b", reference_name, scene_prompt, flags=re.IGNORECASE)
    scene = re.sub(r"\bthe character\b", reference_name, scene, flags=re.IGNORECASE)
    return (
        "TASK: keep one fixed recurring character identical across images. "
        f"The attached reference image(s) define the EXACT appearance of {reference_name}. "
        f"Keep {reference_name} 100% identical to the reference: the same face and facial "
        "features, the same face shape, skin tone, eye color, eyebrows, nose, lips, the same "
        "hair color and hairstyle, the same age and the same body type. "
        "Treat the face as locked — do NOT redraw, beautify, age, de-age, change ethnicity, "
        "swap gender, or generate a different-looking person. It must look like the very same "
        "individual, clearly recognizable as the reference. "
        f"{reference_name} is the only person in the image — no extra people, no crowd. "
        "Only change the pose, the action, the clothing if the scene asks for it, the lighting "
        "and the environment around the person to fit the scene below. "
        "Photorealistic, natural consistent lighting on the face, sharp focus, "
        "no text, no subtitles, no watermark, no logo. "
        f"Scene to place {reference_name} into: {scene}"
    )


def submit_generate(prompt: str, ref: dict, locale: str) -> str:
    api_prompt = build_api_prompt(prompt, ref["name"], locale)

    if IMAGE_ENGINE == "flow":
        # Nano Banana 2 (NARWHAL) с reference_images — генерирует НОВУЮ сцену,
        # используя референс(ы) как образец внешности персонажа. Держит лицо
        # и соблюдает aspect_ratio (16:9). seed фиксирует лицо между кадрами.
        payload: Dict[str, Any] = {
            "prompt": api_prompt,
            "aspect_ratio": ASPECT_RATIO,
            "model": FLOW_MODEL,
            "reference_images": ref["images"],
        }
        if REFERENCE_SEED is not None:
            payload["seed"] = REFERENCE_SEED
        data = post_json(FLOW_IMAGE_ENDPOINT, payload, label="FLOW IMAGE GENERATE")
    else:
        # flower: img2img-редактирование одной картинки (1 кредит, лицо держит слабее).
        payload = {
            "prompt": api_prompt,
            "aspect_ratio": ASPECT_RATIO,
            "reference_image": ref["images"][0],
        }
        data = post_json(FLOWER_IMAGE_ENDPOINT, payload, label="FLOWER IMAGE GENERATE")

    op_id = data.get("operation_id")
    if not op_id:
        raise RuntimeError(f"API не вернул operation_id: {pretty_json(data)}")
    return str(op_id)


def poll_operation(op_id: str) -> dict:
    endpoint = V4_OPERATION_ENDPOINT.format(operation_id=op_id)
    started = time.time()
    while True:
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
            return data
        raise RuntimeError(f"op {op_id}: {state}: {data.get('error') or pretty_json(data)}")


def generate_image(item: PromptItem, out_path: Path, ref: dict, locale: str) -> dict:
    attempt = 0
    while True:
        attempt += 1
        op_id = None
        try:
            log(f"[{locale}] {out_path.name} (#{item.index}) attempt {attempt}")
            op_id = submit_generate(item.prompt, ref, locale)
            log(f"    op_id: {op_id}")
            op_data = poll_operation(op_id)
            save_operation_result(op_data.get("result"), out_path)
            log(f"    saved: {out_path}")
            return {"index": item.index, "locale": locale, "status": "success",
                    "output": str(out_path), "op_id": op_id, "attempts": attempt}
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
    ref: dict,
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
        futures = {ex.submit(generate_image, item, output_path(locale, item), ref, locale): item
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
    global BASE_URL, REFERENCE_NAME, SKIP_EXISTING, ASPECT_RATIO, MAX_IMAGE_WORKERS
    global OPERATION_POLL_SEC, CHARACTER_IMAGE_PATH, PROMPTS_DIR, VISUAL_ROOT
    global IMAGE_ENGINE, FLOW_MODEL, REFERENCE_SEED

    parser = argparse.ArgumentParser(
        description="Генерирует DE/PL/RU картинки с единым персонажем через Fast-Gen (flow/flower)"
    )
    parser.add_argument("--lang", nargs="+", choices=LOCALES, default=None,
                        help="Языки для обработки. По умолчанию — все три.")
    parser.add_argument("--prompts-dir", default=str(PROMPTS_DIR),
                        help="Папка с файлами prompts_de.txt / prompts_pl.txt / prompts_ru.txt")
    parser.add_argument("--visual-root", default=str(VISUAL_ROOT), help="Папка ВИЗУАЛ")
    parser.add_argument("--character-image", nargs="+", default=[str(CHARACTER_IMAGE_PATH)],
                        help="Один или несколько PNG/JPG/WebP файлов персонажа (референсы). "
                             "Несколько ракурсов сильно улучшают стабильность лица (flow: до 10).")
    parser.add_argument("--reference-name", default=REFERENCE_NAME,
                        help="Латинское имя персонажа, например Alex")
    parser.add_argument("--engine", choices=["flow", "flower"], default=IMAGE_ENGINE,
                        help="flow = Nano Banana 2 + reference_images + seed (стабильный персонаж, 4 кр). "
                             "flower = img2img edit (1 кр, лицо держит слабее).")
    parser.add_argument("--flow-model", default=FLOW_MODEL,
                        choices=["NARWHAL", "GEM_PIX_2", "IMAGEN_3_5"],
                        help="Модель для flow: NARWHAL=Nano Banana 2 (реком.), GEM_PIX_2=Nano Pro, IMAGEN_3_5=Imagen 4")
    parser.add_argument("--seed", type=int, default=(REFERENCE_SEED if REFERENCE_SEED is not None else -1),
                        help="Фиксированный seed для воспроизводимости лица (flow). -1 = случайный.")
    parser.add_argument("--api-base", default=BASE_URL)
    parser.add_argument("--aspect-ratio", default=ASPECT_RATIO, choices=["16:9", "9:16", "1:1"])
    parser.add_argument("--workers", type=int, default=MAX_IMAGE_WORKERS,
                        help="Параллельных задач на язык (начни с 1-2, снизь при 429)")
    parser.add_argument("--poll-sec", type=int, default=OPERATION_POLL_SEC)
    parser.add_argument("--no-skip", action="store_true", help="Перегенерировать уже существующие")
    parser.add_argument("--dry-run", action="store_true", help="Показать промпты без API-вызовов")
    args = parser.parse_args()

    BASE_URL             = args.api_base.rstrip("/")
    PROMPTS_DIR          = Path(args.prompts_dir).expanduser()
    VISUAL_ROOT          = Path(args.visual_root).expanduser()
    CHARACTER_IMAGE_PATHS = [Path(p).expanduser() for p in args.character_image]
    CHARACTER_IMAGE_PATH = CHARACTER_IMAGE_PATHS[0]
    REFERENCE_NAME       = (args.reference_name or "Alex").strip() or "Alex"
    ASPECT_RATIO         = args.aspect_ratio
    MAX_IMAGE_WORKERS    = max(1, args.workers)
    OPERATION_POLL_SEC   = max(1, args.poll_sec)
    SKIP_EXISTING        = not args.no_skip
    IMAGE_ENGINE         = args.engine
    FLOW_MODEL           = args.flow_model
    REFERENCE_SEED       = args.seed if args.seed is not None and args.seed >= 0 else None

    langs_to_process = args.lang or LOCALES

    ensure_reference_name_ready(REFERENCE_NAME)
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
                preview = build_api_prompt(item.prompt, REFERENCE_NAME, lang)
                log(f"  {item.index:03d}.png <- {preview[:160]}...")
        return

    ensure_api_ready()
    ref = load_reference_image(CHARACTER_IMAGE_PATHS)
    seed_txt = REFERENCE_SEED if REFERENCE_SEED is not None else "random"
    log(f"Engine: {IMAGE_ENGINE}" + (f" (model={FLOW_MODEL}, seed={seed_txt})" if IMAGE_ENGINE == "flow" else ""))
    log(f"Reference: {len(ref['images'])} шт., режим {ref['mode']}")
    for p in ref["paths"]:
        log(f"  - {p}")

    global_log = load_json(VISUAL_ROOT / "generation_log.json", [])
    if not isinstance(global_log, list):
        global_log = []
    log_lock = Lock()

    # Генерируем каждый язык последовательно (параллелизм внутри языка)
    for lang, items in jobs:
        # Пишем manifest
        write_manifest(lang, items, CHARACTER_IMAGE_PATH)
        process_locale(lang, items, ref, global_log, log_lock, MAX_IMAGE_WORKERS)

    print(f"\n{'=' * 42}")
    print("ALL DONE")
    for lang, _ in jobs:
        count = len(list((VISUAL_ROOT / lang).glob("*.png")))
        print(f"  {lang}: {count} картинок в {VISUAL_ROOT / lang}/")
    print(f"{'=' * 42}")


if __name__ == "__main__":
    main()
