#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FAST visuals generator rewritten for Fast-Gen V4 Flower / Veo 3.1.

Что делает:
1) Читает промпты отдельно для каждого языка из точных файлов:
   GE -> /Users/aleksandrtomilov/Desktop/ПТИЦЫ АВТОМАТИЗАЦИЯ/ПРОМПТЫ/DE_немецкий/de_GE_prompts.txt
   ES -> /Users/aleksandrtomilov/Desktop/ПТИЦЫ АВТОМАТИЗАЦИЯ/ПРОМПТЫ/ES_испанский/es_ES_prompts.txt
   PL -> /Users/aleksandrtomilov/Desktop/ПТИЦЫ АВТОМАТИЗАЦИЯ/ПРОМПТЫ/PL_польский/pl_PL_prompts.txt
   RU -> /Users/aleksandrtomilov/Desktop/ПТИЦЫ АВТОМАТИЗАЦИЯ/ПРОМПТЫ/RU_русский/ru_RU_prompts.txt

2) Создаёт папки:
   /Users/aleksandrtomilov/Desktop/ПТИЦЫ АВТОМАТИЗАЦИЯ/ВИЗУАЛ/RU
   /Users/aleksandrtomilov/Desktop/ПТИЦЫ АВТОМАТИЗАЦИЯ/ВИЗУАЛ/GE
   /Users/aleksandrtomilov/Desktop/ПТИЦЫ АВТОМАТИЗАЦИЯ/ВИЗУАЛ/PL
   /Users/aleksandrtomilov/Desktop/ПТИЦЫ АВТОМАТИЗАЦИЯ/ВИЗУАЛ/ES

3) Для каждого языка:
   - первые ANIMATE_FIRST_N промптов генерирует как MP4-видео через Flower / Veo 3.1;
   - остальные промпты генерирует как PNG-картинки через Flower image generate.

4) Важное отличие от старой версии:
   - старые deprecated v3 endpoints больше не используются;
   - все генерации идут через V4 async endpoints;
   - polling идёт через общий endpoint: /api/v4/operations/{operation_id}.

Запуск:
   cd "/Users/aleksandrtomilov/Desktop/ПТИЦЫ АВТОМАТИЗАЦИЯ"
   source venv/bin/activate
   python "/Users/aleksandrtomilov/Desktop/ПТИЦЫ АВТОМАТИЗАЦИЯ/ВИЗУАЛ/flower_veo31_visual_batch_generator.py"

Ключ лучше хранить так:
   echo 'export FAST_GEN_API_KEY="ТВОЙ_КЛЮЧ"' >> ~/.zshrc
   source ~/.zshrc

Полезные настройки:
   export FAST_GEN_ANIMATE_FIRST_N="2"
   export FAST_GEN_IMAGE_WORKERS="4"
   export FAST_GEN_VIDEO_WORKERS="1"
   export FAST_GEN_LOCALE_WORKERS="1"
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

BASE_DIR = Path("/Users/aleksandrtomilov/Desktop/ПТИЦЫ АВТОМАТИЗАЦИЯ")
PROMPTS_ROOT = BASE_DIR / "ПРОМПТЫ"
VISUAL_ROOT = BASE_DIR / "ВИЗУАЛ"

LOCALES = ["RU", "GE", "PL", "ES"]

PROMPT_FILES_BY_LOCALE: Dict[str, Path] = {
    "GE": Path("/Users/aleksandrtomilov/Desktop/ПТИЦЫ АВТОМАТИЗАЦИЯ/ПРОМПТЫ/DE_немецкий/de_GE_prompts.txt"),
    "ES": Path("/Users/aleksandrtomilov/Desktop/ПТИЦЫ АВТОМАТИЗАЦИЯ/ПРОМПТЫ/ES_испанский/es_ES_prompts.txt"),
    "PL": Path("/Users/aleksandrtomilov/Desktop/ПТИЦЫ АВТОМАТИЗАЦИЯ/ПРОМПТЫ/PL_польский/pl_PL_prompts.txt"),
    "RU": Path("/Users/aleksandrtomilov/Desktop/ПТИЦЫ АВТОМАТИЗАЦИЯ/ПРОМПТЫ/RU_русский/ru_RU_prompts.txt"),
}

# Flower/Veo 3.1 expects simple aspect ratios, not old enum strings.
IMAGE_ASPECT_RATIO = os.getenv("FAST_GEN_IMAGE_ASPECT_RATIO", "16:9")  # 16:9, 9:16, 1:1
VIDEO_ASPECT_RATIO = os.getenv("FAST_GEN_VIDEO_ASPECT_RATIO", "16:9")  # 16:9, 9:16

# Первые N промптов в каждой папке будут видео.
ANIMATE_FIRST_N = int(os.getenv("FAST_GEN_ANIMATE_FIRST_N", "2"))

REQUEST_TIMEOUT = int(os.getenv("FAST_GEN_REQUEST_TIMEOUT", "300"))
OPERATION_POLL_SEC = int(os.getenv("FAST_GEN_OPERATION_POLL_SEC", "10"))
RETRY_DELAY_SEC = int(os.getenv("FAST_GEN_RETRY_DELAY_SEC", "8"))
SKIP_EXISTING = True

# Если файла промптов для языка нет или он пустой/битый — не падать, а пропустить этот язык.
# Можно отключить пропуск и вернуть строгий режим так: export FAST_GEN_SKIP_MISSING_PROMPTS="0"
SKIP_MISSING_PROMPTS = os.getenv("FAST_GEN_SKIP_MISSING_PROMPTS", "1").strip().lower() not in ("0", "false", "no", "нет")

MAX_IMAGE_WORKERS = int(os.getenv("FAST_GEN_IMAGE_WORKERS", "4"))
MAX_VIDEO_WORKERS = int(os.getenv("FAST_GEN_VIDEO_WORKERS", "1"))
MAX_LOCALE_WORKERS = int(os.getenv("FAST_GEN_LOCALE_WORKERS", "1"))

# V4 Flower endpoints.
FLOWER_IMAGE_ENDPOINT = "/api/v4/flower/image/generate"       # Nano Banana 2 via Flower
FLOWER_VIDEO_FROM_IMAGE_ENDPOINT = "/api/v4/flower/video/from-image"  # Veo 3.1 via Flower
V4_OPERATION_ENDPOINT = "/api/v4/operations/{operation_id}"

# Storage нужен, чтобы передавать стартовую картинку в video/from-image как file:ref.
STORAGE_UPLOAD_URL = "https://storage.fast-gen.ai/upload"
STORAGE_GET_URL = "https://storage.fast-gen.ai/file/{file_hash}"

GLOBAL_LOG_FILE = VISUAL_ROOT / "generation_log.json"

TIMECODE_RE = re.compile(
    r"^(?:(\d+)\s*\|\s*)?"
    r"(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*"
    r"(\d{2}:\d{2}:\d{2},\d{3})"
    r"(?:\s*\|.*)?$"
)


# =========================
# DATA MODELS
# =========================

@dataclass
class PromptItem:
    index: int
    start_tc: str
    end_tc: str
    prompt: str


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
# DATA URI / FILE REF HELPERS
# =========================

def split_data_uri(data_uri: str) -> Tuple[str, bytes]:
    if not isinstance(data_uri, str) or not data_uri.startswith("data:"):
        raise ValueError(f"Ответ API не похож на data URI: {str(data_uri)[:120]}")
    if "," not in data_uri:
        raise ValueError("Некорректный data URI: нет запятой")
    header, b64 = data_uri.split(",", 1)
    mime = header.replace("data:", "").replace(";base64", "")
    return mime, base64.b64decode(b64)


def save_data_uri_to_file(data_uri: str, path: Path) -> None:
    _mime, raw = split_data_uri(data_uri)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"Файл не сохранился или пустой: {path}")


def normalize_file_ref(file_hash: str) -> str:
    fh = file_hash.strip()
    return fh if fh.startswith("file:") else f"file:{fh}"


def resolve_result_data_uri(value: Any) -> str:
    """
    V4 обычно возвращает result как список:
    - ["data:image/png;base64,..."]
    - ["data:video/mp4;base64,..."]
    - ["file:abc..."] если polling был с result_format=ref
    """
    if isinstance(value, list):
        if not value:
            raise RuntimeError("Пустой список result")
        value = value[0]

    if not isinstance(value, str):
        raise RuntimeError(f"Неожиданный result от API: {pretty_json(value)}")

    if value.startswith("data:"):
        return value

    if value.startswith("file:"):
        file_hash = value.replace("file:", "", 1)
        url = STORAGE_GET_URL.format(file_hash=file_hash)
        resp = requests.get(url, headers={"X-API-Key": API_KEY}, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        try:
            data = resp.json()
            for key in ["data_uri", "result", "file", "content"]:
                candidate = data.get(key)
                if isinstance(candidate, str) and candidate.startswith("data:"):
                    return candidate
            raise RuntimeError(f"Storage не вернул data URI: {pretty_json(data)}")
        except ValueError:
            text = resp.text.strip()
            if text.startswith("data:"):
                return text
            raise RuntimeError(f"Storage вернул не data URI: {text[:200]}")

    raise RuntimeError(f"Неизвестный формат result: {value[:200]}")


# =========================
# HTTP HELPERS
# =========================

def request_with_retries_post(url: str, payload: dict, *, label: str) -> dict:
    attempt = 0
    while True:
        attempt += 1
        try:
            resp = requests.post(url, headers=headers(), json=payload, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 429:
                raise RuntimeError(f"Rate limit 429: {resp.text}")
            if resp.status_code >= 400:
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text}")
            return resp.json()
        except KeyboardInterrupt:
            raise
        except Exception as e:
            log(f"[WARN] {label}: попытка {attempt} не удалась: {e}")
            log(f"       повтор через {RETRY_DELAY_SEC} сек...")
            time.sleep(RETRY_DELAY_SEC)


def request_get_with_retries(url: str, *, label: str, params: Optional[dict] = None) -> dict:
    attempt = 0
    while True:
        attempt += 1
        try:
            resp = requests.get(url, headers={"X-API-Key": API_KEY}, params=params, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 429:
                raise RuntimeError(f"Rate limit 429: {resp.text}")
            if resp.status_code >= 400:
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text}")
            return resp.json()
        except KeyboardInterrupt:
            raise
        except Exception as e:
            log(f"[WARN] {label}: GET попытка {attempt} не удалась: {e}")
            log(f"       повтор через {RETRY_DELAY_SEC} сек...")
            time.sleep(RETRY_DELAY_SEC)


def upload_file_to_storage(image_path: Path) -> str:
    file_size_mb = image_path.stat().st_size / (1024 * 1024)
    log(f"    upload start image to storage: {image_path.name} ({file_size_mb:.2f} MB)")

    attempt = 0
    while True:
        attempt += 1
        try:
            with image_path.open("rb") as f:
                resp = requests.post(
                    STORAGE_UPLOAD_URL,
                    headers={"X-API-Key": API_KEY},
                    files={"file": (image_path.name, f, "image/png")},
                    timeout=REQUEST_TIMEOUT,
                )
            if resp.status_code >= 400:
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text}")

            data = resp.json()
            file_hash = data.get("file_hash")
            if not file_hash and isinstance(data.get("result"), dict):
                file_hash = data["result"].get("file_hash")

            if not data.get("success", True) or not file_hash:
                raise RuntimeError(f"Не удалось загрузить файл в storage: {pretty_json(data)}")

            return normalize_file_ref(file_hash)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            log(f"[WARN] storage upload попытка {attempt} не удалась: {e}")
            log(f"       повтор через {RETRY_DELAY_SEC} сек...")
            time.sleep(RETRY_DELAY_SEC)


# =========================
# V4 FLOWER API
# =========================

def start_flower_image(prompt: str) -> str:
    url = clean_base_url(BASE_URL) + FLOWER_IMAGE_ENDPOINT
    payload: Dict[str, Any] = {
        "prompt": prompt,
        "aspect_ratio": IMAGE_ASPECT_RATIO,
    }

    data = request_with_retries_post(url, payload, label="FLOWER IMAGE")
    operation_id = data.get("operation_id")
    if not operation_id:
        raise RuntimeError(f"Flower image API не вернул operation_id: {pretty_json(data)}")
    return operation_id


def start_flower_video_from_image(image_ref: str, prompt: str) -> str:
    url = clean_base_url(BASE_URL) + FLOWER_VIDEO_FROM_IMAGE_ENDPOINT
    payload: Dict[str, Any] = {
        "image": image_ref,
        "prompt": build_video_prompt(prompt),
        "aspect_ratio": VIDEO_ASPECT_RATIO,
    }

    data = request_with_retries_post(url, payload, label="FLOWER VEO 3.1 VIDEO FROM IMAGE")
    operation_id = data.get("operation_id")
    if not operation_id:
        raise RuntimeError(f"Flower video API не вернул operation_id: {pretty_json(data)}")
    return operation_id


def poll_operation(operation_id: str, *, label: str) -> str:
    url = clean_base_url(BASE_URL) + V4_OPERATION_ENDPOINT.format(operation_id=operation_id)

    while True:
        data = request_get_with_retries(
            url,
            label=f"{label} STATUS {operation_id}",
            params={"result_format": "data_uri"},
        )
        status = data.get("status")

        if status in ("pending", "processing"):
            log(f"    {label.lower()} status: {status}")
            time.sleep(OPERATION_POLL_SEC)
            continue

        if status == "success":
            result = data.get("result")
            if not result:
                raise RuntimeError(f"{label} завершилось, но result пустой: {pretty_json(data)}")
            return resolve_result_data_uri(result)

        if status == "error":
            error = data.get("error") or pretty_json(data)
            raise RuntimeError(f"{label} закончилось ошибкой: {error}")

        raise RuntimeError(f"Неизвестный статус {label}: {pretty_json(data)}")


def generate_image_flower(prompt: str) -> Tuple[str, str]:
    operation_id = start_flower_image(prompt)
    log(f"    image operation_id: {operation_id}")
    data_uri = poll_operation(operation_id, label="IMAGE")
    return data_uri, operation_id


def build_video_prompt(prompt: str) -> str:
    return (
        "Animate this exact illustrated scene with subtle cinematic motion. "
        "Keep the same subject, composition, colors, lighting, and hand-drawn documentary style. "
        "Use gentle camera push-in, natural atmospheric movement, soft light motion, and realistic depth. "
        "Do not add text, subtitles, labels, logos, or watermark. "
        f"Scene: {prompt}"
    )


# =========================
# GENERATION LOGIC
# =========================

def generate_image_until_done(item: PromptItem, out_path: Path) -> dict:
    attempt = 0
    while True:
        attempt += 1
        try:
            log(f"[IMAGE/FLOWER] {out_path.name} | attempt {attempt}")
            data_uri, operation_id = generate_image_flower(item.prompt)
            save_data_uri_to_file(data_uri, out_path)
            log(f"    saved: {out_path}")
            return {
                "index": item.index,
                "type": "image",
                "output": str(out_path),
                "status": "success",
                "attempts": attempt,
                "operation_id": operation_id,
            }
        except KeyboardInterrupt:
            raise
        except Exception as e:
            log(f"[IMAGE ERROR] {out_path.name}: {e}")
            log(f"    retry in {RETRY_DELAY_SEC}s...")
            time.sleep(RETRY_DELAY_SEC)


def generate_video_until_done(item: PromptItem, out_path: Path, locale_dir: Path) -> dict:
    attempt = 0
    tmp_dir = locale_dir / "_video_start_images_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_img_path = tmp_dir / f"start_{item.index:04d}.png"

    while True:
        attempt += 1
        try:
            log(f"[VIDEO/FLOWER VEO 3.1] {out_path.name} | attempt {attempt}")

            # 1) Делаем стартовую картинку через Flower image generate.
            image_data_uri, image_operation_id = generate_image_flower(item.prompt)
            save_data_uri_to_file(image_data_uri, tmp_img_path)

            # 2) Загружаем стартовую картинку в storage и передаём file:ref в Flower video/from-image.
            file_ref = upload_file_to_storage(tmp_img_path)
            log(f"    storage ref: {file_ref}")

            # 3) Запускаем Veo 3.1 via Flower.
            video_operation_id = start_flower_video_from_image(file_ref, item.prompt)
            log(f"    video operation_id: {video_operation_id}")

            # 4) Ждём результат через V4 operations endpoint.
            video_data_uri = poll_operation(video_operation_id, label="VIDEO")
            save_data_uri_to_file(video_data_uri, out_path)

            try:
                tmp_img_path.unlink(missing_ok=True)
                if not any(tmp_dir.iterdir()):
                    tmp_dir.rmdir()
            except Exception:
                pass

            log(f"    saved: {out_path}")
            return {
                "index": item.index,
                "type": "video",
                "output": str(out_path),
                "status": "success",
                "attempts": attempt,
                "image_operation_id": image_operation_id,
                "operation_id": video_operation_id,
            }
        except KeyboardInterrupt:
            raise
        except Exception as e:
            log(f"[VIDEO ERROR] {out_path.name}: {e}")
            log(f"    retry in {RETRY_DELAY_SEC}s...")
            time.sleep(RETRY_DELAY_SEC)


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
        is_video = item.index <= ANIMATE_FIRST_N
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


# =========================
# MAIN
# =========================

def main() -> None:
    global BASE_URL, PROMPTS_ROOT, VISUAL_ROOT, SKIP_EXISTING, GLOBAL_LOG_FILE

    parser = argparse.ArgumentParser(
        description="Generate RU/GE/PL/ES visuals from exact prompt files via Fast-Gen V4 Flower / Veo 3.1."
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

    log("Fast-Gen V4 Flower / Veo 3.1 visual generator")
    log(f"BASE_URL: {BASE_URL}")
    log(f"Prompts root: {PROMPTS_ROOT}")
    log("Prompt files:")
    for _locale in LOCALES:
        log(f"  {_locale}: {PROMPT_FILES_BY_LOCALE[_locale]}")
    log(f"Visual root: {VISUAL_ROOT}")
    log(f"Locales: {', '.join(LOCALES)}")
    log(f"First videos per locale: {ANIMATE_FIRST_N}")
    log(f"Image API: {FLOWER_IMAGE_ENDPOINT}")
    log(f"Video API: {FLOWER_VIDEO_FROM_IMAGE_ENDPOINT}")
    log(f"Polling API: {V4_OPERATION_ENDPOINT}")
    log(f"Aspect ratio: image={IMAGE_ASPECT_RATIO}, video={VIDEO_ASPECT_RATIO}")
    log(f"Speed: image_workers={MAX_IMAGE_WORKERS}, video_workers={MAX_VIDEO_WORKERS}, locale_workers={MAX_LOCALE_WORKERS}")

    prompts_by_locale = load_prompts_by_locale()

    global_log = load_json(GLOBAL_LOG_FILE, [])
    if not isinstance(global_log, list):
        global_log = []

    process_all_locales(prompts_by_locale, global_log)

    save_json(GLOBAL_LOG_FILE, global_log)
    log("\nALL DONE")
    log(f"Output: {VISUAL_ROOT}")


if __name__ == "__main__":
    main()
