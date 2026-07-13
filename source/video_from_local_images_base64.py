# -*- coding: utf-8 -*-
"""
Оживление локальных картинок в видео через media_gen_api V6 (провайдер flow,
операция flow_video_from_ingredients) без storage upload.

Что делает:
1. Берёт картинки из папки:
   /Users/aleksandrtomilov/Desktop/ПРОМПТЫ/КАРТИНКИ
2. Сохраняет видео в папку:
   /Users/aleksandrtomilov/Desktop/ПРОМПТЫ/ВИДЕО
3. Если есть файл промптов:
   - generated_prompts_detailed.txt
   - или generated_prompts.txt
   то берёт промпт по номеру картинки (0001 -> 1-й промпт, 0002 -> 2-й и т.д.)
4. Если промпта для картинки нет — использует DEFAULT_ANIMATION_PROMPT
5. Отправляет картинку прямо в запрос как data:image/...;base64,...
   (в inputs[]) без upload в storage
6. Работает в несколько потоков
7. Учитывает лимит 150 стартов видео в час

Соответствие документации media_gen_api V6 (openapi 3.1.0):
   - генерация идёт через единый endpoint POST /api/v6/generations
     (GenerationCreateRequest) с canonical operation id flow_video_from_ingredients
     (провайдер flow, модель flow-video-fast, 1 кредит);
   - картинка-ингредиент передаётся в inputs[] как V6MediaInput
     (data URI прямо в запросе, до 5 MB на inline-картинку);
   - aspect_ratio в форме n:n (^[1-9]\\d*:[1-9]\\d*$), напр. 16:9 / 9:16;
   - опциональные параметры запроса: seed, duration_seconds, resolution, ultra —
     отправляются только если заданы через env;
   - ответ на создание — GenerationAcceptedResponse (поле id);
   - статус через GET /api/v6/generations/{generation_id} -> GenerationStatusResponse:
     status = queued|running|succeeded|failed, results[], warnings, error, translations;
   - результат — GenerationResultItem: type (image|video|text), download_url,
     data (inline data URI), mime_type, metadata.storage_id.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


# =========================================================
# НАСТРОЙКИ
# =========================================================

# API-ключ берётся из окружения (не хардкодим секрет в файл, чтобы не утёк в git).
#   export FAST_GEN_API_KEY="ТВОЙ_КЛЮЧ"
API_KEY = (
    os.getenv("FAST_GEN_API_KEY")
    or os.getenv("FASTGEN_API_KEY")
    or os.getenv("MEDIA_GEN_API_KEY")
    or ""
)
BASE_URL = os.getenv("FAST_GEN_API_BASE", "https://api.fast-gen.ai")

BASE_DIR = Path("/Users/aleksandrtomilov/Desktop/ПРОМПТЫ")
IMAGES_DIR = Path("/Users/aleksandrtomilov/Desktop/ПРОМПТЫ/КАРТИНКИ")
VIDEOS_DIR = BASE_DIR / "ВИДЕО"

PROMPTS_FILE_PRIMARY = BASE_DIR / "generated_prompts_detailed.txt"
PROMPTS_FILE_FALLBACK = BASE_DIR / "generated_prompts.txt"

LOG_FILE = BASE_DIR / "video_generation_log.json"
RATE_LIMIT_FILE = BASE_DIR / "video_rate_limit_state.json"

MAX_WORKERS = 8

REQUEST_TIMEOUT = 120
POLL_INTERVAL = 20
POLL_TIMEOUT = 60 * 60 * 4
STATUS_LOG_EVERY = 30
MAX_RETRIES = 4

# V6 требует aspect_ratio в форме n:n, например 16:9 (ландшафт) или 9:16 (вертикаль).
ASPECT_RATIO = os.getenv("FAST_GEN_VIDEO_ASPECT_RATIO", "16:9")

# Canonical V6 operation id для image->video на flow (модель flow-video-fast).
# См. GET /api/v6/capabilities — flow_video_from_ingredients, provider=flow, 1 кредит.
# Flow, а НЕ flower/Veo: у Veo пул аккаунтов часто пуст ("нет доступных аккаунтов"),
# а flow-видео стабильно доступно.
VIDEO_OPERATION = os.getenv("FAST_GEN_VIDEO_OPERATION", "flow_video_from_ingredients")
# Необязательный shorthand-model (например flow-video-lite / flow-video-quality).
VIDEO_MODEL = os.getenv("FAST_GEN_VIDEO_MODEL") or None

# Опциональные параметры генерации (отправляются только если заданы через env).
_SEED_ENV = os.getenv("FAST_GEN_SEED", "").strip()
GENERATION_SEED: Optional[int] = int(_SEED_ENV) if _SEED_ENV.isdigit() else None

_DURATION_ENV = os.getenv("FAST_GEN_VIDEO_DURATION_SECONDS", "").strip()
VIDEO_DURATION_SECONDS: Optional[int] = int(_DURATION_ENV) if _DURATION_ENV.isdigit() else None

# resolution: например 480p или 720p (когда поддерживается моделью).
VIDEO_RESOLUTION = os.getenv("FAST_GEN_VIDEO_RESOLUTION") or None

# ultra: только Flow video — Ultra-tier аккаунты/лимиты.
VIDEO_ULTRA = os.getenv("FAST_GEN_VIDEO_ULTRA", "").strip().lower() in {"1", "true", "yes", "on"}

MAX_VIDEO_STARTS_PER_HOUR = 150
RATE_WINDOW_SECONDS = 3600

RETRY_SLEEP_429 = [20, 35, 60, 90]

# Ёмкостные/аккаунтные ошибки провайдера ("нет доступных аккаунтов" / "no available accounts")
# — это НЕ блок по контенту, а занятый пул аккаунтов провайдера. Повторяем терпеливо и долго.
CAPACITY_ERROR_MARKERS = (
    "no available account", "no accounts available", "no available accounts",
    "no free account", "all accounts", "account pool", "no account",
    "нет доступных аккаунт", "нет свободных аккаунт", "нет аккаунт",
    "try again later", "temporarily unavailable", "over capacity", "overloaded",
    "no capacity", "capacity", "please try again",
)
CAPACITY_RETRIES = max(1, int(os.getenv("FAST_GEN_CAPACITY_RETRIES", "40")))
CAPACITY_SLEEP_START = int(os.getenv("FAST_GEN_CAPACITY_BACKOFF_START_SEC", "15"))
CAPACITY_SLEEP_CAP = int(os.getenv("FAST_GEN_CAPACITY_BACKOFF_CAP_SEC", "120"))


def is_capacity_error(text: str) -> bool:
    low = str(text).lower()
    return any(marker in low for marker in CAPACITY_ERROR_MARKERS)

# Если отдельного промпта для картинки нет, будет использоваться этот.
DEFAULT_ANIMATION_PROMPT = (
    "Animate this image with gentle realistic motion, subtle cinematic camera movement, "
    "natural details, preserve the original composition, subject, lighting and style."
)

# По схеме V6MediaInput для inline data URI максимум 5 MB на одну картинку.
MAX_IMAGE_BYTES = 5 * 1024 * 1024

# V6 endpoints.
V6_GENERATIONS_ENDPOINT = "/api/v6/generations"
V6_GENERATION_STATUS_ENDPOINT = "/api/v6/generations/{generation_id}"

# V6MediaInput: data URI. aspect_ratio: n:n.
DATA_URI_RE = re.compile(r"^data:[^;]+;base64,")
ASPECT_RATIO_RE = re.compile(r"^[1-9]\d*:[1-9]\d*$")

# Терминальные статусы V6 (плюс совместимые синонимы на всякий случай).
TERMINAL_SUCCESS = {"succeeded", "success", "completed", "done"}
TERMINAL_FAILURE = {"failed", "error", "cancelled", "canceled"}


# =========================================================
# LOCKS / LOGGING
# =========================================================

print_lock = threading.Lock()
results_lock = threading.Lock()
rate_limit_lock = threading.Lock()


def log(message: str) -> None:
    with print_lock:
        print(message, flush=True)


# =========================================================
# ОБЩИЕ ВСПОМОГАТЕЛЬНЫЕ
# =========================================================

def normalize_base_url(url: str) -> str:
    return url.rstrip("/")


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def save_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def safe_json(resp: requests.Response) -> Dict[str, Any]:
    try:
        return resp.json()
    except Exception:
        return {"raw_text": resp.text}


def validate_aspect_ratio(aspect_ratio: str) -> str:
    """Проверяет aspect_ratio по контракту V6 (n:n). При несоответствии — фолбэк 16:9."""
    if ASPECT_RATIO_RE.match(aspect_ratio):
        return aspect_ratio
    log(f"[WARN] aspect_ratio {aspect_ratio!r} не в формате n:n — использую '16:9'.")
    return "16:9"


def request_with_retries(method: str, url: str, **kwargs) -> requests.Response:
    last_error: Optional[Exception] = None
    attempt = 0            # обычные попытки (сеть/5xx) — ограничены MAX_RETRIES
    cap_attempt = 0        # ёмкостные ("нет аккаунтов") — отдельный, большой бюджет
    cap_delay = float(CAPACITY_SLEEP_START)

    while attempt < MAX_RETRIES:
        try:
            resp = requests.request(method, url, timeout=REQUEST_TIMEOUT, **kwargs)

            if resp.status_code == 429:
                body = safe_json(resp)
                sleep_for = RETRY_SLEEP_429[min(attempt, len(RETRY_SLEEP_429) - 1)]
                log(f"[WARN] HTTP 429 for {url}: {body}")
                attempt += 1
                if attempt < MAX_RETRIES:
                    time.sleep(sleep_for)
                    continue
                resp.raise_for_status()

            if 500 <= resp.status_code < 600:
                body = safe_json(resp)
                raise RuntimeError(f"HTTP {resp.status_code} for {url}: {body}")

            if 400 <= resp.status_code < 500:
                body = safe_json(resp)
                # "Нет доступных аккаунтов" — не контент, а занятый пул: ждём терпеливо,
                # НЕ тратя обычный бюджет попыток.
                if is_capacity_error(json.dumps(body, ensure_ascii=False)):
                    cap_attempt += 1
                    if cap_attempt <= CAPACITY_RETRIES:
                        log(f"[CAPACITY] нет свободных аккаунтов провайдера "
                            f"({cap_attempt}/{CAPACITY_RETRIES}) — жду {int(cap_delay)}с и повторю")
                        time.sleep(cap_delay)
                        cap_delay = min(cap_delay * 1.5, CAPACITY_SLEEP_CAP)
                        continue
                raise RuntimeError(f"HTTP {resp.status_code} for {url}: {body}")

            return resp

        except Exception as e:
            last_error = e
            # Ёмкостные ошибки, всплывшие как исключение, тоже ждём терпеливо.
            if is_capacity_error(str(e)) and cap_attempt < CAPACITY_RETRIES:
                cap_attempt += 1
                log(f"[CAPACITY] нет свободных аккаунтов провайдера "
                    f"({cap_attempt}/{CAPACITY_RETRIES}) — жду {int(cap_delay)}с и повторю")
                time.sleep(cap_delay)
                cap_delay = min(cap_delay * 1.5, CAPACITY_SLEEP_CAP)
                continue
            attempt += 1
            if attempt < MAX_RETRIES:
                log(f"[WARN] Попытка {attempt}/{MAX_RETRIES} не удалась: {e}")
                time.sleep(2 + attempt * 3)
                continue
            raise

    assert last_error is not None
    raise last_error


@dataclass
class PromptItem:
    index: int
    prompt: str


@dataclass
class SceneItem:
    scene_index: int
    image_path: Path
    prompt: str


def parse_prompts_file(path: Path) -> List[PromptItem]:
    lines = path.read_text(encoding="utf-8").splitlines()
    prompts: List[PromptItem] = []
    idx = 1

    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if line.startswith("###"):
            continue
        prompts.append(PromptItem(index=idx, prompt=line))
        idx += 1

    return prompts


def find_prompts_file_optional() -> Optional[Path]:
    if PROMPTS_FILE_PRIMARY.exists():
        return PROMPTS_FILE_PRIMARY
    if PROMPTS_FILE_FALLBACK.exists():
        return PROMPTS_FILE_FALLBACK
    return None


def load_prompts_map_optional() -> Dict[int, str]:
    prompts_file = find_prompts_file_optional()
    if not prompts_file:
        return {}

    prompts = parse_prompts_file(prompts_file)
    prompts_map = {item.index: item.prompt for item in prompts}
    log(f"[INFO] Файл промптов: {prompts_file.name} ({len(prompts_map)} шт.)")
    return prompts_map


def infer_mime_type(path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    return mime or "application/octet-stream"


def path_sort_key(path: Path):
    stem = path.stem
    if stem.isdigit():
        return (0, int(stem))
    return (1, stem.lower())


def list_images() -> List[Path]:
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    images = [p for p in IMAGES_DIR.iterdir() if p.is_file() and p.suffix.lower() in exts]
    images.sort(key=path_sort_key)
    return images


def image_to_data_uri(image_path: Path) -> str:
    file_size = image_path.stat().st_size
    if file_size > MAX_IMAGE_BYTES:
        raise RuntimeError(
            f"Файл слишком большой для data URI: {image_path.name} "
            f"({file_size / 1024 / 1024:.2f} MB > 5.00 MB). "
            f"Сожми картинку или уменьши размер."
        )

    mime_type = infer_mime_type(image_path)
    raw = image_path.read_bytes()
    encoded = base64.b64encode(raw).decode("ascii")
    data_uri = f"data:{mime_type};base64,{encoded}"
    if not DATA_URI_RE.match(data_uri):
        raise RuntimeError(f"Собранный data URI не соответствует схеме V6MediaInput: {image_path.name}")
    return data_uri


def extract_generation_id(data: Dict[str, Any]) -> Optional[str]:
    # GenerationAcceptedResponse.id — канонический источник; остальное — фолбэк.
    candidates = [
        data.get("id"),
        data.get("operation_id"),
        data.get("data", {}).get("id") if isinstance(data.get("data"), dict) else None,
        data.get("result", {}).get("id") if isinstance(data.get("result"), dict) else None,
        data.get("operation", {}).get("id") if isinstance(data.get("operation"), dict) else None,
    ]
    for c in candidates:
        if isinstance(c, str) and c.strip():
            return c.strip()
    return None


def extract_video_source(data: Dict[str, Any]) -> str:
    """
    Достаёт видео из GenerationStatusResponse.results[] (GenerationResultItem):
    предпочитаем item с type=='video', источник — inline data URI либо download_url.
    """
    results = data.get("results")
    if isinstance(results, list):
        # Сначала явные видео-результаты, затем любой результат с источником.
        ordered = sorted(
            (it for it in results if isinstance(it, dict)),
            key=lambda it: 0 if it.get("type") == "video" else 1,
        )
        for item in ordered:
            inline = item.get("data")
            if isinstance(inline, str) and inline.startswith("data:"):
                return inline
            download_url = item.get("download_url")
            if isinstance(download_url, str) and download_url:
                return download_url

    raise RuntimeError(
        f"Не удалось найти video source в results[]: {json.dumps(data, ensure_ascii=False)[:2000]}"
    )


def save_video_from_source(source: str, out_path: Path) -> None:
    if source.startswith("data:"):
        _, b64 = source.split(",", 1)
        out_path.write_bytes(base64.b64decode(b64))
    elif source.startswith("http://") or source.startswith("https://"):
        with requests.get(
            source,
            stream=True,
            headers={"X-API-Key": API_KEY},
            timeout=REQUEST_TIMEOUT,
        ) as r:
            r.raise_for_status()
            with out_path.open("wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
    else:
        raise RuntimeError(f"Неподдерживаемый формат video source: {source[:300]}")

    if not out_path.exists() or out_path.stat().st_size == 0:
        raise RuntimeError(f"Видео не сохранилось или пустое: {out_path}")


# =========================================================
# LIMITER 150 ВИДЕО В ЧАС
# =========================================================

class HourlyRateLimiter:
    def __init__(self, path: Path, max_events: int, window_seconds: int):
        self.path = path
        self.max_events = max_events
        self.window_seconds = window_seconds
        self.events = deque()
        self._load()

    def _load(self) -> None:
        data = load_json(self.path, [])
        now = time.time()

        if isinstance(data, list):
            for ts in data:
                try:
                    ts = float(ts)
                    if now - ts < self.window_seconds:
                        self.events.append(ts)
                except Exception:
                    pass

        self._save()

    def _save(self) -> None:
        save_json(self.path, list(self.events))

    def acquire(self, scene_index: int) -> None:
        while True:
            with rate_limit_lock:
                now = time.time()

                while self.events and (now - self.events[0] >= self.window_seconds):
                    self.events.popleft()

                if len(self.events) < self.max_events:
                    self.events.append(now)
                    self._save()
                    log(f"[RATE] {scene_index:04d}: слот получен ({len(self.events)}/{self.max_events} за последний час)")
                    return

                wait_time = self.window_seconds - (now - self.events[0])
                wait_time = max(1, int(wait_time))

            log(f"[RATE-WAIT] {scene_index:04d}: достигнут лимит {self.max_events}/час, жду {wait_time} сек")
            time.sleep(min(wait_time, 60))


video_rate_limiter = HourlyRateLimiter(
    path=RATE_LIMIT_FILE,
    max_events=MAX_VIDEO_STARTS_PER_HOUR,
    window_seconds=RATE_WINDOW_SECONDS,
)


# =========================================================
# V6 VIDEO START (flow / flow_video_from_ingredients)
# =========================================================

def build_base_payload(prompt: str) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "operation": VIDEO_OPERATION,
        "prompt": prompt,
        "aspect_ratio": validate_aspect_ratio(ASPECT_RATIO),
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


def start_video_from_image(prompt: str, image_data_uri: str) -> str:
    url = normalize_base_url(BASE_URL) + V6_GENERATIONS_ENDPOINT
    payload = build_base_payload(prompt)
    payload["inputs"] = [image_data_uri]

    resp = request_with_retries(
        "POST",
        url,
        headers={
            "X-API-Key": API_KEY,
            "Content-Type": "application/json",
        },
        json=payload,
    )

    data = safe_json(resp)
    generation_id = extract_generation_id(data)
    if not generation_id:
        raise RuntimeError(f"Не удалось получить generation id из ответа: {data}")
    return generation_id


def get_status_from_operation_response(data: Dict[str, Any]) -> Optional[str]:
    candidates = [
        data.get("status"),
        data.get("data", {}).get("status") if isinstance(data.get("data"), dict) else None,
        data.get("result", {}).get("status") if isinstance(data.get("result"), dict) else None,
        data.get("operation", {}).get("status") if isinstance(data.get("operation"), dict) else None,
    ]
    for c in candidates:
        if isinstance(c, str) and c.strip():
            return c.strip().lower()
    return None


def describe_failure(data: Dict[str, Any]) -> str:
    """Собирает читаемое сообщение об ошибке из error + translations (ru)."""
    err = data.get("error") or "unknown error"
    translations = data.get("translations") or {}
    if isinstance(translations, dict) and translations:
        tr = translations.get("ru") or next(iter(translations.values()), None)
        if tr:
            return f"{err} | {tr}"
    return str(err)


def poll_operation(generation_id: str, scene_index: int) -> Dict[str, Any]:
    url = normalize_base_url(BASE_URL) + V6_GENERATION_STATUS_ENDPOINT.format(generation_id=generation_id)

    started = time.time()
    last_status = None
    last_log_time = 0.0

    while True:
        elapsed = int(time.time() - started)
        if elapsed > POLL_TIMEOUT:
            raise TimeoutError(
                f"[TIMEOUT] {scene_index:04d}: генерация {generation_id} не завершилась за {POLL_TIMEOUT} сек"
            )

        resp = request_with_retries(
            "GET",
            url,
            headers={"X-API-Key": API_KEY},
        )

        data = safe_json(resp)
        status = get_status_from_operation_response(data) or "unknown"

        now = time.time()
        if status != last_status or (now - last_log_time) >= STATUS_LOG_EVERY:
            log(f"[WAIT] {scene_index:04d}: generation {generation_id}, status={status}, elapsed={elapsed}s")
            last_status = status
            last_log_time = now

        # V6 терминальный успех — succeeded (плюс совместимые синонимы).
        if status in TERMINAL_SUCCESS:
            for w in (data.get("warnings") or []):
                log(f"[WARN] {scene_index:04d}: gen {generation_id}: {w}")
            return data

        if status in TERMINAL_FAILURE:
            raise RuntimeError(
                f"[ERROR] {scene_index:04d}: генерация {generation_id} завершилась со статусом {status}: "
                f"{describe_failure(data)}"
            )

        time.sleep(POLL_INTERVAL)


# =========================================================
# ОСНОВНОЙ PIPELINE
# =========================================================

results_log: List[Dict[str, Any]] = []


def append_result_log(item: Dict[str, Any]) -> None:
    with results_lock:
        results_log.append(item)
        results_log.sort(key=lambda x: (str(x.get("output", "")), x.get("scene_index", 0)))
        save_json(LOG_FILE, results_log)


def build_scene_items() -> List[SceneItem]:
    images = list_images()
    if not images:
        raise RuntimeError(f"В папке нет картинок: {IMAGES_DIR}")

    prompts_map = load_prompts_map_optional()
    scenes: List[SceneItem] = []

    for order, image_path in enumerate(images, start=1):
        scene_index = order
        if image_path.stem.isdigit():
            scene_index = int(image_path.stem)

        prompt = (
            prompts_map.get(scene_index)
            or prompts_map.get(order)
            or DEFAULT_ANIMATION_PROMPT
        )

        scenes.append(
            SceneItem(
                scene_index=scene_index,
                image_path=image_path,
                prompt=prompt,
            )
        )

    return scenes


def process_scene_item(item: SceneItem) -> None:
    out_path = VIDEOS_DIR / f"{item.image_path.stem}.mp4"

    if out_path.exists() and out_path.stat().st_size > 0:
        log(f"[SKIP] Видео {item.image_path.stem}.mp4 уже существует")
        append_result_log({
            "scene_index": item.scene_index,
            "status": "skipped_existing",
            "output": str(out_path),
            "image": str(item.image_path),
        })
        return

    try:
        log(f"[VIDEO] {item.scene_index:04d}: оживляю {item.image_path.name}")
        image_data_uri = image_to_data_uri(item.image_path)

        video_rate_limiter.acquire(item.scene_index)
        generation_id = start_video_from_image(item.prompt, image_data_uri)

        op_result = poll_operation(generation_id, item.scene_index)
        video_source = extract_video_source(op_result)
        save_video_from_source(video_source, out_path)

        log(f"[DONE] {item.scene_index:04d}: сохранено {out_path.name}")

        append_result_log({
            "scene_index": item.scene_index,
            "status": "done",
            "mode": "flow_video_from_ingredients_base64",
            "output": str(out_path),
            "generation_id": generation_id,
            "image": str(item.image_path),
            "prompt": item.prompt,
        })

    except Exception as e:
        log(f"[FAIL] {item.scene_index:04d}: {e}")
        append_result_log({
            "scene_index": item.scene_index,
            "status": "failed",
            "error": str(e),
            "output": str(out_path),
            "image": str(item.image_path),
            "prompt": item.prompt,
        })


def main() -> None:
    if not API_KEY:
        raise RuntimeError('Не найден API ключ. Задай его так: export FAST_GEN_API_KEY="ТВОЙ_КЛЮЧ"')

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

    global results_log
    loaded_log = load_json(LOG_FILE, [])
    results_log = loaded_log if isinstance(loaded_log, list) else []

    scenes = build_scene_items()

    log(f"[INFO] Папка с картинками: {IMAGES_DIR}")
    log(f"[INFO] Папка для видео: {VIDEOS_DIR}")
    log(f"[INFO] Найдено картинок: {len(scenes)}")
    log(f"[INFO] Потоков: {MAX_WORKERS}")
    log(f"[INFO] Лимит: {MAX_VIDEO_STARTS_PER_HOUR} стартов видео в час")
    log(f"[INFO] Aspect ratio: {ASPECT_RATIO}")
    log(f"[INFO] Провайдер/модель: flow / flow-video-fast (flow_video_from_ingredients)")
    extras = []
    if VIDEO_MODEL:
        extras.append(f"model={VIDEO_MODEL}")
    if GENERATION_SEED is not None:
        extras.append(f"seed={GENERATION_SEED}")
    if VIDEO_DURATION_SECONDS is not None:
        extras.append(f"duration_seconds={VIDEO_DURATION_SECONDS}")
    if VIDEO_RESOLUTION:
        extras.append(f"resolution={VIDEO_RESOLUTION}")
    if VIDEO_ULTRA:
        extras.append("ultra=true")
    if extras:
        log(f"[INFO] Доп. параметры: {', '.join(extras)}")
    log(f"[INFO] Режим: local image -> base64 -> POST {V6_GENERATIONS_ENDPOINT} (operation={VIDEO_OPERATION})")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_scene_item, item) for item in scenes]

        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                log(f"[FUTURE-ERROR] {e}")

    log("[DONE] Все задачи завершены")


if __name__ == "__main__":
    main()
