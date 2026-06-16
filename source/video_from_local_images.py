# -*- coding: utf-8 -*-
"""
Оживление локальных картинок в видео через Fast Gen (media_gen_api v4).

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
   (без upload в storage и без file:...) через провайдера Flower (Veo 3.1)
6. Работает в несколько потоков
7. Учитывает лимит стартов видео в час

Соответствует актуальной документации media_gen_api (V5, модель Veo 3.1 light):
- GET  /api/v5/models?media_type=video  -> список моделей (берём id нужной модели)
- POST /api/v5/generations
    body: { "model": str, "prompt": str, "inputs": [data:image...],
            "aspect_ratio": "16:9" | "9:16", "resolution": str? }
    -> { "id": str, "status": "queued", ... }
- GET  /api/v5/generations/{id}
    -> { "status": "queued"|"running"|"succeeded"|"failed",
         "results": [{ "download_path": str?, "data": str?, ... }], "error": str? }

Видео в результате приходит либо как download_path (путь на storage-сервере,
качаем {STORAGE_URL}{download_path}), либо инлайном в поле data (data:video/...).
"""

from __future__ import annotations

import base64
import json
import mimetypes
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

API_KEY = "CPemyQY7NuiZrCjEgvQYu32U7pT6X1c7"
BASE_URL = "https://api.fast-gen.ai"
STORAGE_URL = "https://storage.fast-gen.ai"

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

# Допустимые значения по новой документации: "16:9" (1280x720) или "9:16" (720x1280).
ASPECT_RATIO = "16:9"

# Желаемая модель V5. Можно указать точный id (из GET /api/v5/models) или
# свободную фразу — скрипт сам найдёт подходящую видео-модель по id/названию.
# Здесь нужна "Veo 3.1 light".
MODEL = "veo 3.1 light"

# Опциональное разрешение, если модель его поддерживает ("480p"/"720p"). None — по умолчанию.
RESOLUTION: Optional[str] = None

MAX_VIDEO_STARTS_PER_HOUR = 150
RATE_WINDOW_SECONDS = 3600

RETRY_SLEEP_429 = [20, 35, 60, 90]

# Если отдельного промпта для картинки нет, будет использоваться этот.
DEFAULT_ANIMATION_PROMPT = (
    "Animate this image with gentle realistic motion, subtle cinematic camera movement, "
    "natural details, preserve the original composition, subject, lighting and style."
)

# Инлайн-картинки в V5 ограничены 5 MB.
MAX_IMAGE_BYTES = 5 * 1024 * 1024


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


def auth_headers(extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    headers = {"X-API-Key": API_KEY}
    if extra:
        headers.update(extra)
    return headers


def request_with_retries(method: str, url: str, **kwargs) -> requests.Response:
    last_error: Optional[Exception] = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.request(method, url, timeout=REQUEST_TIMEOUT, **kwargs)

            if resp.status_code == 429:
                body = safe_json(resp)
                sleep_for = RETRY_SLEEP_429[min(attempt - 1, len(RETRY_SLEEP_429) - 1)]
                log(f"[WARN] Попытка {attempt}/{MAX_RETRIES} не удалась: HTTP 429 for {url}: {body}")
                if attempt < MAX_RETRIES:
                    time.sleep(sleep_for)
                    continue
                resp.raise_for_status()

            if 500 <= resp.status_code < 600:
                body = safe_json(resp)
                raise RuntimeError(f"HTTP {resp.status_code} for {url}: {body}")

            if 400 <= resp.status_code < 500:
                body = safe_json(resp)
                raise RuntimeError(f"HTTP {resp.status_code} for {url}: {body}")

            return resp

        except Exception as e:
            last_error = e
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
    return f"data:{mime_type};base64,{encoded}"


# =========================================================
# РАЗБОР ОТВЕТОВ API
# =========================================================

def extract_generation_id(data: Dict[str, Any]) -> Optional[str]:
    """id из GenerationAcceptedResponse (POST /api/v5/generations)."""
    gen_id = data.get("id")
    if isinstance(gen_id, str) and gen_id.strip():
        return gen_id.strip()
    return None


def extract_video_source(data: Dict[str, Any]) -> str:
    """Достаёт источник видео из GenerationStatusResponse.results.

    Каждый элемент results может содержать:
      - download_path: путь на storage-сервере -> качаем {STORAGE_URL}{path}
      - data: инлайн "data:video/...;base64,..."
    """
    results = data.get("results")
    if not isinstance(results, list) or not results:
        raise RuntimeError(
            f"В ответе нет results с видео: {json.dumps(data, ensure_ascii=False)[:2000]}"
        )

    # Сначала ищем видео-элемент, затем любой с источником.
    candidates = [r for r in results if isinstance(r, dict) and r.get("type") == "video"]
    candidates += [r for r in results if isinstance(r, dict) and r not in candidates]

    for item in candidates:
        download_path = item.get("download_path")
        if isinstance(download_path, str) and download_path.strip():
            return download_path.strip()
        data_uri = item.get("data")
        if isinstance(data_uri, str) and data_uri.strip():
            return data_uri.strip()

    raise RuntimeError(
        f"Не удалось найти video source в results: {json.dumps(data, ensure_ascii=False)[:2000]}"
    )


def _download_to(url: str, out_path: Path) -> None:
    with requests.get(url, headers=auth_headers(), stream=True, timeout=REQUEST_TIMEOUT) as r:
        r.raise_for_status()
        with out_path.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def save_video_from_source(source: str, out_path: Path) -> None:
    if source.startswith("data:video/") or source.startswith("data:"):
        _, b64 = source.split(",", 1)
        out_path.write_bytes(base64.b64decode(b64))
        return

    if source.startswith("http://") or source.startswith("https://"):
        _download_to(source, out_path)
        return

    # download_path со storage-сервера (например "/file/<hash>/raw").
    path = source if source.startswith("/") else "/" + source
    _download_to(normalize_base_url(STORAGE_URL) + path, out_path)


# =========================================================
# LIMITER ВИДЕО В ЧАС
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
# V5: ВЫБОР МОДЕЛИ
# =========================================================

def _normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def resolve_model_id(desired: str) -> str:
    """Находит точный V5 id видео-модели по id или названию (фразе).

    Если совпадение неоднозначно или не найдено — печатает список доступных
    видео-моделей и бросает исключение, чтобы можно было задать точный id.
    """
    url = normalize_base_url(BASE_URL) + "/api/v5/models"
    resp = request_with_retries(
        "GET", url, headers=auth_headers(), params={"media_type": "video"}
    )
    data = safe_json(resp)
    models = data if isinstance(data, list) else data.get("data") or []

    available = []
    for m in models:
        if not isinstance(m, dict):
            continue
        mid = m.get("id")
        if isinstance(mid, str):
            available.append((mid, m.get("display_name") or "", bool(m.get("deprecated"))))

    # 1) Точное совпадение по id.
    for mid, _name, _dep in available:
        if mid == desired:
            return mid

    # 2) Фаззи-совпадение по id/названию.
    want = _normalize(desired)
    matches = [
        mid for mid, name, dep in available
        if not dep and (want in _normalize(mid) or want in _normalize(name))
    ]
    if len(matches) == 1:
        return matches[0]

    listing = "\n".join(f"  - id={mid!r}  name={name!r}" for mid, name, _ in available)
    if not matches:
        raise RuntimeError(
            f"Модель '{desired}' не найдена среди видео-моделей. Доступные:\n{listing}\n"
            f"Укажи точный id в константе MODEL."
        )
    raise RuntimeError(
        f"Под '{desired}' подходит несколько моделей: {matches}.\n{listing}\n"
        f"Укажи точный id в константе MODEL."
    )


# =========================================================
# V5: GENERATION START / POLL
# =========================================================

def build_payload(model_id: str, prompt: str, image_data_uri: str) -> Dict[str, Any]:
    # V5: одна картинка передаётся в inputs (data URI), prompt обязателен.
    payload: Dict[str, Any] = {
        "model": model_id,
        "prompt": prompt,
        "inputs": [image_data_uri],
        "aspect_ratio": ASPECT_RATIO,
    }
    if RESOLUTION:
        payload["resolution"] = RESOLUTION
    return payload


def start_generation(model_id: str, prompt: str, image_data_uri: str) -> str:
    url = normalize_base_url(BASE_URL) + "/api/v5/generations"
    payload = build_payload(model_id, prompt, image_data_uri)

    resp = request_with_retries(
        "POST",
        url,
        headers=auth_headers({"Content-Type": "application/json"}),
        json=payload,
    )

    data = safe_json(resp)
    generation_id = extract_generation_id(data)
    if not generation_id:
        raise RuntimeError(f"Не удалось получить id из ответа /api/v5/generations: {data}")
    return generation_id


def poll_operation(generation_id: str, scene_index: int) -> Dict[str, Any]:
    url = normalize_base_url(BASE_URL) + f"/api/v5/generations/{generation_id}"

    started = time.time()
    last_status = None
    last_log_time = 0.0

    while True:
        elapsed = int(time.time() - started)
        if elapsed > POLL_TIMEOUT:
            raise TimeoutError(
                f"[TIMEOUT] {scene_index:04d}: генерация {generation_id} не завершилась за {POLL_TIMEOUT} сек"
            )

        resp = request_with_retries("GET", url, headers=auth_headers())

        data = safe_json(resp)
        status = str(data.get("status") or "unknown").strip().lower()

        now = time.time()
        if status != last_status or (now - last_log_time) >= STATUS_LOG_EVERY:
            log(f"[WAIT] {scene_index:04d}: generation {generation_id}, status={status}, elapsed={elapsed}s")
            last_status = status
            last_log_time = now

        if status == "succeeded":
            return data

        if status == "failed":
            err = data.get("error") or data
            raise RuntimeError(
                f"[ERROR] {scene_index:04d}: генерация {generation_id} завершилась с ошибкой: {err}"
            )

        # queued / running / unknown -> ждём дальше
        time.sleep(POLL_INTERVAL)


# =========================================================
# ОСНОВНОЙ PIPELINE
# =========================================================

results_log: List[Dict[str, Any]] = []

# Реальный id модели V5, определяется в main() через resolve_model_id().
RESOLVED_MODEL_ID: str = ""


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
        generation_id = start_generation(RESOLVED_MODEL_ID, item.prompt, image_data_uri)

        op_result = poll_operation(generation_id, item.scene_index)
        video_source = extract_video_source(op_result)
        save_video_from_source(video_source, out_path)

        log(f"[DONE] {item.scene_index:04d}: сохранено {out_path.name}")

        append_result_log({
            "scene_index": item.scene_index,
            "status": "done",
            "mode": "image_to_video_v5",
            "model": RESOLVED_MODEL_ID,
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
    if not API_KEY or API_KEY == "ВСТАВЬ_СЮДА_СВОЙ_API_KEY":
        raise RuntimeError("Вставь свой API_KEY в начале файла")

    if ASPECT_RATIO not in {"16:9", "9:16"}:
        raise RuntimeError('ASPECT_RATIO должен быть "16:9" или "9:16"')

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

    global results_log, RESOLVED_MODEL_ID
    loaded_log = load_json(LOG_FILE, [])
    results_log = loaded_log if isinstance(loaded_log, list) else []

    RESOLVED_MODEL_ID = resolve_model_id(MODEL)

    scenes = build_scene_items()

    log(f"[INFO] Папка с картинками: {IMAGES_DIR}")
    log(f"[INFO] Папка для видео: {VIDEOS_DIR}")
    log(f"[INFO] Найдено картинок: {len(scenes)}")
    log(f"[INFO] Потоков: {MAX_WORKERS}")
    log(f"[INFO] Лимит: {MAX_VIDEO_STARTS_PER_HOUR} стартов видео в час")
    log(f"[INFO] Aspect ratio: {ASPECT_RATIO}")
    log(f"[INFO] Модель: {MODEL!r} -> {RESOLVED_MODEL_ID!r}")
    log("[INFO] Режим: local image -> base64 -> /api/v5/generations (Veo 3.1 light)")

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
