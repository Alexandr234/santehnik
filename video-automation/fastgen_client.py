#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Клиент api.fast-gen.ai (V6).

Вынесен из вашего рабочего скрипта flower_veo31_visual_batch_generator_realistic.py
без изменения логики: те же эндпоинты, те же операции, тот же приём
«стартовый кадр inline base64 в inputs[]», та же политика ретраев
(постоянные ошибки не повторяем, временные — ограниченно).
"""

from __future__ import annotations

import base64
import json
import mimetypes
import time
from pathlib import Path
from typing import Any

import requests

import config

V6_GENERATIONS_ENDPOINT = "/api/v6/generations"
V6_GENERATION_STATUS_ENDPOINT = "/api/v6/generations/{generation_id}"

# Маркеры ПОСТОЯННОЙ ошибки провайдера — такие повторять бессмысленно
PERMANENT_ERROR_MARKERS = (
    "safety", "blocked", "moderation", "content policy",
    "prohibited", "not allowed", "violat",
)


class PermanentError(Exception):
    """Ошибка, которую повторять бессмысленно (safety-фильтр, 4xx)."""


class TransientError(Exception):
    """Временная ошибка (сеть, 429, 5xx) — можно повторить."""


def log(message: str) -> None:
    print(message, flush=True)


def _is_permanent(message: str) -> bool:
    low = message.lower()
    return any(marker in low for marker in PERMANENT_ERROR_MARKERS)


def _headers(json_content: bool = True) -> dict[str, str]:
    h = {"X-API-Key": config.FAST_GEN_API_KEY}
    if json_content:
        h["Content-Type"] = "application/json"
    return h


def _pretty(data: Any) -> str:
    try:
        return json.dumps(data, ensure_ascii=False, indent=2)
    except Exception:
        return str(data)


def image_to_data_uri(image_path: Path) -> str:
    """Кодирует картинку в data:image/...;base64 для inputs[] (лимит 5 MB)."""
    size = image_path.stat().st_size
    if size > config.MAX_IMAGE_BYTES:
        raise PermanentError(
            f"Файл слишком большой для inline data URI: {image_path.name} "
            f"({size / 1024 / 1024:.2f} MB > 5.00 MB)."
        )
    mime, _ = mimetypes.guess_type(str(image_path))
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime or 'image/png'};base64,{encoded}"


def _post(url: str, payload: dict, *, label: str) -> dict:
    last_error: Exception | None = None
    for attempt in range(1, config.MAX_ATTEMPTS + 1):
        try:
            resp = requests.post(url, headers=_headers(), json=payload, timeout=config.REQUEST_TIMEOUT)
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
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            log(f"[WARN] {label}: попытка {attempt}/{config.MAX_ATTEMPTS} не удалась: {exc}")
            if attempt < config.MAX_ATTEMPTS:
                time.sleep(config.RETRY_DELAY_SEC)
    raise TransientError(f"{label}: не удалось за {config.MAX_ATTEMPTS} попыток: {last_error}")


def _get(url: str, *, label: str) -> dict:
    last_error: Exception | None = None
    for attempt in range(1, config.MAX_ATTEMPTS + 1):
        try:
            resp = requests.get(
                url, headers={"X-API-Key": config.FAST_GEN_API_KEY}, timeout=config.REQUEST_TIMEOUT
            )
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
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            log(f"[WARN] {label}: GET попытка {attempt}/{config.MAX_ATTEMPTS} не удалась: {exc}")
            if attempt < config.MAX_ATTEMPTS:
                time.sleep(config.RETRY_DELAY_SEC)
    raise TransientError(f"{label}: не удалось за {config.MAX_ATTEMPTS} попыток: {last_error}")


def create_generation(payload: dict, *, label: str) -> str:
    url = config.FAST_GEN_BASE_URL.rstrip("/") + V6_GENERATIONS_ENDPOINT
    data = _post(url, payload, label=label)
    generation_id = data.get("id")
    if not generation_id:
        raise PermanentError(f"{label}: API не вернул generation id: {_pretty(data)}")
    return generation_id


def poll_generation(generation_id: str, *, label: str) -> dict:
    """Ждёт завершения генерации, возвращает первый result item."""
    url = (
        config.FAST_GEN_BASE_URL.rstrip("/")
        + V6_GENERATION_STATUS_ENDPOINT.format(generation_id=generation_id)
    )
    while True:
        data = _get(url, label=f"{label} STATUS {generation_id}")
        status = str(data.get("status") or "").lower()

        if status in ("queued", "running"):
            log(f"    {label.lower()} status: {status}")
            time.sleep(config.OPERATION_POLL_SEC)
            continue

        if status in ("succeeded", "success", "completed", "done"):
            results = data.get("results") or []
            if not results:
                raise TransientError(f"{label} завершилось, но results пустой: {_pretty(data)}")
            return results[0]

        if status in ("failed", "error", "cancelled", "canceled"):
            error = str(data.get("error") or _pretty(data))
            if _is_permanent(error):
                raise PermanentError(f"{label} заблокировано провайдером: {error}")
            raise TransientError(f"{label} закончилось ошибкой: {error}")

        raise TransientError(f"Неизвестный статус {label}: {_pretty(data)}")


def _download(url: str) -> bytes:
    last_error: Exception | None = None
    for attempt in range(1, config.MAX_ATTEMPTS + 1):
        try:
            resp = requests.get(
                url, headers={"X-API-Key": config.FAST_GEN_API_KEY}, timeout=config.REQUEST_TIMEOUT
            )
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
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            log(f"[WARN] download попытка {attempt}/{config.MAX_ATTEMPTS}: {exc}")
            if attempt < config.MAX_ATTEMPTS:
                time.sleep(config.RETRY_DELAY_SEC)
    raise TransientError(f"Не удалось скачать результат: {last_error}")


def save_result(result_item: dict, path: Path) -> None:
    """Сохраняет результат генерации (inline data URI или download_url) в файл."""
    path.parent.mkdir(parents=True, exist_ok=True)

    inline = result_item.get("data")
    if isinstance(inline, str) and inline.startswith("data:"):
        if "," not in inline:
            raise PermanentError("Некорректный data URI: нет запятой")
        raw = base64.b64decode(inline.split(",", 1)[1])
    else:
        download_url = result_item.get("download_url")
        if not isinstance(download_url, str) or not download_url:
            raise PermanentError(f"Result item без data и download_url: {_pretty(result_item)}")
        raw = _download(download_url)

    path.write_bytes(raw)
    if not path.exists() or path.stat().st_size == 0:
        raise TransientError(f"Файл не сохранился или пустой: {path}")


# =============================================================================
# ВЫСОКОУРОВНЕВЫЕ ОПЕРАЦИИ
# =============================================================================

def reference_filename(index: int, path: Path) -> str:
    """Имя, под которым референс виден модели и на которое ссылается промпт."""
    suffix = path.suffix.lower() or ".jpg"
    return f"face_{index}{suffix}"


def generate_image(
    prompt: str,
    out_path: Path,
    reference_images: list[Path] | Path | None = None,
    aspect_ratio: str | None = None,
) -> str:
    """Генерирует изображение с фото-референсами лица.

    Референсы передаются ИМЕНОВАННЫМИ (V6NamedMediaInput): у каждого есть
    filename, и промпт ссылается на них по этому имени — так модель понимает,
    что все они изображают одного и того же человека, и держит лицо.

    Если провайдер не принимает такие inputs, автоматически откатывается
    на обычный список data URI, а затем и вовсе на генерацию без референсов.
    """
    if isinstance(reference_images, Path):
        reference_images = [reference_images]
    references = [p for p in (reference_images or []) if p.exists()]

    base: dict[str, Any] = {
        "operation": config.OP_IMAGE_GENERATE,
        "prompt": prompt,
        "aspect_ratio": aspect_ratio or config.IMAGE_ASPECT_RATIO,
    }
    if config.IMAGE_UPSCALE_2X:
        base["generation_config"] = {"upscale": {"type": "2x"}}

    variants: list[dict[str, Any]] = []
    if references:
        named = [
            {"filename": reference_filename(i, p), "input": image_to_data_uri(p)}
            for i, p in enumerate(references, 1)
        ]
        variants.append({**base, "inputs": named})
        variants.append({**base, "inputs": [image_to_data_uri(p) for p in references]})
    variants.append(base)

    generation_id: str | None = None
    for index, payload in enumerate(variants):
        try:
            generation_id = create_generation(payload, label="IMAGE")
            break
        except PermanentError as exc:
            if index == len(variants) - 1:
                raise
            log(f"[INFO] Формат запроса не принят ({str(exc)[:120]}). Пробую следующий.")

    if generation_id is None:
        raise PermanentError("не удалось создать генерацию изображения")

    result = poll_generation(generation_id, label="IMAGE")
    save_result(result, out_path)
    return generation_id


def animate_image(image_path: Path, prompt: str, out_path: Path) -> str:
    """Оживляет картинку через flow_video_from_ingredients (модель Flow)."""
    payload: dict[str, Any] = {
        "operation": config.OP_VIDEO_FROM_IMAGE,
        "prompt": prompt,
        "inputs": [image_to_data_uri(image_path)],
        "aspect_ratio": config.VIDEO_ASPECT_RATIO,
    }
    if config.VIDEO_MODEL:
        payload["model"] = config.VIDEO_MODEL

    generation_id = create_generation(payload, label="VIDEO")
    log(f"    video generation_id: {generation_id}")
    result = poll_generation(generation_id, label="VIDEO")
    save_result(result, out_path)
    return generation_id
