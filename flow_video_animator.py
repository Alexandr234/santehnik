#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI-аниматор кадров (image -> video) через Fast-Gen.

Что делает:
  1. Читает план анимации PROMPTS/motion_plan_<lang>.json (его пишет
     psych_prompt_pipeline_v3_multistyle.py). В плане у каждого кадра есть флаг
     "animate" и текст "motion_prompt" (что должно двигаться).
  2. Для КАЖДОГО кадра с animate=true берёт готовую картинку ВИЗУАЛ/<LANG>/NNN.png,
     отправляет её в Fast-Gen image-to-video вместе с motion_prompt, дожидается
     результата и сохраняет ВИЗУАЛ/<LANG>/NNN.mp4.
  3. Исходный PNG перемещает в ВИЗУАЛ/<LANG>/_stills/NNN.png, чтобы монтажёр
     (video_creator) видел ровно ОДИН визуал на индекс — либо png, либо mp4.

Так получается раскладка, которую ты просил:
  - кадры без animate (50%)      -> остаются PNG        -> монтажёр вешает движение камеры
  - кадры animate + camera (25%) -> становятся MP4      -> монтажёр вешает движение камеры поверх
  - кадры animate без camera(25%)-> становятся MP4      -> монтажёр отдаёт как есть

Контракт Fast-Gen (по их OpenAPI, /api/v6/generations):
   operation        = flow_video_from_keyframes   (flow/flow-video, 1 кредит)
   prompt           = motion_prompt (что должно двигаться)
   inputs           = [ data URI картинки ]     (входная картинка = стартовый кадр keyframes)
   keyframes        = true                        (flow: inputs[0] трактуется как стартовый кадр)
   duration_seconds = 1..60
   aspect_ratio     = 16:9
   resolution       = 480p/720p (опционально)
   Переопределить операцию при желании:
       export FAST_GEN_VIDEO_OPERATION="flow_video_lite_from_keyframes"   # lite
       export FAST_GEN_VIDEO_OPERATION="flow_ultra_video_from_keyframes"  # ultra (+ FAST_GEN_VIDEO_ULTRA=1)
       export FAST_GEN_VIDEO_OPERATION="flower_video_from_image"          # вернуться на flower
   Флаги flow-video:
       export FAST_GEN_VIDEO_KEYFRAMES=1   # inputs[0] — стартовый кадр (по умолч. включено)
       export FAST_GEN_VIDEO_ULTRA=0       # ultra-тариф (1080p/ultra-лимиты) для *_ultra_* операций

Ключ — только из окружения:
   export FAST_GEN_API_KEY="veo_..."

Запуск:
   python3 flow_video_animator.py                 # все языки DE/PL/RU
   python3 flow_video_animator.py --lang RU
   python3 flow_video_animator.py --workers 3
   python3 flow_video_animator.py --duration 5
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import random
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from threading import Event, Lock
from typing import Any, Optional

import requests

# =========================
# CONFIG
# =========================
API_KEY = (os.getenv("FAST_GEN_API_KEY", "").strip() or os.getenv("FASTGEN_API_KEY", "").strip())
BASE_URL = os.getenv("FAST_GEN_API_BASE", "https://api.fast-gen.ai")

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT",
    "/Users/aleksandrtomilov/Desktop/ПСИХОЛОГИЯ ГЕРМАНИЯ ПОЛЬША"))
PROMPTS_DIR = PROJECT_ROOT / "ПРОМПТЫ"
VISUAL_ROOT = PROJECT_ROOT / "ВИЗУАЛ"

LOCALES = ["DE", "PL", "RU"]

V6_GENERATIONS_ENDPOINT = "/api/v6/generations"
V6_GENERATION_STATUS_ENDPOINT = "/api/v6/generations/{generation_id}"

# --- параметры видео-операции Fast-Gen (image -> video), контракт по их OpenAPI ---
# operation: flow_video_from_keyframes (flow/flow-video, 1 кредит)
# входная картинка -> в массив inputs[] (data URI) как стартовый кадр keyframes,
# длительность -> duration_seconds.
VIDEO_OPERATION = os.getenv("FAST_GEN_VIDEO_OPERATION", "flow_video_from_keyframes")
VIDEO_ASPECT_RATIO = os.getenv("FAST_GEN_VIDEO_ASPECT_RATIO", "16:9")
VIDEO_DURATION_SEC = int(os.getenv("FAST_GEN_VIDEO_DURATION", "5"))
VIDEO_RESOLUTION = os.getenv("FAST_GEN_VIDEO_RESOLUTION", "").strip()  # напр. 480p/720p; пусто = дефолт модели
# flow-video: keyframes -> inputs[0] это стартовый кадр (image->video). ultra -> ultra-тариф.
VIDEO_KEYFRAMES = os.getenv("FAST_GEN_VIDEO_KEYFRAMES", "1").strip().lower() in {"1", "true", "yes", "on"}
VIDEO_ULTRA = os.getenv("FAST_GEN_VIDEO_ULTRA", "0").strip().lower() in {"1", "true", "yes", "on"}

V6_CAPABILITY_ENDPOINT = "/api/v6/capabilities/{operation_id}"
# Какие опции реально поддерживает выбранная операция. Заполняется при старте из API.
# None = ещё не спрашивали (шлём минимальный payload: operation+prompt+inputs).
SUPPORTED_OPTIONS: Optional[set] = None

REQUEST_TIMEOUT = int(os.getenv("FAST_GEN_REQUEST_TIMEOUT", "600"))
OPERATION_POLL_SEC = int(os.getenv("FAST_GEN_OPERATION_POLL_SEC", "8"))
OPERATION_TIMEOUT_SEC = int(os.getenv("FAST_GEN_OPERATION_TIMEOUT_SEC", "2400"))
RETRY_DELAY_SEC = int(os.getenv("FAST_GEN_RETRY_DELAY_SEC", "8"))
GEN_FAIL_MAX_RETRIES = int(os.getenv("FAST_GEN_GEN_FAIL_MAX_RETRIES", "3"))

DEFAULT_WORKERS = int(os.getenv("FAST_GEN_VIDEO_WORKERS", "10"))
SKIP_EXISTING = True
STILLS_SUBDIR = "_stills"

FATAL_STATUSES = {400, 401, 403, 404, 422}
STOP_EVENT = Event()


class FatalApiError(RuntimeError):
    pass


@dataclass(frozen=True)
class AnimJob:
    locale: str
    index: int
    png_path: Path
    out_path: Path
    motion_prompt: str


# =========================
# HELPERS
# =========================
def log(msg: str) -> None:
    print(msg, flush=True)


def headers(json_content: bool = True) -> dict:
    h = {"X-API-Key": API_KEY}
    if json_content:
        h["Content-Type"] = "application/json"
    return h


def pretty(data: Any) -> str:
    try:
        return json.dumps(data, ensure_ascii=False, indent=2)
    except Exception:
        return str(data)


def file_ok(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size > 0


def image_to_data_uri(path: Path) -> str:
    raw = path.read_bytes()
    b64 = base64.b64encode(raw).decode("ascii")
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return f"data:{mime};base64,{b64}"


def post_json(endpoint: str, payload: dict, *, label: str) -> dict:
    url = BASE_URL.rstrip("/") + endpoint
    attempt = 0
    while True:
        if STOP_EVENT.is_set():
            raise FatalApiError(f"{label}: остановлено")
        attempt += 1
        try:
            resp = requests.post(url, headers=headers(), json=payload, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 429:
                time.sleep(random.uniform(2, 7)); continue
            if resp.status_code in FATAL_STATUSES:
                raise FatalApiError(f"{label}: HTTP {resp.status_code}: {resp.text}\nURL: {url}\n"
                                    f"PAYLOAD keys: {list(payload.keys())}\n"
                                    f"(проверь FAST_GEN_VIDEO_OPERATION — сейчас {VIDEO_OPERATION})")
            if resp.status_code >= 400:
                raise RuntimeError(f"{label}: HTTP {resp.status_code}: {resp.text}")
            data = resp.json()
            if isinstance(data, dict) and data.get("success") is False:
                raise RuntimeError(f"{label}: success=false: {data.get('error') or pretty(data)}")
            return data
        except (KeyboardInterrupt, FatalApiError):
            raise
        except Exception as e:
            if attempt >= 4:
                raise
            log(f"[WARN] {label}: attempt {attempt}: {e}, retry...")
            time.sleep(RETRY_DELAY_SEC)


def get_json(endpoint: str, *, label: str) -> dict:
    url = BASE_URL.rstrip("/") + endpoint
    attempt = 0
    while True:
        if STOP_EVENT.is_set():
            raise FatalApiError(f"{label}: остановлено")
        attempt += 1
        try:
            resp = requests.get(url, headers=headers(json_content=False), timeout=REQUEST_TIMEOUT)
            if resp.status_code == 429:
                time.sleep(random.uniform(2, 7)); continue
            if resp.status_code in FATAL_STATUSES:
                raise FatalApiError(f"{label}: HTTP {resp.status_code}: {resp.text}")
            if resp.status_code >= 400:
                raise RuntimeError(f"{label}: HTTP {resp.status_code}: {resp.text}")
            return resp.json()
        except (KeyboardInterrupt, FatalApiError):
            raise
        except Exception as e:
            if attempt >= 4:
                raise
            log(f"[WARN] {label}: attempt {attempt}: {e}, retry...")
            time.sleep(RETRY_DELAY_SEC)


def download_file(url: str, path: Path, *, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    attempt = 0
    while True:
        attempt += 1
        try:
            resp = requests.get(url, headers=headers(json_content=False), timeout=REQUEST_TIMEOUT)
            if resp.status_code in {401, 403}:
                resp = requests.get(url, timeout=REQUEST_TIMEOUT)  # публичная ссылка без ключа
            if resp.status_code >= 400:
                raise RuntimeError(f"{label}: HTTP {resp.status_code}\nURL: {url}")
            path.write_bytes(resp.content)
            if not file_ok(path):
                raise RuntimeError(f"{label}: пустой файл: {path}")
            return
        except Exception as e:
            if attempt >= 4:
                raise
            log(f"[WARN] {label}: download attempt {attempt}: {e}, retry...")
            time.sleep(RETRY_DELAY_SEC)


# =========================
# FAST-GEN VIDEO
# =========================
def discover_supported_options() -> Optional[set]:
    """Спрашиваем у API, какие опции принимает операция. Возвращаем множество имён
    опций или None, если не удалось (тогда шлём минимальный payload)."""
    try:
        endpoint = V6_CAPABILITY_ENDPOINT.format(operation_id=VIDEO_OPERATION)
        data = get_json(endpoint, label=f"CAPABILITIES {VIDEO_OPERATION}")
        opts = data.get("options")
        if isinstance(opts, dict):
            return set(opts.keys())
        if isinstance(opts, list):
            names = {o.get("name") for o in opts if isinstance(o, dict) and o.get("name")}
            return names or set()
    except Exception as e:
        log(f"[WARN] не смог получить список опций операции: {e} — шлю минимальный payload")
    return None


def build_video_payload(png_path: Path, motion_prompt: str) -> dict:
    # Минимальный, всегда валидный payload.
    payload: dict = {
        "operation": VIDEO_OPERATION,
        "prompt": motion_prompt or "subtle cinematic motion, gentle parallax, floating particles",
        "inputs": [image_to_data_uri(png_path)],   # входная картинка как data URI (стартовый кадр)
    }
    # flow-video: keyframes/ultra — это поля запроса верхнего уровня, а не "options" операции,
    # поэтому шлём их напрямую по env-флагам (для flower их можно выключить FAST_GEN_VIDEO_KEYFRAMES=0).
    if VIDEO_KEYFRAMES:
        payload["keyframes"] = True
    if VIDEO_ULTRA:
        payload["ultra"] = True
    # Доп. опции добавляем ТОЛЬКО если операция их реально поддерживает.
    opts = SUPPORTED_OPTIONS
    if opts:
        if "aspect_ratio" in opts and VIDEO_ASPECT_RATIO:
            payload["aspect_ratio"] = VIDEO_ASPECT_RATIO
        if "duration_seconds" in opts:
            payload["duration_seconds"] = VIDEO_DURATION_SEC
        if "resolution" in opts and VIDEO_RESOLUTION:
            payload["resolution"] = VIDEO_RESOLUTION
    return payload


def save_video_result(results: Any, out_path: Path) -> None:
    if not isinstance(results, list) or not results:
        raise RuntimeError(f"video result пустой/неожиданный: {pretty(results)}")
    item = None
    for r in results:
        if isinstance(r, dict) and (r.get("type") == "video" or r.get("download_url") or r.get("data")):
            item = r; break
    if not isinstance(item, dict):
        raise RuntimeError(f"не нашёл video-result: {pretty(results)}")
    url = item.get("download_url")
    data_uri = item.get("data")
    if isinstance(url, str) and url:
        download_file(url, out_path, label="VIDEO download_url")
        return
    if isinstance(data_uri, str) and data_uri.startswith("data:") and "," in data_uri:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(base64.b64decode(data_uri.split(",", 1)[1]))
        return
    raise RuntimeError(f"неизвестный формат video-result: {pretty(item)}")


def submit_video(png_path: Path, motion_prompt: str) -> str:
    data = post_json(V6_GENERATIONS_ENDPOINT, build_video_payload(png_path, motion_prompt),
                     label=f"VIDEO {VIDEO_OPERATION}")
    gid = data.get("id")
    if not gid:
        raise RuntimeError(f"API не вернул id генерации: {pretty(data)}")
    return str(gid)


def poll_video(generation_id: str) -> dict:
    endpoint = V6_GENERATION_STATUS_ENDPOINT.format(generation_id=generation_id)
    started = time.time()
    while True:
        data = get_json(endpoint, label=f"VIDEO GEN {generation_id}")
        state = data.get("status")
        if state in {"queued", "running", "processing", "starting"}:
            if time.time() - started > OPERATION_TIMEOUT_SEC:
                raise RuntimeError(f"Timeout {generation_id}")
            time.sleep(OPERATION_POLL_SEC)
            continue
        if state == "succeeded":
            if not data.get("results"):
                raise RuntimeError(f"gen {generation_id}: succeeded, но results пустой")
            return data
        raise RuntimeError(f"gen {generation_id}: {state}: {data.get('error') or pretty(data)}")


def animate_one(job: AnimJob) -> dict:
    attempt = 0
    while True:
        if STOP_EVENT.is_set():
            raise FatalApiError("остановлено")
        attempt += 1
        try:
            log(f"[{job.locale}] анимирую #{job.index:03d} (attempt {attempt}) — {job.motion_prompt[:60]}")
            gid = submit_video(job.png_path, job.motion_prompt)
            op_data = poll_video(gid)
            save_video_result(op_data.get("results"), job.out_path)
            # переносим исходный PNG в _stills, чтобы монтажёр видел один визуал
            stills_dir = job.png_path.parent / STILLS_SUBDIR
            stills_dir.mkdir(parents=True, exist_ok=True)
            try:
                shutil.move(str(job.png_path), str(stills_dir / job.png_path.name))
            except Exception as e:
                log(f"    [WARN] не смог перенести PNG в _stills: {e}")
            log(f"    ✅ сохранено видео: {job.out_path.name}")
            return {"index": job.index, "locale": job.locale, "status": "success", "output": str(job.out_path)}
        except (KeyboardInterrupt, FatalApiError):
            raise
        except Exception as e:
            if attempt >= GEN_FAIL_MAX_RETRIES:
                log(f"[GIVE-UP] [{job.locale}] #{job.index}: {e}")
                return {"index": job.index, "locale": job.locale, "status": "error", "error": str(e)}
            delay = min(60.0, RETRY_DELAY_SEC * (1.5 ** (attempt - 1)))
            log(f"[ERROR] [{job.locale}] #{job.index}: {e}, retry in {delay:.0f}s")
            time.sleep(delay)


# =========================
# PLAN / JOBS
# =========================
def load_plan(locale: str) -> list[dict]:
    lc = locale.lower()
    plan_path = PROMPTS_DIR / f"motion_plan_{lc}.json"
    if not plan_path.exists():
        log(f"[WARN] {locale}: не найден план {plan_path} — пропускаю (запусти сначала pipeline).")
        return []
    data = json.loads(plan_path.read_text(encoding="utf-8"))
    return data.get("items", [])


def build_jobs(locale: str) -> list[AnimJob]:
    items = load_plan(locale)
    visual_dir = VISUAL_ROOT / locale
    jobs: list[AnimJob] = []
    for it in items:
        if not it.get("animate"):
            continue
        idx = int(it["index"])
        png = visual_dir / f"{idx:03d}.png"
        out = visual_dir / f"{idx:03d}.mp4"
        if SKIP_EXISTING and file_ok(out):
            continue
        if not file_ok(png):
            # уже перенесён в _stills или ещё не сгенерирован
            alt = visual_dir / STILLS_SUBDIR / f"{idx:03d}.png"
            if file_ok(out):
                continue
            if file_ok(alt) and not file_ok(out):
                png = alt  # повторная попытка из уже перенесённого стилла
            else:
                log(f"[WARN] {locale} #{idx:03d}: нет картинки {png.name} — пропуск")
                continue
        jobs.append(AnimJob(locale=locale, index=idx, png_path=png, out_path=out,
                            motion_prompt=it.get("motion_prompt", "")))
    return jobs


def main() -> None:
    global VIDEO_DURATION_SEC, VIDEO_OPERATION, SKIP_EXISTING, SUPPORTED_OPTIONS
    parser = argparse.ArgumentParser(description="AI-оживление кадров (image->video) по motion_plan")
    parser.add_argument("--lang", nargs="+", choices=LOCALES, default=None)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--duration", type=int, default=VIDEO_DURATION_SEC)
    parser.add_argument("--operation", default=VIDEO_OPERATION, help="имя image->video операции Fast-Gen")
    parser.add_argument("--no-skip", action="store_true")
    args = parser.parse_args()

    VIDEO_DURATION_SEC = args.duration
    VIDEO_OPERATION = args.operation
    SKIP_EXISTING = not args.no_skip

    if not API_KEY:
        print("[ERROR] Не задан ключ. export FAST_GEN_API_KEY='veo_...'", file=sys.stderr)
        sys.exit(1)

    langs = args.lang or LOCALES
    SUPPORTED_OPTIONS = discover_supported_options()
    if SUPPORTED_OPTIONS is None:
        log(f"[VIDEO] operation={VIDEO_OPERATION} | payload=минимальный (operation+prompt+inputs)")
    else:
        log(f"[VIDEO] operation={VIDEO_OPERATION} | поддерживаемые опции: {sorted(SUPPORTED_OPTIONS) or '—'}")

    all_jobs: list[AnimJob] = []
    for lc in langs:
        js = build_jobs(lc)
        log(f"  {lc}: к оживлению {len(js)} кадров")
        all_jobs.extend(js)

    if not all_jobs:
        log("[OK] нечего оживлять (нет кадров с animate=true или все уже .mp4).")
        return

    workers = max(1, min(args.workers, len(all_jobs)))
    log(f"\n=== Оживление: {len(all_jobs)} кадров | workers={workers} ===")
    log_lock = Lock()
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(animate_one, j): j for j in all_jobs}
        try:
            for fut in as_completed(futures):
                res = fut.result()
                with log_lock:
                    done += 1 if res.get("status") == "success" else 0
        except (FatalApiError, KeyboardInterrupt):
            STOP_EVENT.set()
            raise

    log(f"\n=== ГОТОВО: успешно оживлено {done}/{len(all_jobs)} ===")


if __name__ == "__main__":
    try:
        main()
    except FatalApiError as e:
        print(f"\nFATAL: {e}", file=sys.stderr)
        sys.exit(2)
    except KeyboardInterrupt:
        print("\nОстановлено пользователем.")
        sys.exit(1)
