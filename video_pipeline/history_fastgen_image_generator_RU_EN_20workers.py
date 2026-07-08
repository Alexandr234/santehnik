#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
media_gen_api V6 image generator for ИСТОРИЯ АВТОМАТИЗАЦИЯ.
MULTI-LANG version: RU + EN.

Что делает:
1) Читает промпты по блокам из:
   - /Users/aleksandrtomilov/Desktop/ИСТОРИЯ АВТОМАТИЗАЦИЯ/ПРОМПТЫ/ru_RU_prompts.txt
   - /Users/aleksandrtomilov/Desktop/ИСТОРИЯ АВТОМАТИЗАЦИЯ/ПРОМПТЫ/en_EN_prompts.txt

2) Генерирует визуал отдельно:
   - RU -> /Users/.../ВИЗУАЛ/RU_SCENE_LOCK
   - EN -> /Users/.../ВИЗУАЛ/EN_SCENE_LOCK

3) По умолчанию обрабатывает обе языковые версии, если файлы существуют.

4) Работает быстро:
   - по умолчанию 20 потоков;
   - можно менять через --workers.

5) Не читает "3-я, 7-я, 11-я строка".
   Читает файл надёжно по блокам:
      001 | time --> time | type: ...
      TEXT: ...
      LABELS: ...
      сам промпт

6) LABELS:
   - если LABELS нет -> запрещает любой текст внутри изображения;
   - если LABELS есть -> разрешает только эти подписи.

7) Создаёт manifest.csv, generation_log.json и parsed_prompts.json отдельно в RU/EN папках.

8) Что изменилось относительно V4-версии (новая API-документация media_gen_api V6):
   - старые V4 endpoints (/api/v4/flower/image/generate и /api/v4/operations/...) больше не используются;
   - генерация идёт через единый endpoint POST /api/v6/generations
     с canonical operation id flower_image_generate;
   - статус запрашивается через GET /api/v6/generations/{generation_id};
   - статусы теперь: queued / running / succeeded / failed;
   - результат приходит в results[] как download_url (файл) либо inline data (data URI).

Запуск:
   python3 history_fastgen_image_generator_RU_EN_20workers.py
   python3 history_fastgen_image_generator_RU_EN_20workers.py --lang RU
   python3 history_fastgen_image_generator_RU_EN_20workers.py --lang EN
   python3 history_fastgen_image_generator_RU_EN_20workers.py --workers 20 --no-skip-existing
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
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

import requests


# =========================
# CONFIG
# =========================

API_KEY = ""
ENV_API_KEY = (
    os.getenv("FAST_GEN_API_KEY")
    or os.getenv("FASTGEN_API_KEY")
    or os.getenv("MEDIA_GEN_API_KEY")
    or ""
)
if not API_KEY:
    API_KEY = ENV_API_KEY

BASE_URL = os.getenv("FAST_GEN_API_BASE", "https://api.fast-gen.ai")

BASE_DIR = Path("/Users/aleksandrtomilov/Desktop/ИСТОРИЯ АВТОМАТИЗАЦИЯ")
PROMPTS_DIR = BASE_DIR / "ПРОМПТЫ"
VISUAL_ROOT = BASE_DIR / "ВИЗУАЛ"

PROMPTS_FILE_BY_LANG = {
    "RU": PROMPTS_DIR / "ru_RU_prompts.txt",
    "EN": PROMPTS_DIR / "en_EN_prompts.txt",
}

OUTPUT_SUBDIR_BY_LANG = {
    "RU": os.getenv("FAST_GEN_OUTPUT_SUBDIR_RU", "RU_SCENE_LOCK"),
    "EN": os.getenv("FAST_GEN_OUTPUT_SUBDIR_EN", "EN_SCENE_LOCK"),
}

IMAGE_ASPECT_RATIO = os.getenv("FAST_GEN_IMAGE_ASPECT_RATIO", "16:9")
REQUEST_TIMEOUT = int(os.getenv("FAST_GEN_REQUEST_TIMEOUT", "300"))
OPERATION_POLL_SEC = int(os.getenv("FAST_GEN_OPERATION_POLL_SEC", "10"))
RETRY_DELAY_SEC = int(os.getenv("FAST_GEN_RETRY_DELAY_SEC", "8"))
DEFAULT_MAX_IMAGE_WORKERS = int(os.getenv("FAST_GEN_IMAGE_WORKERS", "20"))

SKIP_EXISTING = True

# V6 media_gen_api endpoints.
V6_GENERATIONS_ENDPOINT = "/api/v6/generations"                       # POST create generation
V6_GENERATION_STATUS_ENDPOINT = "/api/v6/generations/{generation_id}"  # GET status

# Canonical V6 operation id (см. GET /api/v6/capabilities). Переопределяется через env.
OP_IMAGE_GENERATE = os.getenv("FAST_GEN_IMAGE_OPERATION", "flower_image_generate")

TIMECODE_RE = re.compile(
    r"^(?:(\d+)\s*\|\s*)?"
    r"(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*"
    r"(\d{2}:\d{2}:\d{2},\d{3})"
    r"(?:\s*\|.*)?$"
)
TYPE_RE = re.compile(r"\|\s*type\s*:\s*([^|]+)", re.IGNORECASE)
DURATION_RE = re.compile(r"\|\s*duration\s*:\s*([0-9.,]+)s?", re.IGNORECASE)
SCENE_RE = re.compile(r"\|\s*scene\s*:\s*([^|]+)", re.IGNORECASE)
SHOT_RE = re.compile(r"\|\s*shot\s*:\s*([^|]+)", re.IGNORECASE)
CONTINUITY_RE = re.compile(r"\|\s*continuity\s*:\s*([^|]+)", re.IGNORECASE)

STYLE_GUARD_RU = (
    "СТРОГО ЕДИНЫЙ СТИЛЬ: flat 2D vector cartoon в духе вирусного образовательного YouTube-объяснялки "
    "(история/эволюция/наука). "
    "Жирные ровные ЧЁРНЫЕ контуры одинаковой толщины у каждого объекта; уверенные чёткие линии. "
    "Полностью плоская заливка ровными цветами: без градиентов, без мягких теней, без текстуры; "
    "максимум одна плоская тёмная тень-пятно. "
    "Тёплая землистая приглушённая палитра: песочно-бежевый, оранжевое небо, оливково-зелёный, тёплые коричневые, охра. "
    "Человек — ВСЕГДА один и тот же стикмен: белая круглая голова с чёрным контуром, "
    "круглые белые глаза с чёрными точками-зрачками, простые выразительные брови и рот, "
    "тонкие руки и ноги одной чёрной линией, при необходимости лохматые волосы сплошным коричневым пятном; "
    "без проработанного тела и мышц. "
    "Животные и предметы — узнаваемые, с правильными пропорциями плоские мультяшные формы "
    "с таким же жирным чёрным контуром и плоской заливкой (лев выглядит как лев), НИКОГДА не стикмены. "
    "Простой, но цельный плоский фон по смыслу сцены с чёткой линией горизонта. "
    "НЕ пастель, НЕ карандаш, НЕ эскиз, НЕ каракуль, НЕ фотореализм, НЕ фотография, НЕ 3D, НЕ CGI, "
    "НЕ живопись, НЕ аниме, НЕ live-action, НЕ реалистичная анатомия человека. "
)

STYLE_GUARD_EN = (
    "STRICT UNIFIED STYLE: flat 2D vector cartoon in the style of a viral educational YouTube explainer "
    "(history / evolution / science). "
    "Bold, even-weight BLACK outlines of uniform thickness on every shape; confident clean linework. "
    "Fully flat solid color fills: no gradients, no soft shading, no texture; at most one flat darker shadow shape. "
    "Warm earthy muted palette: sandy beige, orange sky, olive green, warm browns, ochre. "
    "The human is ALWAYS the same stick figure: a plain white round head with a black outline, "
    "round white eyes with black dot pupils, simple expressive eyebrows and mouth, "
    "thin single-line black stick arms and legs, optional shaggy solid-brown hair; no detailed body, no muscles. "
    "Animals and objects are recognizable, correctly proportioned flat cartoon shapes "
    "with the same bold black outline and flat fill (a lion looks like a lion), NEVER stick figures. "
    "A simple but complete flat-color background that fits the scene, with a clear horizon line. "
    "NOT pastel, NOT pencil, NOT sketchy, NOT scribble, NOT photorealistic, NOT photography, NOT 3D, NOT CGI, "
    "NOT painterly, NOT anime, NOT live-action, NOT realistic human anatomy. "
)

NO_TEXT_RULE_RU = (
    "ВАЖНО: внутри изображения не должно быть текста, букв, субтитров, логотипов, водяных знаков "
    "и случайных надписей. Только рисунок. "
)

NO_TEXT_RULE_EN = (
    "IMPORTANT: there must be no text, letters, subtitles, logos, watermarks, "
    "or random writing inside the image. Only the illustration. "
)

LABEL_TEXT_RULE_TEMPLATE_RU = (
    "ВАЖНО: внутри изображения разрешены только эти короткие подписи: {labels}. "
    "Никаких других слов, субтитров, логотипов или лишних надписей. "
)

LABEL_TEXT_RULE_TEMPLATE_EN = (
    "IMPORTANT: only these short labels are allowed inside the image: {labels}. "
    "No other words, subtitles, logos, or extra text. "
)


# =========================
# MODELS
# =========================

@dataclass
class PromptItem:
    index: int
    timing_line_number: int
    prompt_line_numbers: List[int]
    start_tc: str
    end_tc: str
    duration: str
    block_type: str
    scene_id: str
    shot_type: str
    continuity_mode: str
    scene_title: str
    scene_lock: str
    voice_text: str
    labels: str
    raw_prompt: str
    final_prompt: str


# =========================
# HELPERS
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
        print("Вариант 1: вставь ключ прямо в скрипт в переменную API_KEY.")
        print('Вариант 2: export FAST_GEN_API_KEY="ТВОЙ_КЛЮЧ"')
        sys.exit(1)


def pretty_json(data: Any) -> str:
    try:
        return json.dumps(data, ensure_ascii=False, indent=2)
    except Exception:
        return str(data)


def safe_timecode(value: str) -> str:
    if not value:
        return "no-time"
    return value.replace(":", "-").replace(",", "-").replace(" ", "")


def output_name(item: PromptItem) -> str:
    if item.start_tc and item.end_tc:
        return f"{item.index:04d}_{safe_timecode(item.start_tc)}__{safe_timecode(item.end_tc)}.png"
    return f"{item.index:04d}.png"


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


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def strip_prompt_prefix(prompt: str) -> str:
    text = prompt.strip()
    text = re.sub(r"^\s*(промпт|prompt|image prompt|картинка|image)\s*[:\-—]\s*", "", text, flags=re.IGNORECASE)
    return text.strip()


def parse_timing_line(line: str) -> Tuple[Optional[int], str, str, str, str, str, str, str]:
    m = TIMECODE_RE.match(line.strip())
    if not m:
        return None, "", "", "", "scene", "", "", ""

    raw_index, start_tc, end_tc = m.groups()
    index = int(raw_index) if raw_index else None

    duration_match = DURATION_RE.search(line)
    duration = duration_match.group(1).replace(",", ".") if duration_match else ""

    type_match = TYPE_RE.search(line)
    block_type = clean_text(type_match.group(1)) if type_match else "scene"

    scene_match = SCENE_RE.search(line)
    scene_id = clean_text(scene_match.group(1)) if scene_match else ""

    shot_match = SHOT_RE.search(line)
    shot_type = clean_text(shot_match.group(1)) if shot_match else ""

    continuity_match = CONTINUITY_RE.search(line)
    continuity_mode = clean_text(continuity_match.group(1)) if continuity_match else ""

    return index, start_tc, end_tc, duration, block_type, scene_id, shot_type, continuity_mode


def looks_like_real_prompt(prompt: str) -> bool:
    p = prompt.lower()
    if len(prompt) < 100:
        return False
    prompt_markers = [
        "сцена:", "2d", "рисован", "иллюстрац", "инфограф",
        "educational", "cartoon", "illustration", "scene:"
    ]
    return any(marker in p for marker in prompt_markers)


def build_fallback_prompt(voice_text: str, labels: str, block_type: str, lang: str) -> str:
    if lang == "EN":
        base = (
            "Flat 2D vector cartoon in the style of a viral educational YouTube explainer, "
            "bold even-weight black outlines, fully flat solid color fills, no gradients, no soft shading, "
            "warm earthy muted palette (sandy beige, orange sky, olive green, warm browns, ochre), "
            "a simple but complete flat-color background with a clear horizon line. "
            "The human is a simple stick figure: plain white round head, black dot pupils, "
            "thin single-line black arms and legs, optional shaggy brown hair. "
            "Animals and objects are recognizable flat cartoon shapes with the same bold black outline, never stick figures. "
            "Not pastel, not pencil, not photorealistic, not 3D, not CGI. "
        )
        if block_type.lower() in {"comparison", "infographic", "scheme", "map", "diagram"}:
            base += "Use a simple explanatory layout: 1-2 flat figures, a black arrow, bold uppercase labels only if truly needed. "
        scene = f"Scene: clearly visualize the narrator's phrase: {voice_text}. "
        scene += f"Allowed labels inside the image (bold uppercase): {labels}. " if labels else "No text inside the image. "
        return base + scene

    base = (
        "плоский 2D векторный мультфильм в стиле вирусной образовательной YouTube-объяснялки, "
        "жирные ровные чёрные контуры, полностью плоская заливка ровными цветами, без градиентов и мягких теней, "
        "тёплая землистая приглушённая палитра (песочно-бежевый, оранжевое небо, оливково-зелёный, тёплые коричневые, охра), "
        "простой, но цельный плоский фон с чёткой линией горизонта. "
        "Человек — простой стикмен: белая круглая голова, чёрные точки-зрачки, "
        "тонкие руки и ноги одной чёрной линией, при необходимости лохматые коричневые волосы. "
        "Животные и предметы — узнаваемые плоские мультяшные формы с таким же жирным чёрным контуром, никогда не стикмены. "
        "Не пастель, не карандаш, без фотореализма, без 3D, без CGI. "
    )
    if block_type.lower() in {"comparison", "infographic", "scheme", "map", "diagram"}:
        base += "Простая объясняющая раскладка: 1-2 плоские фигуры, чёрная стрелка, жирные заглавные подписи только если реально нужно. "
    scene = f"Сцена: ясно визуализировать фразу диктора: {voice_text}. "
    scene += f"Разрешённые подписи внутри изображения (жирные заглавные): {labels}. " if labels else "Без текста внутри изображения. "
    return base + scene


def strengthen_prompt(
    raw_prompt: str,
    labels: str,
    lang: str,
    scene_title: str = "",
    scene_lock: str = "",
    scene_id: str = "",
    shot_type: str = "",
    continuity_mode: str = "",
) -> str:
    prompt = strip_prompt_prefix(clean_text(raw_prompt))

    if lang == "EN":
        text_rule = LABEL_TEXT_RULE_TEMPLATE_EN.format(labels=labels) if labels else NO_TEXT_RULE_EN
        style_guard = STYLE_GUARD_EN
        continuity_rule = ""
        if scene_id or scene_lock or scene_title:
            continuity_rule = (
                f"VISUAL CONTINUITY: scene_id={scene_id or 'unknown'}; "
                f"scene_title={scene_title or 'unknown'}; "
                f"scene_lock={scene_lock or 'not specified'}. "
            )
        if continuity_mode and continuity_mode not in {"new_scene", "infographic_reset"}:
            continuity_rule += (
                "This is a continuation of the same scene: preserve the same location, palette, character, era and narrative situation; "
                "change only camera angle, action, framing or one detail. "
            )
        if shot_type:
            continuity_rule += f"Shot type: {shot_type}. "
        final_prompt = (
            style_guard
            + text_rule
            + continuity_rule
            + "MAIN PROMPT: "
            + prompt
            + " Final reminder: keep the unified flat 2D vector explainer style — bold even black outlines, "
            + "fully flat solid fills (no gradients, no soft shading), warm earthy palette; "
            + "humans only as simple white-headed stick figures, animals and objects as recognizable flat cartoon shapes."
        )
        return clean_text(final_prompt)

    text_rule = LABEL_TEXT_RULE_TEMPLATE_RU.format(labels=labels) if labels else NO_TEXT_RULE_RU
    style_guard = STYLE_GUARD_RU
    continuity_rule = ""
    if scene_id or scene_lock or scene_title:
        continuity_rule = (
            f"ВИЗУАЛЬНАЯ НЕПРЕРЫВНОСТЬ: scene_id={scene_id or 'unknown'}; "
            f"scene_title={scene_title or 'unknown'}; "
            f"scene_lock={scene_lock or 'not specified'}. "
        )
    if continuity_mode and continuity_mode not in {"new_scene", "infographic_reset"}:
        continuity_rule += (
            "Это продолжение той же сцены: сохраняй ту же локацию, палитру, героя, эпоху и смысловую ситуацию; "
            "меняй только ракурс, действие, крупность плана или конкретную деталь. "
        )
    if shot_type:
        continuity_rule += f"Тип кадра: {shot_type}. "
    final_prompt = (
        style_guard
        + text_rule
        + continuity_rule
        + "ОСНОВНОЙ ПРОМПТ: "
        + prompt
        + " Финальное напоминание: сохранить единый плоский 2D векторный explainer-стиль — жирные ровные чёрные контуры, "
        + "полностью плоская заливка (без градиентов, без мягких теней), тёплая землистая палитра; "
        + "люди — только простые стикмены с белой головой, животные и предметы — узнаваемые плоские мультяшные формы."
    )
    return clean_text(final_prompt)


def parse_prompts_by_blocks(path: Path, lang: str) -> List[PromptItem]:
    if not path.exists():
        raise FileNotFoundError(f"Не найден файл промптов: {path}")
    lines = path.read_text(encoding="utf-8").splitlines()
    items: List[PromptItem] = []

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not TIMECODE_RE.match(line):
            i += 1
            continue

        parsed_index, start_tc, end_tc, duration, block_type, scene_id, shot_type, continuity_mode = parse_timing_line(line)
        timing_line_number = i + 1
        block_index = parsed_index if parsed_index is not None else len(items) + 1

        i += 1
        voice_text_lines: List[str] = []
        labels_lines: List[str] = []
        scene_title_lines: List[str] = []
        scene_lock_lines: List[str] = []
        prompt_lines: List[str] = []
        prompt_line_numbers: List[int] = []

        while i < len(lines):
            current = lines[i].strip()
            if TIMECODE_RE.match(current):
                break
            if not current:
                i += 1
                continue
            if current.upper().startswith("TEXT:"):
                voice_text_lines.append(current.split(":", 1)[1].strip())
            elif current.upper().startswith("LABELS:"):
                labels_lines.append(current.split(":", 1)[1].strip())
            elif current.upper().startswith("SCENE_TITLE:"):
                scene_title_lines.append(current.split(":", 1)[1].strip())
            elif current.upper().startswith("SCENE_LOCK:"):
                scene_lock_lines.append(current.split(":", 1)[1].strip())
            else:
                prompt_lines.append(current)
                prompt_line_numbers.append(i + 1)
            i += 1

        voice_text = clean_text(" ".join(voice_text_lines))
        labels = clean_text("; ".join(labels_lines))
        scene_title = clean_text(" ".join(scene_title_lines))
        scene_lock = clean_text(" ".join(scene_lock_lines))
        raw_prompt = clean_text(" ".join(prompt_lines))

        if not looks_like_real_prompt(raw_prompt):
            log(f"[WARN] {lang} block {block_index:03d}: suspicious prompt lines {prompt_line_numbers or 'нет'}. Building fallback.")
            raw_prompt = build_fallback_prompt(voice_text, labels, block_type, lang)

        final_prompt = strengthen_prompt(
            raw_prompt=raw_prompt,
            labels=labels,
            lang=lang,
            scene_title=scene_title,
            scene_lock=scene_lock,
            scene_id=scene_id,
            shot_type=shot_type,
            continuity_mode=continuity_mode,
        )

        items.append(
            PromptItem(
                index=block_index,
                timing_line_number=timing_line_number,
                prompt_line_numbers=prompt_line_numbers,
                start_tc=start_tc,
                end_tc=end_tc,
                duration=duration,
                block_type=block_type,
                scene_id=scene_id,
                shot_type=shot_type,
                continuity_mode=continuity_mode,
                scene_title=scene_title,
                scene_lock=scene_lock,
                voice_text=voice_text,
                labels=labels,
                raw_prompt=raw_prompt,
                final_prompt=final_prompt,
            )
        )

    if not items:
        raise RuntimeError(f"Не нашёл блоки в файле {path}")
    return items


def write_parsed_prompts_debug(items: List[PromptItem], output_dir: Path, lang: str) -> None:
    save_json(output_dir / "parsed_prompts.json", [asdict(item) for item in items])

    preview_path = output_dir / "prompt_parse_preview.txt"
    lines: List[str] = []
    for item in items[:60]:
        lines.append(f"{item.index:03d} | {item.start_tc} --> {item.end_tc} | scene={item.scene_id} | type={item.block_type} | shot={item.shot_type} | continuity={item.continuity_mode}")
        lines.append(f"TEXT: {item.voice_text}")
        if item.scene_title:
            lines.append(f"SCENE_TITLE: {item.scene_title}")
        if item.scene_lock:
            lines.append(f"SCENE_LOCK: {item.scene_lock}")
        if item.labels:
            lines.append(f"LABELS: {item.labels}")
        lines.append(f"PROMPT LINES: {item.prompt_line_numbers}")
        lines.append(f"RAW PROMPT START: {item.raw_prompt[:300]}")
        lines.append("")
    preview_path.write_text("\n".join(lines), encoding="utf-8")


# =========================
# DATA URI / RESULT HELPERS
# =========================

def split_data_uri(data_uri: str) -> Tuple[str, bytes]:
    if not isinstance(data_uri, str) or not data_uri.startswith("data:"):
        raise ValueError(f"Ответ API не похож на data URI: {str(data_uri)[:120]}")
    if "," not in data_uri:
        raise ValueError("Некорректный data URI: нет запятой")
    header, b64 = data_uri.split(",", 1)
    mime = header.replace("data:", "").replace(";base64", "")
    return mime, base64.b64decode(b64)


def write_bytes_checked(raw: bytes, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"Файл не сохранился или пустой: {path}")


def download_url_to_bytes(url: str) -> bytes:
    """Скачивает файл результата по download_url (с ключом на случай приватного стораджа)."""
    attempt = 0
    while True:
        attempt += 1
        try:
            resp = requests.get(url, headers={"X-API-Key": API_KEY}, timeout=REQUEST_TIMEOUT)
            if resp.status_code >= 400:
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
            if not resp.content:
                raise RuntimeError("Пустой ответ при скачивании результата")
            return resp.content
        except KeyboardInterrupt:
            raise
        except Exception as e:
            log(f"[WARN] download result попытка {attempt} не удалась: {e}")
            time.sleep(RETRY_DELAY_SEC)


def save_result_item_to_file(result_item: Dict[str, Any], path: Path) -> None:
    """
    V6 результат приходит как GenerationResultItem:
      - data:        inline data URI (когда провайдер вернул inline)
      - download_url: ссылка на файл в сторадже
    Сохраняем в path, предпочитая inline data, затем download_url.
    """
    inline = result_item.get("data")
    if isinstance(inline, str) and inline.startswith("data:"):
        _mime, raw = split_data_uri(inline)
        write_bytes_checked(raw, path)
        return

    download_url = result_item.get("download_url")
    if isinstance(download_url, str) and download_url:
        raw = download_url_to_bytes(download_url)
        write_bytes_checked(raw, path)
        return

    raise RuntimeError(f"Result item без data и download_url: {pretty_json(result_item)}")


# =========================
# HTTP / V6 API
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
            time.sleep(RETRY_DELAY_SEC)


def start_flower_image(prompt: str) -> str:
    url = clean_base_url(BASE_URL) + V6_GENERATIONS_ENDPOINT
    payload: Dict[str, Any] = {
        "operation": OP_IMAGE_GENERATE,
        "prompt": prompt,
        "aspect_ratio": IMAGE_ASPECT_RATIO,
    }
    data = request_with_retries_post(url, payload, label="FLOWER IMAGE GENERATE")
    generation_id = data.get("id")
    if not generation_id:
        raise RuntimeError(f"Image API не вернул generation id: {pretty_json(data)}")
    return generation_id


def poll_generation(generation_id: str, *, label: str) -> Dict[str, Any]:
    """Ждёт завершения генерации и возвращает первый result item."""
    url = clean_base_url(BASE_URL) + V6_GENERATION_STATUS_ENDPOINT.format(generation_id=generation_id)
    while True:
        data = request_get_with_retries(url, label=f"{label} STATUS {generation_id}")
        status = data.get("status")
        if status in ("queued", "running"):
            time.sleep(OPERATION_POLL_SEC)
            continue
        if status == "succeeded":
            results = data.get("results") or []
            if not results:
                raise RuntimeError(f"{label} завершилось, но results пустой: {pretty_json(data)}")
            return results[0]
        if status == "failed":
            error = data.get("error") or pretty_json(data)
            raise RuntimeError(f"{label} закончилось ошибкой: {error}")
        raise RuntimeError(f"Неизвестный статус {label}: {pretty_json(data)}")


def generate_image_flower(prompt: str) -> Tuple[Dict[str, Any], str]:
    generation_id = start_flower_image(prompt)
    result_item = poll_generation(generation_id, label="IMAGE")
    return result_item, generation_id


# =========================
# GENERATION
# =========================

def write_manifest(items: List[PromptItem], output_dir: Path) -> None:
    manifest_path = output_dir / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "index", "type", "block_type", "scene_id", "shot_type", "continuity_mode",
                "scene_title", "scene_lock", "start_tc", "end_tc", "duration", "filename",
                "timing_line_number", "prompt_line_numbers", "voice_text", "labels",
                "raw_prompt", "final_prompt",
            ],
        )
        writer.writeheader()
        for item in items:
            writer.writerow(
                {
                    "index": item.index,
                    "type": "image",
                    "block_type": item.block_type,
                    "scene_id": item.scene_id,
                    "shot_type": item.shot_type,
                    "continuity_mode": item.continuity_mode,
                    "scene_title": item.scene_title,
                    "scene_lock": item.scene_lock,
                    "start_tc": item.start_tc,
                    "end_tc": item.end_tc,
                    "duration": item.duration,
                    "filename": output_name(item),
                    "timing_line_number": item.timing_line_number,
                    "prompt_line_numbers": ";".join(str(n) for n in item.prompt_line_numbers),
                    "voice_text": item.voice_text,
                    "labels": item.labels,
                    "raw_prompt": item.raw_prompt,
                    "final_prompt": item.final_prompt,
                }
            )


def generate_image_until_done(item: PromptItem, out_path: Path, lang: str) -> dict:
    attempt = 0
    while True:
        attempt += 1
        try:
            log(f"[{lang}] {out_path.name} | block {item.index:03d} | attempt {attempt}")
            result_item, generation_id = generate_image_flower(item.final_prompt)
            save_result_item_to_file(result_item, out_path)
            return {
                "index": item.index,
                "timing_line_number": item.timing_line_number,
                "prompt_line_numbers": item.prompt_line_numbers,
                "type": "image",
                "block_type": item.block_type,
                "output": str(out_path),
                "status": "success",
                "attempts": attempt,
                "generation_id": generation_id,
            }
        except KeyboardInterrupt:
            raise
        except Exception as e:
            log(f"[{lang} IMAGE ERROR] {out_path.name}: {e}")
            time.sleep(RETRY_DELAY_SEC)


def process_items(items: List[PromptItem], output_dir: Path, lang: str, max_workers: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    global_log_file = output_dir / "generation_log.json"
    global_log = load_json(global_log_file, [])
    if not isinstance(global_log, list):
        global_log = []

    log_lock = Lock()
    jobs: List[tuple[PromptItem, Path]] = []

    for item in items:
        filename = output_name(item)
        out_path = output_dir / filename

        if SKIP_EXISTING and out_path.exists() and out_path.stat().st_size > 0:
            log(f"[{lang} SKIP] {filename} уже существует")
            with log_lock:
                global_log.append(
                    {
                        "index": item.index,
                        "timing_line_number": item.timing_line_number,
                        "prompt_line_numbers": item.prompt_line_numbers,
                        "type": "image",
                        "block_type": item.block_type,
                        "output": str(out_path),
                        "status": "skipped_existing",
                    }
                )
                save_json(global_log_file, global_log)
            continue

        jobs.append((item, out_path))

    log(f"[{lang}] Всего блоков/промптов: {len(items)}")
    log(f"[{lang}] Нужно создать картинок: {len(jobs)}")
    log(f"[{lang}] Потоки image_workers={max_workers}")

    if not jobs:
        return

    workers = max(1, min(max_workers, len(jobs)))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_job = {
            executor.submit(generate_image_until_done, item, out_path, lang): (item, out_path)
            for item, out_path in jobs
        }

        for future in as_completed(future_to_job):
            item, _out_path = future_to_job[future]
            result = future.result()
            result["start_tc"] = item.start_tc
            result["end_tc"] = item.end_tc
            result["duration"] = item.duration
            result["voice_text"] = item.voice_text
            result["labels"] = item.labels
            result["raw_prompt"] = item.raw_prompt
            result["final_prompt"] = item.final_prompt
            with log_lock:
                global_log.append(result)
                save_json(global_log_file, global_log)


def process_language(lang: str, prompts_file: Path, output_dir: Path, max_workers: int, dry_run: bool) -> None:
    log("=" * 70)
    log(f"[{lang}] Prompts file: {prompts_file}")
    log(f"[{lang}] Output dir:   {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    items = parse_prompts_by_blocks(prompts_file, lang)
    write_parsed_prompts_debug(items, output_dir, lang)
    write_manifest(items, output_dir)

    if dry_run:
        log(f"[{lang}] DRY RUN DONE — генерация не запускалась.")
        return

    process_items(items, output_dir, lang, max_workers=max_workers)
    log(f"[{lang}] DONE")


def main() -> None:
    global BASE_URL, SKIP_EXISTING

    parser = argparse.ArgumentParser(description="Generate RU/EN images from prompt files with media_gen_api V6.")
    parser.add_argument("--lang", choices=["RU", "EN", "ALL"], default="ALL", help="Что генерировать: RU, EN или ALL")
    parser.add_argument("--prompts-file-ru", default=str(PROMPTS_FILE_BY_LANG["RU"]))
    parser.add_argument("--prompts-file-en", default=str(PROMPTS_FILE_BY_LANG["EN"]))
    parser.add_argument("--visual-root", default=str(VISUAL_ROOT))
    parser.add_argument("--output-subdir-ru", default=OUTPUT_SUBDIR_BY_LANG["RU"])
    parser.add_argument("--output-subdir-en", default=OUTPUT_SUBDIR_BY_LANG["EN"])
    parser.add_argument("--api-base", default=BASE_URL)
    parser.add_argument("--workers", type=int, default=DEFAULT_MAX_IMAGE_WORKERS, help="Число потоков. По умолчанию 20")
    parser.add_argument("--no-skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    BASE_URL = args.api_base
    SKIP_EXISTING = not args.no_skip_existing
    ensure_api_ready()

    visual_root = Path(args.visual_root).expanduser()
    visual_root.mkdir(parents=True, exist_ok=True)

    targets: List[Tuple[str, Path, Path]] = []
    if args.lang in {"RU", "ALL"}:
        p = Path(args.prompts_file_ru).expanduser()
        if p.exists():
            targets.append(("RU", p, visual_root / args.output_subdir_ru))
        else:
            log(f"[RU] Пропускаю: не найден {p}")

    if args.lang in {"EN", "ALL"}:
        p = Path(args.prompts_file_en).expanduser()
        if p.exists():
            targets.append(("EN", p, visual_root / args.output_subdir_en))
        else:
            log(f"[EN] Пропускаю: не найден {p}")

    if not targets:
        raise RuntimeError("Не найдено ни одного файла промптов для обработки.")

    log("media_gen_api V6 image generator — RU+EN")
    log(f"BASE_URL: {BASE_URL}")
    log(f"Create endpoint: POST {V6_GENERATIONS_ENDPOINT}")
    log(f"Status endpoint: GET {V6_GENERATION_STATUS_ENDPOINT}")
    log(f"Image operation: {OP_IMAGE_GENERATE}")
    log(f"Visual root: {visual_root}")
    log(f"Aspect ratio: {IMAGE_ASPECT_RATIO}")
    log(f"Workers: {args.workers}")
    log(f"Skip existing: {SKIP_EXISTING}")

    for lang, prompts_file, output_dir in targets:
        process_language(
            lang=lang,
            prompts_file=prompts_file,
            output_dir=output_dir,
            max_workers=args.workers,
            dry_run=args.dry_run,
        )

    log("ALL DONE")


if __name__ == "__main__":
    main()
