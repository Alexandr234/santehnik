#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline: RU / PL / DE voiceovers -> Whisper transcription -> visual blocks with timecodes -> image prompts.

v3 changes (по сравнению с v2):
  * УБРАНА привязка к постоянному персонажу. Раньше в каждом кадре был один и тот же
    "парень в оливковом худи". Теперь героя-константы НЕТ: если в кадре есть человек,
    это БЕЗЛИКАЯ анонимная фигура (силуэт со спины, без узнаваемого лица), маленькая
    в масштабе огромной сцены. Часто человека может не быть вовсе — только пейзаж/символ.
  * Полностью изменён СТИЛЬ (ART_STYLE): вместо плоского минималистичного мультика —
    эпичная живописная digital-oil / concept-art картина: густые мазки, драматический
    кинематографический свет, палитра золото + глубокий багрово-красный, лучи света,
    космический масштаб, сюрреалистичная символика, атмосфера благоговения (the sublime).
    (Стиль взят с референс-кадров: одинокий силуэт на краю обрыва перед сияющим морем,
    столпы золотого света, монолиты, космический гигант из туманностей и т.п.)
  * normalize_scene / промпты больше НЕ требуют "the main character" и не описывают
    одежду/волосы. Фигура анонимна и опциональна.

Тема ролика по-прежнему задаётся через VIDEO_THEME (env VIDEO_THEME или флаг --theme),
чтобы сцены были ПО ТЕМЕ конкретного видео.

Стиль (кратко):
    EPIC PAINTERLY DIGITAL OIL / CONCEPT-ART. Одинокая безликая фигура (или толпа
    силуэтов) в огромном, залитом золотым и багровым светом мире. Густые мазки,
    драматический свет, сюрреалистичный символизм, ощущение благоговения.

Default folders:
    Voiceovers:  /Users/aleksandrtomilov/Desktop/ПСИХОЛОГИЯ ГЕРМАНИЯ ПОЛЬША/СКРИПТ ОЗВУЧКИ
    Output:      /Users/aleksandrtomilov/Desktop/ПСИХОЛОГИЯ ГЕРМАНИЯ ПОЛЬША/ПРОМПТЫ

Install:
    pip install openai
    brew install ffmpeg   # macOS

Run:
    python3 psych_prompt_pipeline_v2.py
    python3 psych_prompt_pipeline_v2.py --all
    python3 psych_prompt_pipeline_v2.py --file-ru "/path/RU.mp3"
    python3 psych_prompt_pipeline_v2.py --theme "о трансформации личности, тени и внутреннем порядке в хаосе"
    python3 psych_prompt_pipeline_v2.py --target-seconds 2.35
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import subprocess
import time
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from openai import OpenAI, APIConnectionError, APITimeoutError, RateLimitError, APIError

# ============================================================
# 1) PATHS & API
# ============================================================
DEFAULT_VOICEOVER_FOLDER = "/Users/aleksandrtomilov/Desktop/ПСИХОЛОГИЯ ГЕРМАНИЯ ПОЛЬША/СКРИПТ ОЗВУЧКИ"
DEFAULT_PROMPTS_FOLDER   = "/Users/aleksandrtomilov/Desktop/ПСИХОЛОГИЯ ГЕРМАНИЯ ПОЛЬША/ПРОМПТЫ"

# Вставь ключ сюда или задай переменную окружения OPENAI_API_KEY
OPENAI_API_KEY = ""

TRANSCRIBE_MODEL = os.getenv("TRANSCRIBE_MODEL", "whisper-1")
# Дешёвый/быстрый tier по умолчанию. Сцены — это короткий структурированный JSON,
# тяжёлая модель тут не нужна. Переопределить: export PROMPT_MODEL=gpt-5.4
PROMPT_MODEL     = os.getenv("PROMPT_MODEL", "gpt-5-mini")

LANGUAGE_ORDER = ["ru", "pl", "de"]
LANGUAGES: dict[str, dict[str, Any]] = {
    "ru": {
        "name": "Russian", "native": "русский", "folder": "RU",
        "whisper": "ru",
        "cues": ["_ru", "-ru", " ru", "ru_", "ru-", "рус", "русский", "russian", "rus"],
    },
    "pl": {
        "name": "Polish", "native": "polski", "folder": "PL",
        "whisper": "pl",
        "cues": ["_pl", "-pl", " pl", "pl_", "pl-", "pol", "polski", "polish"],
    },
    "de": {
        "name": "German", "native": "Deutsch", "folder": "DE",
        "whisper": "de",
        "cues": ["_de", "-de", " de", "de_", "de-", "deu", "deutsch", "german", "ger"],
    },
}

SUPPORTED_INPUTS = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".mp4", ".mov", ".mkv", ".webm"}

DEFAULT_TARGET_SECONDS = 2.35
MIN_BLOCK_SECONDS      = 1.0
MAX_BLOCK_SECONDS      = 3.0
SOFT_TARGET_SECONDS    = 2.35
MAX_WORDS_PER_IMAGE    = 18
DEFAULT_PROMPT_WORKERS = int(os.getenv("PROMPT_WORKERS", "4"))

# ============================================================
# 2) GLOBAL ART STYLE + FIGURE SHEET + VIDEO THEME
# ------------------------------------------------------------
# Эти строки добавляются в КАЖДЫЙ финальный промпт, чтобы все кадры были в одном
# эпично-живописном стиле. ВАЖНО: НЕТ постоянного персонажа — если в кадре есть
# человек, это анонимная безликая фигура. Локация задаётся отдельно для каждой сцены.
# ============================================================
ART_STYLE = (
    "epic painterly digital oil painting, cinematic concept art, thick expressive impasto "
    "brushstrokes and visible palette-knife texture, dramatic volumetric lighting with glowing "
    "god-rays and radiant light, rich palette of molten gold, warm amber and deep crimson red with "
    "dark shadow, high contrast between blazing warm light and deep shadow, vast sublime scale, "
    "surreal symbolic and mythic atmosphere, awe-inspiring and reverent mood, atmospheric haze and "
    "glowing embers, dramatic sky, matte-painting depth, monumental and emotional"
)

# ТЕМА РОЛИКА. Описывает, О ЧЁМ это конкретное видео, чтобы модель подбирала
# уместные сцены и символы. Можно переопределить:
#   export VIDEO_THEME="..."
# или флагом:  --theme "..."
DEFAULT_VIDEO_THEME = (
    "a deep, contemplative psychology and philosophy video about the inner journey of a person: "
    "confronting fear and the unknown, personal transformation and rebirth, facing one's shadow, "
    "the search for meaning, and finding a hidden order within chaos. "
    "The visuals are metaphorical and symbolic rather than literal — a lone anonymous human before "
    "something vast and luminous."
)
VIDEO_THEME = (os.getenv("VIDEO_THEME", "").strip() or DEFAULT_VIDEO_THEME)

# How any human FIGURE in the frame is rendered. There is NO fixed recurring character.
# Если в сцене есть человек — он безликий, анонимный, маленький в масштабе.
#   export FIGURE_DESC="..."
# или флагом:  --figure "..."
DEFAULT_FIGURE_STYLE = (
    "any human figure is an anonymous everyperson — a small lone silhouette seen mostly from behind "
    "or in dark contour, without a recognizable face or identifying features, dwarfed by the immense "
    "glowing scene, painted in the same epic oil-painting style; there is NO fixed recurring character"
)
FIGURE_STYLE = (os.getenv("FIGURE_DESC", "").strip() or DEFAULT_FIGURE_STYLE)

# How a CROWD / other people must look when the line is about society, the masses, others.
CROWD_STYLE = (
    "crowds or other people are rendered as vast masses of faceless dark silhouettes, an anonymous "
    "sea of figures, painted loosely with the same brushwork, never individualized"
)

# Things we never want from the image model. NOTE: we now WANT painterly cinematic light,
# so we only forbid photographic realism, 3d renders, and any text/logos.
NEGATIVE_SUFFIX = (
    "no photorealism, no realistic photograph, no 3d render, no cartoon, no flat vector, no anime, "
    "no text, no captions, no subtitles, no watermark, no logo, no signature, no modern clutter"
)

# ============================================================
# 3) SCENE DIVERSITY BANKS  (эпичные, символические, разные локации)
# ------------------------------------------------------------
# ВАЖНО: банки модели БОЛЬШЕ НЕ ПОКАЗЫВАЮТСЯ как подсказки — иначе они тянули бы её к
# заранее заданным местам. Локацию и символ модель выбирает САМА из смысла реплики.
# Банки остаются только как тихий fallback в normalize_scene, если поле пустое/generic.
# ============================================================
SCENE_TYPES: dict[str, dict[str, str]] = {
    "figure_before_vastness": {
        "description": "A lone anonymous figure stands small before an immense glowing landscape or light.",
        "template": "tiny lone silhouette facing a vast luminous scene",
    },
    "pure_landscape": {
        "description": "No people — only an epic symbolic landscape (sunrise, sea of light, burning sky).",
        "template": "epic empty symbolic landscape, no figure",
    },
    "the_crowd": {
        "description": "A vast sea of faceless silhouettes / the masses, sometimes with one figure apart.",
        "template": "immense crowd of faceless silhouettes",
    },
    "figure_and_crowd": {
        "description": "One lone figure set apart from or facing an anonymous crowd, showing the individual vs the many.",
        "template": "single figure contrasted against a faceless crowd",
    },
    "threshold_or_choice": {
        "description": "A doorway, path of light, or gates — the figure at a threshold, about to cross or choose.",
        "template": "figure at a glowing threshold or fork of paths",
    },
    "cosmic_or_giant": {
        "description": "A colossal luminous being / cosmic giant of nebula and stars looms over a tiny person.",
        "template": "cosmic giant of stars looming over a small figure",
    },
    "monoliths_or_pillars": {
        "description": "Monumental stone monoliths, pillars, mirrors or columns around a small figure.",
        "template": "small figure among towering monoliths",
    },
    "ascent_or_descent": {
        "description": "The figure climbing, descending, or moving through a dramatic passage toward light or dark.",
        "template": "figure ascending or descending through dramatic light",
    },
    "inner_storm": {
        "description": "Swirling embers, sparks, fire, or chaos of light expressing an inner emotional state.",
        "template": "swirling storm of light and embers around a figure",
    },
    "symbolic_metaphor": {
        "description": "A single powerful floating symbol dominates the frame (thread of light, spiral, cracked mirror).",
        "template": "one dominant glowing symbol filling the scene",
    },
    "reflection_or_mirror": {
        "description": "The figure faces a reflection, mirror, or double — the self confronting itself.",
        "template": "figure facing its own reflection or double",
    },
    "revelation": {
        "description": "A blinding burst of radiant light, an epiphany, a sun breaking over the horizon.",
        "template": "radiant burst of light, a moment of revelation",
    },
}

DEFAULT_CAMERA_BY_TYPE = {
    "figure_before_vastness":  "extreme wide shot, tiny figure low in a huge frame, seen from behind",
    "pure_landscape":          "sweeping epic wide landscape shot, no figure",
    "the_crowd":               "vast high wide shot over an endless crowd of silhouettes",
    "figure_and_crowd":        "wide shot, single figure foreground, crowd filling the background",
    "threshold_or_choice":     "wide symmetrical shot, figure centered before the threshold",
    "cosmic_or_giant":         "low-angle wide shot looking up at the towering cosmic figure",
    "monoliths_or_pillars":    "wide shot, small figure dwarfed among the monoliths",
    "ascent_or_descent":       "dramatic wide shot emphasizing vertical scale",
    "inner_storm":             "medium-wide shot, figure engulfed by swirling light",
    "symbolic_metaphor":       "centered wide shot, the glowing symbol dominating the frame",
    "reflection_or_mirror":    "wide shot showing the figure and its reflection",
    "revelation":              "wide shot into a blinding radiant light source",
}

# Эпичные символические локации (референс-стиль: обрывы, сияющее море, столпы света, космос).
ENVIRONMENT_BANK = [
    "the edge of a towering red cliff overlooking an endless glowing golden sea at sunrise",
    "a vast luminous plain stretching to a blazing sun on the horizon",
    "a cathedral of golden god-ray light beams falling from a deep blood-red sky",
    "an immense crowd of faceless silhouettes stretching to the horizon under a red sky",
    "a surreal hall of colossal cracked stone monoliths in crimson and gold",
    "a colossal cosmic giant made of nebulae and stars looming over a tiny figure",
    "an infinite golden desert beneath a burning amber sky",
    "a single narrow path of light cutting through vast darkness",
    "a storm of swirling golden embers and sparks in a dark void",
    "a mirror-smooth reflective plain glowing under a radiant low sun",
    "a row of monumental doorways of light standing in darkness",
    "molten golden waves rolling toward a red horizon",
    "a lone figure on a dark ridge silhouetted against a glowing sky",
    "a shattered mirror standing upright in a glowing golden wasteland",
]

# Символические действия/позы (референс-стиль). Общие, метафоричные.
ACTION_BANK = [
    "standing alone at the very edge, facing the vast glowing light",
    "walking a thin thread of light toward the distant horizon",
    "standing small and still before an immense luminous presence",
    "reaching one hand toward a distant radiant sun",
    "turning away from a dark crowd toward the light",
    "stepping through a glowing threshold into the unknown",
    "gazing up at a towering cosmic figure of stars",
    "arms slightly open, engulfed in swirling golden light",
    "kneeling small beneath a burst of radiant light",
    "standing between towering monoliths, tiny in scale",
    "facing its own dark reflection in a standing mirror",
    "climbing toward a blazing light high above",
    "silhouetted on a ridge against a burning sky",
    "descending into deep warm shadow away from the light",
]

# Кадрирование — эпичные широкие планы, крошечная фигура в огромном кадре.
CAMERA_BANK = [
    "extreme wide shot, tiny lone figure low in a huge frame, seen from behind",
    "sweeping epic wide landscape shot with deep atmospheric distance",
    "low-angle wide shot looking up at something towering and luminous",
    "vast high wide shot over an endless crowd of silhouettes",
    "centered symmetrical wide shot with a glowing focal light",
    "dramatic wide shot emphasizing vertical scale and depth",
    "wide silhouette shot, dark figure against blazing light",
    "medium-wide shot, figure engulfed by swirling light and embers",
]

EMOTION_BANK = [
    "awe and reverence", "existential wonder", "solemn transformation", "the sublime",
    "quiet resolve", "overwhelmed insignificance", "spiritual awakening", "sacred dread",
    "hope breaking through", "profound stillness", "transcendence", "longing toward the light",
]

# Мощные символы под тему трансформации/поиска смысла/порядка в хаосе (плюс «no symbol»).
SYMBOL_BANK = [
    "no symbol",
    "no symbol",
    "a single glowing thread or line of light",
    "a radiant sun bursting over the horizon",
    "a swirling galaxy or cosmic spiral",
    "a shattered mirror reflecting light",
    "a doorway of pure light in darkness",
    "a lone flame or ember rising",
    "a towering monolith of stone",
    "a vast sea of faceless silhouettes",
    "an ascending path of light",
    "hidden geometric order glowing within chaos",
]

DIVERSITY_MEMORY = 12
MAX_SAME_TYPE_IN_RECENT = 3
MAX_SAME_ENV_IN_RECENT  = 2
MAX_SAME_CAM_IN_RECENT  = 3
MAX_SAME_SYM_IN_RECENT  = 2

GENERIC_ENV_PATTERNS = [r"\bgeneric\b", r"\binterior\b$", r"^\s*room\s*$", r"^\s*space\s*$", r"^\s*background\s*$"]
GENERIC_ACT_PATTERNS = [r"holding still", r"looking away", r"standing quietly", r"sitting quietly", r"doing nothing"]

# ============================================================
# 4) DATACLASSES
# ============================================================
@dataclass
class Segment:
    start: float
    end: float
    text: str


@dataclass
class VisualBlock:
    index: int
    start: float
    end: float
    text: str


@dataclass
class PromptRow:
    index: int
    language_code: str
    start: float
    end: float
    duration: float
    text: str
    scene_type: str
    environment: str
    action: str
    emotion: str
    figure: str
    symbol: str
    camera_framing: str
    image_prompt: str


@dataclass
class ProcessingJob:
    language_code: str
    input_path: Path


# ============================================================
# 5) UTILITIES
# ============================================================
def clean_text(text: Any) -> str:
    if text is None:
        return ""
    if isinstance(text, (list, tuple)):
        text = " ".join(str(x) for x in text)
    return re.sub(r"\s+", " ", str(text)).strip()


def word_count(text: str) -> int:
    return len(re.findall(r"[\w']+", text, flags=re.UNICODE))


def safe_stem(path: Path) -> str:
    stem = re.sub(r"[\\/:*?\"<>|]+", "_", path.stem.strip())
    return re.sub(r"\s+", "_", stem) or "audio"


def seconds_to_srt_time(s: float) -> str:
    if s < 0:
        s = 0
    ms = int(round((s - int(s)) * 1000))
    w = int(s)
    if ms == 1000:
        w += 1; ms = 0
    return f"{w // 3600:02d}:{(w % 3600) // 60:02d}:{w % 60:02d},{ms:03d}"


def seconds_to_label(s: float) -> str:
    w = int(s)
    return f"{w // 3600:02d}:{(w % 3600) // 60:02d}:{w % 60:02d}"


def is_generic(text: str, patterns: list[str]) -> bool:
    low = text.lower()
    return any(re.search(p, low) for p in patterns)


def get_api_key() -> str:
    return (OPENAI_API_KEY or "").strip() or (os.getenv("OPENAI_API_KEY") or "").strip()


def _retryable(err: Exception) -> bool:
    code = getattr(err, "status_code", None)
    return code is None or code in {408, 409, 429} or code >= 500


def api_call(fn, desc: str, max_retries: int = 5):
    for attempt in range(1, max_retries + 1):
        try:
            return fn()
        except (APIConnectionError, APITimeoutError, RateLimitError, APIError) as e:
            if isinstance(e, APIError) and not _retryable(e):
                raise
            if attempt >= max_retries:
                raise
            delay = min(60, 5 * attempt)
            print(f"  Retry {attempt}/{max_retries} [{desc}]: {e}. Wait {delay}s...")
            time.sleep(delay)


# ============================================================
# 6) AUDIO / TRANSCRIPTION
# ============================================================
def require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found. Install: brew install ffmpeg")


def get_media_duration(path: Path) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        check=True, capture_output=True, text=True,
    )
    return float(r.stdout.strip())


def make_tmp_audio(input_path: Path, work_dir: Path) -> Path:
    require_ffmpeg()
    out = work_dir / f"{safe_stem(input_path)}_tmp.mp3"
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(input_path), "-vn", "-ac", "1", "-ar", "16000", "-b:a", "64k", str(out)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return out


def split_audio_if_needed(path: Path, work_dir: Path, chunk_sec: int = 600, max_mb: int = 24) -> list[Path]:
    if path.stat().st_size / (1024 * 1024) <= max_mb:
        return [path]
    out_dir = work_dir / f"{path.stem}_chunks"
    out_dir.mkdir(exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(path), "-f", "segment", "-segment_time", str(chunk_sec),
         "-c", "copy", str(out_dir / "chunk_%03d.mp3")],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return sorted(out_dir.glob("chunk_*.mp3"))


def transcribe_audio(client: OpenAI, input_path: Path, work_dir: Path, lang_code: str) -> list[Segment]:
    tmp = make_tmp_audio(input_path, work_dir)
    chunks = split_audio_if_needed(tmp, work_dir)
    whisper_lang = LANGUAGES.get(lang_code, {}).get("whisper")
    segments: list[Segment] = []
    offset = 0.0
    for chunk in chunks:
        print(f"  Transcribing {chunk.name} | lang={whisper_lang or 'auto'}")
        req: dict[str, Any] = {"model": TRANSCRIBE_MODEL, "response_format": "verbose_json", "temperature": 0}
        if whisper_lang:
            req["language"] = whisper_lang

        def call():
            with chunk.open("rb") as f:
                return client.audio.transcriptions.create(**{**req, "file": f})

        data = api_call(call, f"transcribe {chunk.name}")
        if hasattr(data, "model_dump"):
            data = data.model_dump()
        elif not isinstance(data, dict):
            data = json.loads(data.model_dump_json())

        segs = data.get("segments") or []
        if not segs and data.get("text"):
            segs = [{"start": 0.0, "end": get_media_duration(chunk), "text": data["text"]}]
        for s in segs:
            t = clean_text(s.get("text", ""))
            st = float(s.get("start", 0)) + offset
            en = float(s.get("end", st + 1)) + offset
            if t and en > st:
                segments.append(Segment(st, en, t))
        offset += get_media_duration(chunk)

    try:
        tmp.unlink(missing_ok=True)
    except Exception:
        pass
    return segments


# ============================================================
# 7) VISUAL BLOCK BUILDING
# ============================================================
def split_long_segment(seg: Segment) -> list[Segment]:
    dur = max(0.05, seg.end - seg.start)
    text = clean_text(seg.text)
    if dur <= MAX_BLOCK_SECONDS and word_count(text) <= MAX_WORDS_PER_IMAGE:
        return [seg]
    n = max(1, math.ceil(dur / MAX_BLOCK_SECONDS), math.ceil(word_count(text) / MAX_WORDS_PER_IMAGE))
    words = text.split()
    pieces = [" ".join(words[round(i * len(words) / n):round((i + 1) * len(words) / n)]) for i in range(n)]
    pieces = [p for p in pieces if p.strip()]
    total_w = sum(max(1, word_count(p)) for p in pieces)
    out = []
    cursor = seg.start
    for p in pieces:
        frac = max(1, word_count(p)) / total_w
        end = cursor + dur * frac
        out.append(Segment(cursor, end, clean_text(p)))
        cursor = end
    if out:
        out[-1].end = seg.end
    return out


def build_visual_blocks(segments: list[Segment], target: float = DEFAULT_TARGET_SECONDS) -> list[VisualBlock]:
    target = max(MIN_BLOCK_SECONDS, min(target, MAX_BLOCK_SECONDS))
    units: list[Segment] = []
    for seg in segments:
        units.extend(split_long_segment(seg))

    blocks: list[VisualBlock] = []
    cur: list[Segment] = []

    def flush():
        if not cur:
            return
        t = clean_text(" ".join(s.text for s in cur))
        if t:
            blocks.append(VisualBlock(len(blocks) + 1, cur[0].start, cur[-1].end, t))
        cur.clear()

    for seg in units:
        if not seg.text.strip():
            continue
        if not cur:
            cur.append(seg); continue
        cand_dur = seg.end - cur[0].start
        cur_text = clean_text(" ".join(s.text for s in cur))
        cur_dur = cur[-1].end - cur[0].start
        ends_sentence = bool(re.search(r"[.!?…]$", cur_text))
        should_flush = (
            cand_dur > MAX_BLOCK_SECONDS + 0.01
            or (word_count(clean_text(cur_text + " " + seg.text)) > MAX_WORDS_PER_IMAGE and cur_dur >= MIN_BLOCK_SECONDS)
            or (cur_dur >= target and ends_sentence)
            or (cur_dur >= SOFT_TARGET_SECONDS and word_count(cur_text) >= 8)
        )
        if should_flush:
            flush()
        cur.append(seg)
    flush()

    # Merge too-short trailing blocks
    merged: list[VisualBlock] = []
    for b in blocks:
        if merged and b.end - b.start < MIN_BLOCK_SECONDS and (b.end - merged[-1].start) <= MAX_BLOCK_SECONDS + 0.01:
            p = merged[-1]
            merged[-1] = VisualBlock(p.index, p.start, b.end, clean_text(p.text + " " + b.text))
        else:
            merged.append(b)

    for i, b in enumerate(merged, 1):
        b.index = i
    return merged


# ============================================================
# 8) DIVERSITY HELPERS
# ============================================================
def count_recent(scenes: list[dict], field: str) -> Counter:
    return Counter(clean_text(s.get(field, "")).lower() for s in scenes[-DIVERSITY_MEMORY:] if s.get(field))


def pick_fallback(pool: list[str], scenes: list[dict], field: str, max_rep: int, salt: int = 0) -> str:
    """Берём из банка вариант, который реже всего встречался в недавних сценах.
    Используется ТОЛЬКО как запасной вариант, если модель не дала своего значения."""
    import random
    counts = count_recent(scenes, field)
    rnd = random.Random(salt)
    candidates = pool[:]
    rnd.shuffle(candidates)
    candidates.sort(key=lambda c: counts[clean_text(c).lower()])
    for c in candidates:
        c_clean = clean_text(c)
        if c_clean and counts[c_clean.lower()] < max_rep:
            return c_clean
    return clean_text(candidates[0]) if candidates else ""


# ============================================================
# 9) PROMPT GENERATION (GPT)
# ============================================================
SCENE_TYPE_REFERENCE = "\n".join(f"- {k}: {v['description']}" for k, v in SCENE_TYPES.items())


def build_system_prompt() -> str:
    """Системный промпт строится из ТЕКУЩИХ VIDEO_THEME и FIGURE_STYLE (их можно
    поменять через env/флаги), поэтому это функция, а не константа на момент импорта."""
    return f"""
You plan visual scenes for a deep psychology / philosophy short video told as EPIC PAINTERLY
DIGITAL OIL PAINTINGS (cinematic concept-art). Every frame is a monumental, symbolic, awe-inspiring
image bathed in golden and deep-red light. This is metaphorical art, NOT literal illustration.

WHAT THIS VIDEO IS ABOUT (use this to choose relevant symbols and settings): {VIDEO_THEME}.

GLOBAL ART STYLE (always the same): {ART_STYLE}.
HUMAN FIGURES (there is NO fixed recurring character): {FIGURE_STYLE}.
CROWDS: {CROWD_STYLE}.

Your job: for each spoken line, design ONE powerful, cinematic scene that VISUALLY and SYMBOLICALLY
communicates the meaning of that line, grounded in the video's topic above.

CRITICAL RULES ABOUT WHAT TO SHOW:
- There is NO consistent protagonist. Do NOT describe a specific person's face, hair, clothes, age or
  identity. If a human appears, it is an anonymous, faceless, small silhouette (often seen from behind).
- Prefer symbolic / metaphorical imagery over literal depiction. Translate the idea of the line into an
  epic visual metaphor (light, darkness, scale, thresholds, crowds, cosmos, monoliths, storms, mirrors).
- Some scenes should have NO human figure at all — a pure symbolic landscape (use scene_type "pure_landscape").
- CHANGE the scene dramatically from line to line. Vary environment, scale, symbol and composition.
- Keep the majestic mood: vast scale, glowing light, deep shadow, few but powerful elements, lots of atmosphere.

OTHER GUIDANCE:
- Use crowds only when the line is about other people / society / the masses / comparison; otherwise "none".
- Use a strong SYMBOL when it helps express the idea; otherwise "no symbol".
- Never realistic photography, never a cartoon, never modern clutter. Always painterly and mythic.

Return JSON only. No prose outside JSON. Describe ONLY scene content (what/who is in frame, the setting,
the symbol, the composition). Do NOT restate the art style — it is added automatically. Never invent a
recurring named character or describe clothing/face.
""".strip()


def build_blocks_payload(blocks: list[VisualBlock]) -> list[dict]:
    payload = []
    for i, b in enumerate(blocks):
        prev = clean_text(blocks[i - 1].text) if i > 0 else ""
        nxt = clean_text(blocks[i + 1].text) if i < len(blocks) - 1 else ""
        payload.append({
            "index": b.index,
            "start": seconds_to_label(b.start),
            "end": seconds_to_label(b.end),
            "duration_seconds": round(b.end - b.start, 2),
            "previous_text": prev,
            "current_text": b.text,
            "next_text": nxt,
        })
    return payload


def generate_batch(
    client: OpenAI,
    blocks: list[VisualBlock],
    lang_code: str,
    lang_name: str,
    recent_scenes: list[dict],
) -> list[dict]:
    import random
    recent_compact = [
        {"scene_type": s.get("scene_type"), "environment": s.get("environment"),
         "action": s.get("action"), "symbol": s.get("symbol"), "camera_framing": s.get("camera_framing")}
        for s in recent_scenes[-8:]
    ]
    rnd = random.Random(len(recent_scenes) * 7 + blocks[0].index)
    cam_pool   = rnd.sample(CAMERA_BANK, min(6, len(CAMERA_BANK)))

    user_prompt = f"""
LANGUAGE OF VOICEOVER: {lang_name} ({lang_code})
(All scene fields must be in English. The voiceover text is in {lang_name}.)

VIDEO TOPIC: {VIDEO_THEME}

AVAILABLE SCENE TYPES:
{SCENE_TYPE_REFERENCE}

RECENT SCENES (make the next ones visually different, especially the SETTING and SYMBOL): {json.dumps(recent_compact, ensure_ascii=False)}

HOW TO CHOOSE THE SETTING, SYMBOL AND COMPOSITION:
- Do NOT pick from a fixed list. YOU decide, from the meaning of current_text, the most powerful epic
  visual metaphor for the idea. Translate abstract psychology into a monumental symbolic image
  (a lone figure before the sublime, a crowd, a threshold of light, a cosmic giant, a storm of embers,
  a shattered mirror, a path of light through darkness, a burning horizon...).
- A human figure is OPTIONAL and always anonymous. Prefer no figure when a pure landscape says it better.
- These framing options are the ONLY suggestions, and they are purely about camera, not content:
  cameras: {json.dumps(cam_pool)}

VOICEOVER BLOCKS:
{json.dumps(build_blocks_payload(blocks), ensure_ascii=False, indent=2)}

PRIMARY TASK: each scene must VISUALLY and SYMBOLICALLY communicate the meaning of its block's spoken text,
grounded in the VIDEO TOPIC, as an epic painterly image bathed in golden and crimson light.
Ask yourself: if a viewer watches without sound, will they FEEL the idea of this exact line?

Return JSON:
{{
  "items": [
    {{
      "index": <block index>,
      "scene_type": "<one from available scene types>",
      "subject": "<what is in the frame; if a person, an anonymous faceless silhouette; may be 'no figure'>",
      "crowd": "<faceless crowd present and what it does, or 'none'>",
      "symbol": "<one powerful visual metaphor/symbol, or 'no symbol'>",
      "environment": "<a concrete epic symbolic setting that fits this line and the topic>",
      "action": "<the key visible action or the core visual event of the scene>",
      "emotion": "<dominant mood, e.g. awe, dread, revelation>",
      "camera_framing": "<epic framing, usually a vast wide shot>",
      "reason": "<one sentence: how this scene conveys this line>"
    }}
  ]
}}

Rules:
- Exactly one item per block. All items in English.
- subject describes what is visible; if a human appears it is anonymous/faceless (never a named recurring character, no clothing/face detail). Use "no figure" for pure landscapes.
- Illustrate the SPECIFIC current_text as an epic visual metaphor, grounded in the topic.
- CHANGE the setting, scale and symbol across consecutive scenes; keep every frame monumental and painterly.
- Crowds, when present, are vast faceless silhouettes. Use them when the line is about people/society; otherwise "none".
- Use a symbol only when it truly helps; otherwise "no symbol".
- Keep it majestic and minimal in elements (few but powerful), lots of glowing atmosphere. Never photographic, never cartoon, never cluttered.
- Action / event must embody or metaphorically represent what is spoken in current_text.
- Keep descriptions concise: 4-14 words per field.
""".strip()

    response = api_call(
        lambda: client.chat.completions.create(
            model=PROMPT_MODEL,
            temperature=0.85,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": build_system_prompt()},
                {"role": "user", "content": user_prompt},
            ],
        ),
        desc=f"{lang_code.upper()} blocks {blocks[0].index}-{blocks[-1].index}",
    )
    content = response.choices[0].message.content or "{}"
    data = json.loads(content)
    return data.get("items", [])


def build_final_prompt(scene: dict) -> str:
    """Assemble the full image prompt: global epic style + scene content + crowd + symbol + framing + negatives.
    NOTE: no fixed character is injected; a figure is only present if the scene calls for one."""
    subject = clean_text(scene.get("subject", ""))
    action = clean_text(scene.get("action", ""))
    crowd = clean_text(scene.get("crowd", ""))
    symbol = clean_text(scene.get("symbol", ""))
    environment = clean_text(scene.get("environment", ""))
    emotion = clean_text(scene.get("emotion", ""))
    framing = clean_text(scene.get("camera_framing", ""))

    # Drop "empty" sentinels
    no_figure = subject.lower() in {"", "no figure", "none", "no person", "empty"}
    if crowd.lower() in {"", "none", "no one", "nobody"}:
        crowd = ""
    if symbol.lower() in {"", "no symbol", "none"}:
        symbol = ""

    parts = [ART_STYLE]

    # The subject / main visual content of the frame.
    if no_figure:
        # pure landscape: the action still describes the core visual event, no figure sheet added
        if action:
            parts.append(action)
    else:
        parts.append(", ".join(p for p in [subject, action] if p))
        parts.append(FIGURE_STYLE)

    if crowd:
        parts.append(f"{crowd}; {CROWD_STYLE}")
    if symbol:
        parts.append(f"symbolic element: {symbol}")
    if environment:
        parts.append(f"setting: {environment}")
    parts.append(emotion)
    parts.append(framing)
    parts.append(NEGATIVE_SUFFIX)

    prompt = ". ".join(p for p in parts if p)
    return re.sub(r"\s+", " ", prompt).strip()


def normalize_scene(raw: dict, b: VisualBlock, recent: list[dict], salt: int) -> dict:
    """ДОВЕРЯЕМ модели. Берём её поля как есть; банк используется только если поле
    пустое или явно generic. Никакого постоянного персонажа — фигура анонимна/опциональна."""
    scene_type = clean_text(raw.get("scene_type", "")) or "figure_before_vastness"
    if scene_type not in SCENE_TYPES:
        scene_type = "figure_before_vastness"

    environment = clean_text(raw.get("environment", ""))
    if not environment or is_generic(environment, GENERIC_ENV_PATTERNS):
        environment = pick_fallback(ENVIRONMENT_BANK, recent, "environment", MAX_SAME_ENV_IN_RECENT, salt)

    action = clean_text(raw.get("action", ""))
    if not action or is_generic(action, GENERIC_ACT_PATTERNS):
        action = pick_fallback(ACTION_BANK, recent, "action", 2, salt + 1)

    emotion = clean_text(raw.get("emotion", "")) or pick_fallback(EMOTION_BANK, recent, "emotion", 3, salt + 2)

    camera = clean_text(raw.get("camera_framing", "")) or \
        DEFAULT_CAMERA_BY_TYPE.get(scene_type, "extreme wide shot, tiny lone figure low in a huge frame")

    # subject: anonymous figure or explicit "no figure" for pure landscapes.
    subject = clean_text(raw.get("subject", ""))
    if not subject:
        subject = "no figure" if scene_type == "pure_landscape" else "a lone anonymous silhouette"

    # accept crowd from "crowd" or legacy "others" field
    crowd = clean_text(raw.get("crowd", "")) or clean_text(raw.get("others", ""))
    if crowd.lower() in {"none", "no one", "nobody", ""}:
        crowd = "none"

    symbol = clean_text(raw.get("symbol", ""))
    if symbol.lower() in {"no symbol", "none", ""}:
        symbol = "no symbol"

    return {
        "scene_type": scene_type,
        "subject": subject,
        "crowd": crowd,
        "symbol": symbol,
        "environment": environment,
        "action": action,
        "emotion": emotion,
        "camera_framing": camera,
        "reason": clean_text(raw.get("reason", "")),
    }


FALLBACK_SCENES = [
    {"scene_type": "figure_before_vastness", "subject": "a lone anonymous silhouette seen from behind",
     "crowd": "none", "symbol": "a radiant sun bursting over the horizon",
     "environment": "the edge of a towering red cliff overlooking an endless glowing golden sea at sunrise",
     "action": "standing alone at the very edge, facing the vast glowing light",
     "emotion": "awe and reverence", "camera_framing": "extreme wide shot, tiny figure low in a huge frame, seen from behind"},
    {"scene_type": "the_crowd", "subject": "a vast sea of faceless dark silhouettes",
     "crowd": "an endless anonymous crowd stretching to the horizon", "symbol": "no symbol",
     "environment": "an immense crowd of faceless silhouettes stretching to the horizon under a red sky",
     "action": "an ocean of silhouettes facing a distant burning light",
     "emotion": "overwhelmed insignificance", "camera_framing": "vast high wide shot over an endless crowd of silhouettes"},
    {"scene_type": "cosmic_or_giant", "subject": "a tiny anonymous figure beneath a colossal cosmic being",
     "crowd": "none", "symbol": "a swirling galaxy or cosmic spiral",
     "environment": "a colossal cosmic giant made of nebulae and stars looming over a tiny figure",
     "action": "gazing up at a towering cosmic figure of stars",
     "emotion": "the sublime", "camera_framing": "low-angle wide shot looking up at the towering cosmic figure"},
    {"scene_type": "threshold_or_choice", "subject": "a small silhouette before doorways of light",
     "crowd": "none", "symbol": "a doorway of pure light in darkness",
     "environment": "a row of monumental doorways of light standing in darkness",
     "action": "stepping through a glowing threshold into the unknown",
     "emotion": "solemn transformation", "camera_framing": "wide symmetrical shot, figure centered before the threshold"},
]


def generate_prompts(
    client: OpenAI,
    blocks: list[VisualBlock],
    lang_code: str,
    batch_size: int = 24,
    workers: int = DEFAULT_PROMPT_WORKERS,
) -> list[PromptRow]:
    lang_name = LANGUAGES.get(lang_code, {}).get("name", lang_code.upper())
    batches = [blocks[i:i + batch_size] for i in range(0, len(blocks), batch_size)]
    print(f"  Prompt batches ({lang_code.upper()}): {len(batches)}  workers={workers}")

    by_index: dict[int, dict] = {}

    if workers <= 1:
        recent: list[dict] = []
        for batch in batches:
            print(f"  Generating {lang_code.upper()} [{batch[0].index}-{batch[-1].index}]...")
            raw_items = generate_batch(client, batch, lang_code, lang_name, recent)
            raw_by_idx = {int(it["index"]): it for it in raw_items if "index" in it}
            for b in batch:
                salt = b.index * 31
                raw = raw_by_idx.get(b.index, {})
                scene = normalize_scene(raw, b, recent, salt)
                scene["block_index"] = b.index
                by_index[b.index] = scene
                recent.append(scene)
                if len(recent) > DIVERSITY_MEMORY:
                    recent.pop(0)
    else:
        futures_map = {}
        with ThreadPoolExecutor(max_workers=min(workers, len(batches))) as ex:
            for batch in batches:
                f = ex.submit(generate_batch, client, batch, lang_code, lang_name, [])
                futures_map[f] = batch
            for future in as_completed(futures_map):
                batch = futures_map[future]
                raw_items = future.result()
                raw_by_idx = {int(it["index"]): it for it in raw_items if "index" in it}
                for b in batch:
                    raw = raw_by_idx.get(b.index, {})
                    scene = normalize_scene(raw, b, [], b.index * 31)
                    scene["block_index"] = b.index
                    by_index[b.index] = scene

    rows: list[PromptRow] = []
    for b in blocks:
        scene = by_index.get(b.index)
        if not scene:
            fb = FALLBACK_SCENES[b.index % len(FALLBACK_SCENES)]
            scene = {**fb, "block_index": b.index, "reason": "fallback"}
        rows.append(PromptRow(
            index=b.index, language_code=lang_code,
            start=b.start, end=b.end, duration=b.end - b.start,
            text=b.text,
            scene_type=scene.get("scene_type", ""),
            environment=scene.get("environment", ""),
            action=scene.get("action", ""),
            emotion=scene.get("emotion", ""),
            figure=scene.get("subject", ""),
            symbol=scene.get("symbol", "no symbol"),
            camera_framing=scene.get("camera_framing", ""),
            image_prompt=build_final_prompt(scene),
        ))
    return rows


# ============================================================
# 10) OUTPUT WRITERS
# ============================================================
def write_srt(items: list[VisualBlock] | list[Segment], path: Path) -> None:
    lines = []
    for i, seg in enumerate(items, 1):
        t = clean_text(seg.text)
        if not t:
            continue
        lines += [str(i), f"{seconds_to_srt_time(seg.start)} --> {seconds_to_srt_time(seg.end)}", t, ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_outputs(
    input_path: Path, lang_code: str,
    segments: list[Segment], blocks: list[VisualBlock], rows: list[PromptRow],
    lang_dir: Path, root_dir: Path, target: float,
) -> None:
    stem = f"{lang_code}_{safe_stem(input_path)}"

    write_srt(segments, lang_dir / f"{stem}_raw.srt")
    write_srt(blocks,   lang_dir / f"{stem}_image_blocks.srt")
    (lang_dir / f"{stem}_visual_blocks.json").write_text(
        json.dumps([asdict(b) for b in blocks], ensure_ascii=False, indent=2), encoding="utf-8")

    # prompts.json
    prompts_data = [asdict(r) for r in rows]
    (lang_dir / f"{stem}_prompts.json").write_text(
        json.dumps(prompts_data, ensure_ascii=False, indent=2), encoding="utf-8")

    # prompts.csv
    fieldnames = ["index", "language_code", "start", "end", "duration",
                  "text", "scene_type", "environment", "action", "emotion",
                  "figure", "symbol", "camera_framing", "image_prompt"]
    with (lang_dir / f"{stem}_prompts.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({
                "index": r.index, "language_code": r.language_code,
                "start": seconds_to_srt_time(r.start), "end": seconds_to_srt_time(r.end),
                "duration": round(r.duration, 2), "text": r.text,
                "scene_type": r.scene_type, "environment": r.environment,
                "action": r.action, "emotion": r.emotion,
                "figure": r.figure, "symbol": r.symbol,
                "camera_framing": r.camera_framing, "image_prompt": r.image_prompt,
            })

    # prompts.txt — human-readable with timecodes
    txt_lines = []
    for r in rows:
        txt_lines.append(f"{r.index:03d} | {seconds_to_srt_time(r.start)} --> {seconds_to_srt_time(r.end)} | {r.duration:.2f}s | {r.scene_type}")
        txt_lines.append(f"TEXT: {r.text}")
        txt_lines.append(r.image_prompt)
        txt_lines.append("")
    (lang_dir / f"{stem}_prompts.txt").write_text("\n".join(txt_lines), encoding="utf-8")

    # image_times
    times_data = []
    time_txt_lines = []
    for r in rows:
        times_data.append({
            "index": r.index, "image_filename": f"{r.index:03d}.png",
            "start": seconds_to_srt_time(r.start), "end": seconds_to_srt_time(r.end),
            "start_seconds": round(r.start, 3), "end_seconds": round(r.end, 3),
            "duration": round(r.duration, 3), "text": r.text,
        })
        time_txt_lines.append(f"{r.index:03d} | {seconds_to_srt_time(r.start)} --> {seconds_to_srt_time(r.end)} | {r.duration:.2f}s | image: {r.index:03d}.png")

    (lang_dir / f"{stem}_image_times.json").write_text(json.dumps(times_data, ensure_ascii=False, indent=2), encoding="utf-8")
    (lang_dir / f"{stem}_image_times.txt").write_text("\n".join(time_txt_lines) + "\n", encoding="utf-8")
    with (lang_dir / f"{stem}_image_times.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["index", "image_filename", "start", "end", "start_seconds", "end_seconds", "duration", "text"])
        w.writeheader()
        for item in times_data:
            w.writerow(item)

    # Root-level copies for easy access
    for src_name, dst_name in [
        (f"{stem}_prompts.txt",      f"prompts_{lang_code}.txt"),
        (f"{stem}_image_times.txt",  f"image_times_{lang_code}.txt"),
        (f"{stem}_image_times.json", f"image_times_{lang_code}.json"),
        (f"{stem}_image_blocks.srt", f"image_blocks_{lang_code}.srt"),
    ]:
        src = lang_dir / src_name
        if src.exists():
            shutil.copyfile(src, root_dir / dst_name)

    # Stats summary
    dur = max((s.end for s in segments), default=0.0)
    sc_counts = Counter(r.scene_type for r in rows)
    env_counts = Counter(r.environment for r in rows)
    avg = dur / len(rows) if rows else 0
    summary = (
        f"Input: {input_path.name}\n"
        f"Language: {LANGUAGES.get(lang_code, {}).get('name', lang_code)} ({lang_code})\n"
        f"Video theme: {VIDEO_THEME}\n"
        f"Duration: {dur:.2f}s ({dur/60:.2f} min)\n"
        f"Whisper segments: {len(segments)}\n"
        f"Image blocks / prompts: {len(rows)}\n"
        f"Target seconds/image: {target:.2f} | Max: {MAX_BLOCK_SECONDS:.2f}\n"
        f"Average seconds/image: {avg:.2f}\n"
        f"Scene type distribution: {dict(sc_counts.most_common())}\n"
        f"Distinct environments used: {len(env_counts)}\n"
    )
    (lang_dir / f"{stem}_summary.txt").write_text(summary, encoding="utf-8")
    print(summary)


# ============================================================
# 11) JOB DETECTION
# ============================================================
def find_input_files(folder: Path) -> list[Path]:
    return sorted(
        [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_INPUTS
         and "_tmp" not in p.name],
        key=lambda p: p.stat().st_mtime, reverse=True,
    )


def detect_lang(path: Path) -> str | None:
    text = f" {path.stem.lower()} ".replace(".", " ")
    for code in LANGUAGE_ORDER:
        for cue in LANGUAGES[code]["cues"]:
            if cue in text:
                return code
    return None


def build_jobs(args: argparse.Namespace) -> list[ProcessingJob]:
    folder = Path(args.voiceover_folder).expanduser()
    if not folder.exists():
        raise RuntimeError(f"Voiceover folder not found: {folder}")

    # Manual overrides
    jobs = []
    for code in LANGUAGE_ORDER:
        val = getattr(args, f"file_{code}", None)
        if val:
            jobs.append(ProcessingJob(code, Path(val).expanduser()))
    if args.file:
        code = args.language or detect_lang(Path(args.file)) or "unknown"
        jobs.append(ProcessingJob(code, Path(args.file).expanduser()))
    if jobs:
        return jobs

    files = find_input_files(folder)
    if not files:
        raise RuntimeError(f"No supported audio files found in: {folder}")

    if args.all:
        return [ProcessingJob(detect_lang(p) or "unknown", p) for p in files]

    selected: dict[str, Path] = {}
    unknown: list[Path] = []
    for p in files:
        code = detect_lang(p)
        if code and code not in selected:
            selected[code] = p
        elif code is None:
            unknown.append(p)

    # Auto-assign unknown files to missing languages
    for code in LANGUAGE_ORDER:
        if code not in selected and unknown:
            selected[code] = unknown.pop(0)
            print(f"  INFO: Auto-assigned '{selected[code].name}' -> {code.upper()}")

    if not selected:
        raise RuntimeError(
            "Could not detect language from filenames. "
            "Add _ru / _pl / _de markers, or use --file-ru / --file-pl / --file-de."
        )

    return [ProcessingJob(code, selected[code]) for code in LANGUAGE_ORDER if code in selected]


# ============================================================
# 12) MAIN
# ============================================================
def process_job(
    client: OpenAI, job: ProcessingJob,
    root_dir: Path, target: float, force: bool, workers: int,
) -> None:
    path = job.input_path.expanduser()
    if not path.exists():
        raise RuntimeError(f"Input file not found: {path}")
    lang_dir = root_dir / LANGUAGES.get(job.language_code, {}).get("folder", job.language_code.upper())
    lang_dir.mkdir(parents=True, exist_ok=True)

    stem = f"{job.language_code}_{safe_stem(path)}"
    prompts_txt = lang_dir / f"{stem}_prompts.txt"
    if not force and prompts_txt.exists() and prompts_txt.stat().st_size > 100:
        print(f"  SKIP {path.name} ({job.language_code.upper()}) — outputs already exist. Use --force to regenerate.")
        for src_name, dst_name in [
            (f"{stem}_prompts.txt",      f"prompts_{job.language_code}.txt"),
            (f"{stem}_image_times.txt",  f"image_times_{job.language_code}.txt"),
            (f"{stem}_image_times.json", f"image_times_{job.language_code}.json"),
            (f"{stem}_image_blocks.srt", f"image_blocks_{job.language_code}.srt"),
        ]:
            src = lang_dir / src_name
            if src.exists():
                shutil.copyfile(src, root_dir / dst_name)
        return

    print(f"\n=== {path.name} | {job.language_code.upper()} ===")
    segments = transcribe_audio(client, path, lang_dir, job.language_code)
    if not segments:
        raise RuntimeError(f"No transcription segments for {path}")
    print(f"  Segments: {len(segments)}")

    blocks = build_visual_blocks(segments, target)
    print(f"  Visual blocks: {len(blocks)}")

    rows = generate_prompts(client, blocks, job.language_code, workers=workers)
    write_outputs(path, job.language_code, segments, blocks, rows, lang_dir, root_dir, target)


def main() -> None:
    parser = argparse.ArgumentParser(description="RU/PL/DE voiceover -> timecoded epic-painterly image prompts (v3, no fixed character)")
    parser.add_argument("--voiceover-folder", default=DEFAULT_VOICEOVER_FOLDER)
    parser.add_argument("--output-folder",    default=DEFAULT_PROMPTS_FOLDER)
    parser.add_argument("--all", action="store_true", help="Process all files in voiceover folder")
    parser.add_argument("--file",    default=None, help="Process a single file")
    parser.add_argument("--language", choices=LANGUAGE_ORDER + ["unknown"], default=None)
    parser.add_argument("--file-ru", default=None)
    parser.add_argument("--file-pl", default=None)
    parser.add_argument("--file-de", default=None)
    parser.add_argument("--target-seconds", type=float, default=DEFAULT_TARGET_SECONDS,
                        help="Target seconds per image (1.0-3.0, default 2.35)")
    parser.add_argument("--theme", default=None,
                        help="Тема ролика одной фразой (англ. лучше всего): о чём видео, чтобы сцены были по теме. "
                             "По умолчанию берётся VIDEO_THEME из окружения или встроенная.")
    parser.add_argument("--figure", default=None,
                        help="Переопределить, как рисуются АНОНИМНЫЕ фигуры (одной строкой). "
                             "Постоянного персонажа нет; по умолчанию берётся FIGURE_DESC из окружения или встроенное.")
    parser.add_argument("--force",          action="store_true", help="Regenerate even if outputs exist")
    parser.add_argument("--prompt-workers", type=int, default=DEFAULT_PROMPT_WORKERS,
                        help="Parallel GPT workers (default 4)")
    args = parser.parse_args()

    global FIGURE_STYLE, VIDEO_THEME
    if args.figure and args.figure.strip():
        FIGURE_STYLE = args.figure.strip()
    if args.theme and args.theme.strip():
        VIDEO_THEME = args.theme.strip()
    print(f"VIDEO THEME: {VIDEO_THEME}")
    print(f"FIGURE:      {FIGURE_STYLE}")

    api_key = get_api_key()
    if not api_key:
        raise RuntimeError(
            "Set OPENAI_API_KEY in this script (OPENAI_API_KEY = 'sk-...') "
            "or via: export OPENAI_API_KEY='sk-...'"
        )

    root_dir = Path(args.output_folder).expanduser()
    root_dir.mkdir(parents=True, exist_ok=True)

    jobs = build_jobs(args)
    client = OpenAI(api_key=api_key)

    print(f"Jobs: {[(j.language_code.upper(), j.input_path.name) for j in jobs]}")
    print(f"Output: {root_dir}")

    for job in jobs:
        process_job(client, job, root_dir,
                    target=args.target_seconds, force=args.force,
                    workers=args.prompt_workers)

    print(f"\nDone. All outputs saved to: {root_dir}")


if __name__ == "__main__":
    main()
