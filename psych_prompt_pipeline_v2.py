#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline: RU / PL / DE voiceovers -> Whisper transcription -> visual blocks with timecodes -> image prompts.

v2 changes (по сравнению с gpt54):
  * Стиль рисунка (ART_STYLE) больше НЕ привязывает персонажа к одной пустой комнате —
    локация теперь свободная и разная от сцены к сцене.
  * Добавлена тема ролика VIDEO_THEME (env VIDEO_THEME или флаг --theme), которая
    подаётся модели, чтобы сцены были ПО ТЕМЕ конкретного видео.
  * Банки сцен/локаций/действий/символов сделаны общими и бытовыми (двор, улица,
    салон авто, дорога, магазин, комната), а не «травма + цепи + маски».
  * normalize_scene ДОВЕРЯЕТ модели: environment/action/symbol берутся как есть,
    банк используется только если поле пустое. Жёсткой случайной подмены больше нет —
    именно она раньше делала картинки нерелевантными тексту.
  * Озвучки читаются из папки ОЗВУЧКА (куда их кладёт voicer_batch_tts_RU_PL_DE.py).

Стиль остаётся: FLAT 2D HAND-DRAWN MINIMALIST CARTOON (простой "webcomic / explainer")
с ОДНИМ постоянным героем:

    Молодой парень с короткими растрёпанными каштановыми волосами и минималистичным
    лицом-точками, в оливково-зелёном худи и таких же зелёных штанах.

Герой описан в ОДНОМ месте (MAIN_CHARACTER). Меняй там, либо через env MAIN_CHARACTER_DESC,
либо флагом --character.

Все остальные люди в кадре — простые БЕЛЫЕ безликие фигуры (другие люди / общество),
контрастирующие с цветным героем.

Default folders:
    Voiceovers:  /Users/aleksandrtomilov/Desktop/ПСИХОЛОГИЯ ГЕРМАНИЯ ПОЛЬША/ОЗВУЧКА
    Output:      /Users/aleksandrtomilov/Desktop/ПСИХОЛОГИЯ ГЕРМАНИЯ ПОЛЬША/ПРОМПТЫ

Install:
    pip install openai
    brew install ffmpeg   # macOS

Run:
    python3 psych_prompt_pipeline_v2.py
    python3 psych_prompt_pipeline_v2.py --all
    python3 psych_prompt_pipeline_v2.py --file-ru "/path/RU.mp3"
    python3 psych_prompt_pipeline_v2.py --theme "про человека, который 10 лет ездит на одной машине; о потреблении, статусе и внутренней стабильности"
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
# ВАЖНО: озвучки читаются из папки ОЗВУЧКА — именно туда voicer_batch_tts_RU_PL_DE.py
# сохраняет scenario_ru.mp3 / scenario_pl.mp3 / scenario_de.mp3. Папки обязаны совпадать,
# иначе пайплайн подхватит не те (старые) файлы.
DEFAULT_VOICEOVER_FOLDER = "/Users/aleksandrtomilov/Desktop/ПСИХОЛОГИЯ ГЕРМАНИЯ ПОЛЬША/ОЗВУЧКА"
DEFAULT_PROMPTS_FOLDER   = "/Users/aleksandrtomilov/Desktop/ПСИХОЛОГИЯ ГЕРМАНИЯ ПОЛЬША/ПРОМПТЫ"

# Вставь ключ сюда или задай переменную окружения OPENAI_API_KEY
OPENAI_API_KEY = ""

TRANSCRIBE_MODEL = os.getenv("TRANSCRIBE_MODEL", "whisper-1")
PROMPT_MODEL     = os.getenv("PROMPT_MODEL", "gpt-5.4")

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
# 2) GLOBAL ART STYLE + CHARACTER SHEET + VIDEO THEME
# ------------------------------------------------------------
# Эти строки добавляются в КАЖДЫЙ финальный промпт, чтобы все кадры были в одном
# стиле и с одним и тем же героем. ВАЖНО: локация сюда больше НЕ зашита —
# окружение задаётся отдельно для каждой сцены, поэтому герой может быть где угодно.
# ============================================================
ART_STYLE = (
    "flat 2D hand-drawn cartoon illustration, minimalist webcomic explainer style, "
    "simple clean thin black outlines, flat cel shading with no gradients, "
    "muted earthy color palette of sage green, olive, cream, beige and soft warm grey, "
    "simple shapes and very few details, soft oval drop shadow under each figure, "
    "flat even soft lighting, plenty of empty negative space, calm restrained quiet mood, "
    "simple and clean, storyboard frame"
)

# ТЕМА РОЛИКА. Описывает, О ЧЁМ это конкретное видео, чтобы модель подбирала
# уместные сцены и реквизит (а не случайные предметы). Можно переопределить:
#   export VIDEO_THEME="..."
# или флагом:  --theme "..."
DEFAULT_VIDEO_THEME = (
    "a calm psychology video about consumer culture and the urge to constantly replace things "
    "(phones, shoes, furniture, gadgets, cars, even relationships and identity), advertising that "
    "manufactures dissatisfaction, status symbols and social comparison, and the quiet inner "
    "stability of a person who is content with what he already has. "
    "An old car kept for ten years is only the RECURRING EXAMPLE that opens and closes the video — "
    "it is NOT what every scene is about. Most lines are about consumption in general, not cars."
)
VIDEO_THEME = (os.getenv("VIDEO_THEME", "").strip() or DEFAULT_VIDEO_THEME)

# The ONE recurring protagonist. Always rendered the same way.
# ВАЖНО: это ЕДИНСТВЕННОЕ место, где описан персонаж. Меняй здесь —
# и все промпты на всех языках сразу станут с правильным персонажем.
#   export MAIN_CHARACTER_DESC="the main character is ..."
# или флагом:  --character "the main character is ..."
DEFAULT_MAIN_CHARACTER = (
    "the main character is a young man with short tousled brown hair (he is NOT bald), "
    "a simple minimal cartoon face (two small dot eyes and a small subtle mouth), light plain skin, "
    "wearing an olive-green hooded sweatshirt and matching green sweatpants, drawn in the same flat cartoon style"
)

MAIN_CHARACTER = (os.getenv("MAIN_CHARACTER_DESC", "").strip() or DEFAULT_MAIN_CHARACTER)

# How every OTHER person in the frame must look (the crowd / family / society).
OTHERS_STYLE = (
    "any other people are drawn as plain blank pure-white humanoid figures with smooth featureless "
    "white bodies and little or no facial features, clearly contrasting with the colored main character"
)

# Things we never want from the image model.
NEGATIVE_SUFFIX = (
    "no photorealism, no 3d render, no realistic photo, no cinematic lighting, no film grain, "
    "no detailed realistic background, no text, no captions, no watermark, no logo, no signature"
)

# ============================================================
# 3) SCENE DIVERSITY BANKS  (общие, бытовые, с РАЗНЫМИ локациями)
# ------------------------------------------------------------
# ВАЖНО: эти банки модели БОЛЬШЕ НЕ ПОКАЗЫВАЮТСЯ как подсказки — иначе они тянули бы
# её к заранее заданным местам/предметам. Локацию и реквизит модель выбирает САМА из
# смысла конкретной реплики. Банки остаются только как тихий fallback в normalize_scene,
# если модель вдруг вернула пустое или явно generic поле.
# ============================================================
SCENE_TYPES: dict[str, dict[str, str]] = {
    "subject_alone": {
        "description": "Main character alone in a setting that fits the narration, full body, calm.",
        "template": "main character alone in a meaningful simple setting",
    },
    "with_object": {
        "description": "Character interacting with a key object that embodies the topic (e.g. his old car, a phone, a thing he keeps).",
        "template": "character touching or using the object the story is about",
    },
    "in_environment": {
        "description": "Character placed in a specific everyday location relevant to the line (street, driveway, shop, road).",
        "template": "character standing inside a clear real-world location",
    },
    "others_around": {
        "description": "Blank white figures (neighbours, crowd, society) react, watch, or pass by the character.",
        "template": "blank white figures around the main character",
    },
    "comparison_split": {
        "description": "Character/old thing on one side, blank figures or a shiny new thing on the other, to show contrast.",
        "template": "side-by-side contrast: the character's old thing vs a new shiny one",
    },
    "observed_or_judged": {
        "description": "A blank white figure questions, points at, or looks at the character, who stays calm.",
        "template": "white figure asking or judging, character unbothered",
    },
    "symbolic_metaphor": {
        "description": "A single simple floating symbol expresses the idea of the line (price tag, sparkle, arrow, anchor).",
        "template": "one clear simple symbol expressing the idea",
    },
    "in_transit": {
        "description": "Character moving through the world: driving, walking, going somewhere.",
        "template": "character driving or walking, in motion",
    },
    "contemplative": {
        "description": "Character pauses and reflects, relaxed and thoughtful.",
        "template": "character standing or sitting, calm and reflective",
    },
    "surrounded_by_choices": {
        "description": "Character among many products / options / shiny new things to show pressure to upgrade.",
        "template": "character surrounded by many tempting new objects",
    },
    "calm_contrast": {
        "description": "Character is calm and content while blank white figures rush, upgrade, or chase the new.",
        "template": "calm character while white figures rush around him",
    },
}

DEFAULT_CAMERA_BY_TYPE = {
    "subject_alone":          "full-body wide shot at eye level, character centered",
    "with_object":            "full-body medium-wide shot showing the character and the object",
    "in_environment":         "wide shot showing the character inside the location",
    "others_around":          "full-body wide shot, character centered among the white figures",
    "comparison_split":       "wide shot with both sides framed for contrast",
    "observed_or_judged":     "full-body medium-wide shot, slight side angle on both figures",
    "symbolic_metaphor":      "medium full-body shot, the floating symbol clearly visible",
    "in_transit":             "side-on wide shot showing motion",
    "contemplative":          "medium full-body shot, slight side angle",
    "surrounded_by_choices":  "wide shot with the character among many objects",
    "calm_contrast":          "wide shot, calm character still while figures blur past",
}

# Разнообразные минималистичные БЫТОВЫЕ локации (дом, магазины, улица + ОДНА-две авто).
# Намеренно общие: машина здесь лишь один из вариантов, а не основа.
ENVIRONMENT_BANK = [
    "simple living room with a sofa and a low table",
    "simple kitchen with a plain counter and one cup",
    "minimal bedroom with a bed and a small shelf",
    "plain shop interior with simple shelves of products",
    "minimal electronics store wall with a row of identical phones",
    "simple shoe-shop shelf with a few pairs of shoes",
    "quiet residential street suggested by a sidewalk and one small tree",
    "sidewalk in front of a simple house",
    "minimal park suggested by a bench and a single tree",
    "bare cream room with two walls meeting in a corner and a grey-green floor",
    "wide empty horizon with a flat ground line and open sky",
    "simple driveway with a plain older car parked on it",
    "minimal interior of an old car, simple dashboard and steering wheel",
    "small living room glowing from a TV showing an advert",
]

# Простые, читаемые действия и жесты (бытовые, по теме «вещи / выбор / спокойствие»).
# Общие, НЕ привязанные к машине: предмет в руках зависит от конкретной реплики.
ACTION_BANK = [
    "standing calmly with a relaxed, easy posture",
    "holding an old worn phone while ignoring a shiny new one",
    "looking at a wall of identical new products without reaching for any",
    "shrugging lightly in answer to a question",
    "standing still and content while white figures rush past with shopping bags",
    "walking calmly down a simple street",
    "sitting relaxed and looking thoughtfully into the distance",
    "crossing his arms with quiet confidence",
    "watching a glowing advert on a screen with a calm face",
    "keeping a familiar old object while others hold shiny new ones",
    "fondly using a well-worn everyday object",
    "standing between an old thing and a new shiny one, choosing the old",
    "resting one hand fondly on his old car (only when the line is about the car)",
    "calmly driving his old car (only when the line is about the car)",
]

# Варианты кадрирования (почти всегда полный рост, на уровне глаз).
CAMERA_BANK = [
    "full-body wide shot at eye level, character centered",
    "wide shot with the character drawn small in a large frame",
    "full-body medium-wide shot, slight side angle",
    "front-facing full-body shot at eye level",
    "wide shot showing the whole location and the floor shadow",
    "medium full-body shot focused on the character and one object",
    "side-on wide shot showing motion",
    "wide symmetrical shot with the character in the middle",
]

EMOTION_BANK = [
    "calm contentment", "quiet confidence", "unbothered ease",
    "peaceful satisfaction", "mild nostalgia", "gentle pride",
    "thoughtful calm", "secure and grounded", "indifference to trends",
    "quiet stubborn loyalty", "relaxed familiarity", "settled inner stability",
]

# Простые символы под тему потребления/статуса/стабильности (плюс «no symbol»).
SYMBOL_BANK = [
    "no symbol",
    "no symbol",
    "a glowing price tag or dollar sign",
    "a shiny 'new' sparkle effect on an object",
    "a small upward status arrow",
    "a steady anchor symbol for stability",
    "a thought bubble with a question mark",
    "a clock or calendar showing years passing",
    "a small heart of attachment over an old object",
    "a row of identical shiny new products",
    "a green checkmark of contentment",
    "a treadmill / hamster-wheel symbol of endless upgrading",
]

DIVERSITY_MEMORY = 12
MAX_SAME_TYPE_IN_RECENT = 3
MAX_SAME_ENV_IN_RECENT  = 2
MAX_SAME_CAM_IN_RECENT  = 3
MAX_SAME_SYM_IN_RECENT  = 2

GENERIC_ENV_PATTERNS = [r"\bgeneric room\b", r"\binterior\b$", r"^\s*room\s*$", r"^\s*space\s*$"]
GENERIC_ACT_PATTERNS = [r"holding still", r"looking away", r"standing quietly", r"sitting quietly"]

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
    others: str
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
    """Системный промпт строится из ТЕКУЩИХ VIDEO_THEME и MAIN_CHARACTER (их можно
    поменять через env/флаги), поэтому это функция, а не константа на момент импорта."""
    return f"""
You plan visual scenes for a psychology short video told in a FLAT 2D HAND-DRAWN MINIMALIST CARTOON style
(simple "webcomic / explainer" look). There is ONE recurring main character in every scene.

WHAT THIS VIDEO IS ABOUT (use this to choose relevant settings and props): {VIDEO_THEME}.

GLOBAL ART STYLE (always the same): {ART_STYLE}.
MAIN CHARACTER (always the same, keep him visually identical in every scene): {MAIN_CHARACTER}.
OTHER PEOPLE: {OTHERS_STYLE}.

Your job: for each spoken line, design ONE clear cartoon scene that VISUALLY COMMUNICATES the meaning
of that line, grounded in the video's topic above.

CRITICAL RULES ABOUT WHAT TO SHOW:
- Illustrate the SPECIFIC current line, not the overall topic. Whatever object that line mentions
  (a phone, shoes, furniture, a gadget, an advert, a crowd, money, an identity) is what should appear.
- Do NOT put a car in every scene. Show the car ONLY when the current line literally talks about the car
  or about driving. For all other lines use the object that line is actually about. The old car is just
  the opening/closing example of the video, not a default prop.
- Do NOT keep the character in the same place every time. CHANGE the environment from scene to scene
  (living room, shop, electronics store, street, his home, a bare room, etc.).
- The room with bare cream walls is just ONE possible location, not the default. Avoid reusing it repeatedly.

OTHER GUIDANCE:
- Use blank white figures only when the line is about other people / society / comparison; otherwise "none".
- Use a simple SYMBOL only when it genuinely helps express the idea; otherwise "no symbol".
- Keep everything minimalist: few props, lots of empty space, simple shapes. Never realistic or photographic.

Return JSON only. No prose outside JSON. Describe ONLY scene content (pose, location, other white figures,
optional symbol, framing). Do NOT restate the art style or the character's appearance/clothing in your fields —
that is added automatically. Never describe his hair or head, and never call him bald — his look is fixed.
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

RECENT SCENES (make the next ones visually different, especially the LOCATION): {json.dumps(recent_compact, ensure_ascii=False)}

HOW TO CHOOSE THE LOCATION AND PROPS:
- Do NOT pick from a fixed list. YOU decide, from the meaning of current_text, the most natural
  place for the character to be and the object he interacts with. Infer the setting that the line
  itself implies (e.g. a line about buying -> a shop; about scrolling -> looking at a phone;
  about home comfort -> a room; about others judging -> a public place with white figures).
- Invent whatever everyday setting fits best; it does not have to be from any template.
- These framing options are the ONLY suggestions, and they are purely about camera, not content:
  cameras: {json.dumps(cam_pool)}

VOICEOVER BLOCKS:
{json.dumps(build_blocks_payload(blocks), ensure_ascii=False, indent=2)}

PRIMARY TASK: each scene must VISUALLY COMMUNICATE the meaning of its block's spoken text,
grounded in the VIDEO TOPIC, using the main character + (optionally) blank white figures + (optionally) one simple symbol.
Ask yourself: if a viewer watches without sound, will they feel the idea of this exact line?

Return JSON:
{{
  "items": [
    {{
      "index": <block index>,
      "scene_type": "<one from available scene types>",
      "subject": "<the main character's pose/expression, must start with 'the main character'>",
      "others": "<blank white figures present and what they do, or 'none'>",
      "symbol": "<one simple visual metaphor/symbol, or 'no symbol'>",
      "environment": "<a concrete simple location that fits this line and the topic>",
      "action": "<one clear visible action that embodies the line's meaning>",
      "emotion": "<dominant inner state>",
      "camera_framing": "<simple framing, usually full-body at eye level>",
      "reason": "<one sentence: how this scene conveys this line>"
    }}
  ]
}}

Rules:
- Exactly one item per block. All items in English.
- subject must start with "the main character" and describe pose/expression only (do NOT describe clothing or art style).
- Illustrate the SPECIFIC current_text. Show whatever object that line is about (phone, shoes, furniture, advert, crowd, money...).
- Do NOT show a car unless current_text is literally about the car or driving. The car is only the opening/closing example.
- CHANGE the environment across consecutive scenes; place him in locations that fit the line, not always a bare room.
- Other people, when present, must be blank white figures. Use them when the line is about other people; otherwise "none".
- Use a symbol only when it truly helps; otherwise "no symbol".
- Keep environments simple and minimalist (few props, lots of empty space). No realistic, cluttered, or photographic settings.
- Action must embody or metaphorically represent what is spoken in current_text.
- Keep descriptions concise: 4-12 words per field.
""".strip()

    response = api_call(
        lambda: client.chat.completions.create(
            model=PROMPT_MODEL,
            temperature=0.8,
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
    """Assemble the full image prompt: global style + fixed character + scene content + symbol + framing + negatives."""
    subject = clean_text(scene.get("subject", "")) or "the main character standing calmly"
    action = clean_text(scene.get("action", ""))
    others = clean_text(scene.get("others", ""))
    symbol = clean_text(scene.get("symbol", ""))
    environment = clean_text(scene.get("environment", ""))
    emotion = clean_text(scene.get("emotion", ""))
    framing = clean_text(scene.get("camera_framing", ""))

    # Drop "empty" sentinels
    if others.lower() in {"", "none", "no one", "nobody"}:
        others = ""
    if symbol.lower() in {"", "no symbol", "none"}:
        symbol = ""

    parts = [
        ART_STYLE,
        MAIN_CHARACTER,
        ", ".join(p for p in [subject, action] if p),
    ]
    if others:
        parts.append(f"{others}; {OTHERS_STYLE}")
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
    пустое или явно generic. Никакой жёсткой случайной подмены (это раньше ломало
    связь картинки с текстом)."""
    scene_type = clean_text(raw.get("scene_type", "")) or "subject_alone"
    if scene_type not in SCENE_TYPES:
        scene_type = "subject_alone"

    environment = clean_text(raw.get("environment", ""))
    if not environment or is_generic(environment, GENERIC_ENV_PATTERNS):
        environment = pick_fallback(ENVIRONMENT_BANK, recent, "environment", MAX_SAME_ENV_IN_RECENT, salt)

    action = clean_text(raw.get("action", ""))
    if not action or is_generic(action, GENERIC_ACT_PATTERNS):
        action = pick_fallback(ACTION_BANK, recent, "action", 2, salt + 1)

    emotion = clean_text(raw.get("emotion", "")) or pick_fallback(EMOTION_BANK, recent, "emotion", 3, salt + 2)

    camera = clean_text(raw.get("camera_framing", "")) or \
        DEFAULT_CAMERA_BY_TYPE.get(scene_type, "full-body wide shot at eye level, character centered")

    subject = clean_text(raw.get("subject", "")) or "the main character"
    if not subject.lower().startswith("the main character"):
        subject = f"the main character, {subject}"

    others = clean_text(raw.get("others", ""))
    if others.lower() in {"none", "no one", "nobody", ""}:
        others = "none"

    symbol = clean_text(raw.get("symbol", ""))
    if symbol.lower() in {"no symbol", "none", ""}:
        symbol = "no symbol"

    return {
        "scene_type": scene_type,
        "subject": subject,
        "others": others,
        "symbol": symbol,
        "environment": environment,
        "action": action,
        "emotion": emotion,
        "camera_framing": camera,
        "reason": clean_text(raw.get("reason", "")),
    }


FALLBACK_SCENES = [
    {"scene_type": "with_object", "subject": "the main character holding a worn everyday object he keeps",
     "others": "none", "symbol": "no symbol",
     "environment": "simple living room with a sofa and a low table",
     "action": "fondly using a well-worn everyday object",
     "emotion": "calm contentment", "camera_framing": "full-body medium-wide shot showing the character and the object"},
    {"scene_type": "calm_contrast", "subject": "the main character standing calm and still",
     "others": "several blank white figures rush past carrying shopping bags", "symbol": "no symbol",
     "environment": "plain shop interior with simple shelves of products",
     "action": "standing still and content while white figures rush past with shopping bags",
     "emotion": "unbothered ease", "camera_framing": "wide shot, calm character still while figures blur past"},
    {"scene_type": "surrounded_by_choices", "subject": "the main character calmly facing a wall of new products",
     "others": "none", "symbol": "a row of identical shiny new products",
     "environment": "minimal electronics store wall with a row of identical phones",
     "action": "looking at a wall of identical new products without reaching for any",
     "emotion": "indifference to trends", "camera_framing": "wide shot with the character among many objects"},
    {"scene_type": "symbolic_metaphor", "subject": "the main character standing grounded and still",
     "others": "none", "symbol": "a treadmill / hamster-wheel symbol of endless upgrading",
     "environment": "bare cream room with two walls meeting in a corner and a grey-green floor",
     "action": "standing calmly with a relaxed, easy posture",
     "emotion": "settled inner stability", "camera_framing": "medium full-body shot, the floating symbol clearly visible"},
]


def generate_prompts(
    client: OpenAI,
    blocks: list[VisualBlock],
    lang_code: str,
    batch_size: int = 16,
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
            others=scene.get("others", "none"),
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
                  "others", "symbol", "camera_framing", "image_prompt"]
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
                "others": r.others, "symbol": r.symbol,
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
    parser = argparse.ArgumentParser(description="RU/PL/DE voiceover -> timecoded flat-cartoon image prompts (v2)")
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
                             "По умолчанию берётся VIDEO_THEME из окружения или встроенная (про старую машину).")
    parser.add_argument("--character", default=None,
                        help="Переопределить описание персонажа (одной строкой, начинать с 'the main character is '). "
                             "По умолчанию берётся MAIN_CHARACTER_DESC из окружения или встроенный (парень с волосами).")
    parser.add_argument("--force",          action="store_true", help="Regenerate even if outputs exist")
    parser.add_argument("--prompt-workers", type=int, default=DEFAULT_PROMPT_WORKERS,
                        help="Parallel GPT workers (default 4)")
    args = parser.parse_args()

    global MAIN_CHARACTER, VIDEO_THEME
    if args.character and args.character.strip():
        MAIN_CHARACTER = args.character.strip()
    if args.theme and args.theme.strip():
        VIDEO_THEME = args.theme.strip()
    print(f"VIDEO THEME:    {VIDEO_THEME}")
    print(f"MAIN CHARACTER: {MAIN_CHARACTER}")

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
