#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline: RU / PL / DE voiceovers -> Whisper transcription -> visual blocks with timecodes -> image prompts.

This version generates prompts for a FLAT 2D HAND-DRAWN MINIMALIST CARTOON style
(simple "webcomic / explainer" look) built around ONE recurring main character:

    A bald egg-headed figure with a minimal dot-eyes face,
    wearing an olive-green hoodie and matching green sweatpants.

Everybody else in the frame is drawn as a plain BLANK WHITE faceless humanoid figure
(representing other people / family / society), contrasting with the colored hero.
Scenes are staged in bare rooms (cream walls + corner + grey floor) and lean heavily
on simple visual metaphors (red heart, chains, puppet strings, wavy words, storm cloud).

Default folders:
    Voiceovers:  /Users/aleksandrtomilov/Desktop/ПСИХОЛОГИЯ ГЕРМАНИЯ ПОЛЬША/СКРИПТ ОЗВУЧКИ
    Output:      /Users/aleksandrtomilov/Desktop/ПСИХОЛОГИЯ ГЕРМАНИЯ ПОЛЬША/ПРОМПТЫ

What it creates for each voiceover:
    1) raw Whisper transcript SRT
    2) image-block SRT (each subtitle block = one future image, 1-3 seconds)
    3) visual_blocks.json
    4) prompts.txt / prompts.csv / prompts.json
    5) image_times.txt / image_times.csv / image_times.json
    6) summary.txt

Install:
    pip install openai
    brew install ffmpeg   # macOS

Run:
    python3 psych_prompt_pipeline_gpt54.py
    python3 psych_prompt_pipeline_gpt54.py --all
    python3 psych_prompt_pipeline_gpt54.py --file-ru "/path/RU.mp3"
    python3 psych_prompt_pipeline_gpt54.py --target-seconds 2.35
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
# 2) GLOBAL ART STYLE + CHARACTER SHEET
# ------------------------------------------------------------
# These strings are injected into EVERY final image prompt so that every
# generated frame keeps the exact same flat cartoon look and the exact same
# recurring main character.
# ============================================================
ART_STYLE = (
    "flat 2D hand-drawn cartoon illustration, minimalist webcomic explainer style, "
    "simple clean thin black outlines, flat cel shading with no gradients, "
    "muted earthy color palette of sage green, olive, cream, beige and soft warm grey, "
    "plain bare room shown in simple perspective with two light cream walls meeting in a corner "
    "and a soft grey-green floor, soft oval drop shadow under each figure, "
    "flat even soft lighting, lots of empty negative space, calm restrained quiet mood, "
    "simple and clean, storyboard frame"
)

# The ONE recurring protagonist. Always rendered the same way.
MAIN_CHARACTER = (
    "the main character is a bald egg-headed person with a smooth round head and a very simple "
    "minimal face (two small dot eyes and a small subtle mouth), light plain skin, "
    "wearing an olive-green hooded sweatshirt and matching green sweatpants, drawn in the same flat cartoon style"
)

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
# 3) SCENE DIVERSITY BANKS  (re-themed for the flat cartoon style)
# ============================================================
SCENE_TYPES: dict[str, dict[str, str]] = {
    "alone_centered": {
        "description": "Main character alone, centered in a bare room, full body, quiet and exposed.",
        "template": "main character alone and centered in an empty room",
    },
    "small_in_space": {
        "description": "Main character drawn small inside a large empty room to express isolation.",
        "template": "tiny figure in a big empty room, strong sense of loneliness",
    },
    "surrounded_by_others": {
        "description": "Main character surrounded by several blank white figures (family, crowd, society).",
        "template": "main character encircled by blank white people who press in",
    },
    "confrontation": {
        "description": "A blank white figure points at, leans toward, or accuses the main character.",
        "template": "white figure pointing or accusing, character shrinking back",
    },
    "symbolic_burden": {
        "description": "A physical metaphor object weighs on the character (chains, weight, cage, ropes).",
        "template": "heavy symbolic object physically binding or burdening the character",
    },
    "inner_symbol": {
        "description": "A floating simple symbol near the character shows the hidden inner feeling.",
        "template": "single clear floating symbol expressing the inner state",
    },
    "masking_emotion": {
        "description": "Character holds a forced small smile while a symbol or posture shows the real pain.",
        "template": "fake smile on the outside, visible hidden hurt",
    },
    "being_manipulated": {
        "description": "Blank white figures physically act on the character (chains, pushing, puppet strings).",
        "template": "white figures controlling or restraining the character",
    },
    "seated_hunched": {
        "description": "Character sits hunched on a chair, stool, or floor, shoulders drawn inward.",
        "template": "character seated and hunched, closed-off vulnerable posture",
    },
    "threshold_choice": {
        "description": "Character stands before a single plain door, stairs, or a fork between two paths.",
        "template": "character facing a doorway or a choice between two directions",
    },
    "comparison_split": {
        "description": "Character stands beside a blank white figure to show a contrast between them.",
        "template": "main character next to a blank figure, visible contrast",
    },
    "gesture_reaching": {
        "description": "Hands or figures reach toward or pull away from the character.",
        "template": "reaching hands or figures, connection or rejection through gesture",
    },
}

DEFAULT_CAMERA_BY_TYPE = {
    "alone_centered":       "full-body wide shot at eye level, character centered",
    "small_in_space":       "wide shot with the character drawn small in a large empty frame",
    "surrounded_by_others": "full-body wide shot, character centered among the white figures",
    "confrontation":        "full-body medium-wide shot, slight side angle on both figures",
    "symbolic_burden":      "full-body front-facing shot showing the whole symbolic object",
    "inner_symbol":         "medium full-body shot, the floating symbol clearly visible",
    "masking_emotion":      "medium full-body front-facing shot at eye level",
    "being_manipulated":    "full-body wide shot showing the character and the white figures",
    "seated_hunched":       "medium full-body shot, slight side angle on the seated character",
    "threshold_choice":     "full-body wide shot framing the door or the two paths",
    "comparison_split":     "full-body wide shot with both figures side by side",
    "gesture_reaching":     "full-body medium-wide shot emphasizing the reaching gesture",
}

# Simple, symbolic, deliberately bare settings (the style uses almost-empty rooms).
ENVIRONMENT_BANK = [
    "bare cream room with two walls meeting in a corner and a grey-green floor",
    "empty room with a plain wooden table and simple chairs",
    "bare room with a single plain wooden stool",
    "empty room with one closed plain door",
    "stage-like empty room with a soft round spotlight on the floor",
    "plain corridor suggested by two simple walls",
    "bare room with a single small wooden chair",
    "empty beige room with nothing but the floor shadow",
    "simple room with a low wooden table and one cup",
    "bare room with a window shape drawn on the wall",
    "empty room with a short flight of plain steps",
    "bare room with two simple doors on opposite walls",
    "minimal kitchen suggested by a plain counter and one cup",
    "empty room with a single hanging light bulb",
]

# Simple, readable body language and metaphor actions.
ACTION_BANK = [
    "clasping both hands tightly together in front of the chest",
    "standing stiffly while forcing a small smile",
    "sitting hunched with shoulders drawn inward and head low",
    "standing still while blank white figures laugh around him",
    "being wrapped in heavy chains by two blank white figures",
    "looking down at a small glowing red heart on his chest",
    "holding both hands over his chest where a small heart shows",
    "standing small and alone in the center of a large empty room",
    "shrinking back as a white figure points a finger at him",
    "covering his face with both hands",
    "holding a small mask in front of a sad face",
    "carrying a heavy stone or weight on his shoulders",
    "standing motionless while wavy lines of words hit him from a white figure",
    "reaching one hand toward a white figure that turns away",
    "sitting on the floor hugging his knees",
]

# Framing options (the style is almost always full-body, eye-level, centered).
CAMERA_BANK = [
    "full-body wide shot at eye level, character centered",
    "wide shot with the character drawn small in a large empty frame",
    "full-body medium-wide shot, slight side angle",
    "front-facing full-body staged shot at eye level",
    "wide shot showing the whole simple room and the floor shadow",
    "medium full-body shot focused on the character and one symbol",
    "full-body shot from a slight low angle",
    "wide symmetrical shot with the character in the middle",
]

EMOTION_BANK = [
    "suppressed pain behind a calm face", "forced cheerfulness hiding hurt",
    "quiet loneliness", "helpless compliance", "private shame",
    "guarded vulnerability", "swallowed anger", "tired resignation",
    "fragile hope", "inner conflict", "numb emptiness", "silent endurance",
]

# Simple visual metaphors / symbols that carry the psychological meaning.
SYMBOL_BANK = [
    "a small glowing red heart",
    "a cracked or breaking heart symbol",
    "heavy dark iron chains",
    "thin puppet strings attached to the limbs",
    "wavy lines of harsh words coming from a white figure",
    "a small dark storm cloud above the head",
    "a thin cage or bars drawn around the character",
    "a heavy grey stone or weight",
    "a mask held in one hand",
    "a tangled knot or rope around the chest",
    "a single falling tear",
    "a wall or barrier between two figures",
    "no symbol",
]

DIVERSITY_MEMORY = 12
MAX_SAME_TYPE_IN_RECENT = 2
MAX_SAME_ENV_IN_RECENT  = 1
MAX_SAME_CAM_IN_RECENT  = 2
MAX_SAME_SYM_IN_RECENT  = 1

GENERIC_ENV_PATTERNS = [r"\bgeneric room\b", r"\binterior\b", r"\bspace\b"]
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


def avoid_types(scenes: list[dict]) -> list[str]:
    return [t for t, c in count_recent(scenes, "scene_type").items() if c >= MAX_SAME_TYPE_IN_RECENT]


def pick_non_repeated(current: str, pool: list[str], scenes: list[dict], field: str, max_rep: int, salt: int = 0) -> str:
    import random
    counts = count_recent(scenes, field)
    if current and counts[current.lower()] < max_rep and not is_generic(current, GENERIC_ENV_PATTERNS if field == "environment" else []):
        return current
    rnd = random.Random(salt)
    candidates = pool[:]
    rnd.shuffle(candidates)
    for c in candidates:
        c_clean = clean_text(c)
        if c_clean and counts[c_clean.lower()] < max_rep:
            return c_clean
    return current or (candidates[0] if candidates else "specific place")


def enforce_scene_diversity(scene_type: str, scenes: list[dict], salt: int = 0) -> str:
    import random
    if scene_type not in SCENE_TYPES:
        scene_type = "alone_centered"
    avoid = set(avoid_types(scenes))
    if scene_type not in avoid:
        return scene_type
    all_types = list(SCENE_TYPES.keys())
    rnd = random.Random(salt)
    rnd.shuffle(all_types)
    counts = count_recent(scenes, "scene_type")
    for t in all_types:
        if t not in avoid and counts[t.lower()] < MAX_SAME_TYPE_IN_RECENT:
            return t
    return scene_type


# ============================================================
# 9) PROMPT GENERATION (GPT)
# ============================================================
SCENE_TYPE_REFERENCE = "\n".join(f"- {k}: {v['description']}" for k, v in SCENE_TYPES.items())

SYSTEM_PROMPT = f"""
You plan visual scenes for a psychological short video told in a FLAT 2D HAND-DRAWN MINIMALIST CARTOON style
(simple "webcomic / explainer" look). There is ONE recurring main character in every scene.

GLOBAL ART STYLE (always the same): {ART_STYLE}.
MAIN CHARACTER (always the same): {MAIN_CHARACTER}.
OTHER PEOPLE: {OTHERS_STYLE}.

The visuals use SIMPLE VISUAL METAPHORS and SYMBOLS (red heart, chains, puppet strings, wavy harsh words,
storm cloud, cage, heavy stone, mask) to express the inner feeling described by the voiceover.
Scenes are staged in nearly-empty rooms with very few props. Most shots are full-body and at eye level.

Return JSON only. No prose outside JSON. Describe ONLY the content of each scene
(pose, other white figures, symbol, simple setting, framing). Do NOT restate the art style or the
character's clothing in your fields — that is added automatically. Never invent realistic photographic detail.
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
    avoid = avoid_types(recent_scenes)
    recent_compact = [
        {"scene_type": s.get("scene_type"), "environment": s.get("environment"),
         "action": s.get("action"), "symbol": s.get("symbol"), "camera_framing": s.get("camera_framing")}
        for s in recent_scenes[-8:]
    ]
    rnd = random.Random(len(recent_scenes) * 7 + blocks[0].index)
    env_pool   = rnd.sample(ENVIRONMENT_BANK, min(8, len(ENVIRONMENT_BANK)))
    act_pool   = rnd.sample(ACTION_BANK, min(8, len(ACTION_BANK)))
    cam_pool   = rnd.sample(CAMERA_BANK, min(6, len(CAMERA_BANK)))
    emo_pool   = rnd.sample(EMOTION_BANK, min(6, len(EMOTION_BANK)))
    sym_pool   = rnd.sample(SYMBOL_BANK, min(7, len(SYMBOL_BANK)))

    user_prompt = f"""
LANGUAGE OF VOICEOVER: {lang_name} ({lang_code})
(All scene fields must be in English. The voiceover text is in {lang_name}.)

AVAILABLE SCENE TYPES:
{SCENE_TYPE_REFERENCE}

SCENE TYPES TO AVOID (overused recently): {json.dumps(avoid)}
RECENT SCENES (avoid repeating): {json.dumps(recent_compact, ensure_ascii=False)}

VARIATION POOL:
  environments: {json.dumps(env_pool)}
  actions: {json.dumps(act_pool)}
  cameras: {json.dumps(cam_pool)}
  emotions: {json.dumps(emo_pool)}
  symbols: {json.dumps(sym_pool)}

VOICEOVER BLOCKS:
{json.dumps(build_blocks_payload(blocks), ensure_ascii=False, indent=2)}

PRIMARY TASK: each scene must VISUALLY COMMUNICATE the meaning of its block's spoken text
in the flat cartoon style, using the main character + (optionally) blank white figures + (optionally) one simple symbol.
Ask yourself: if a viewer watches without sound, will they feel the idea of the text?

Return JSON:
{{
  "items": [
    {{
      "index": <block index>,
      "scene_type": "<one from available scene types>",
      "subject": "<the main character's pose/expression, must start with 'the main character'>",
      "others": "<blank white figures present and what they do, or 'none'>",
      "symbol": "<one simple visual metaphor/symbol from the symbol pool, or 'no symbol'>",
      "environment": "<simple nearly-empty staged setting, not realistic or detailed>",
      "action": "<one clear visible action that embodies the text meaning>",
      "emotion": "<dominant inner state>",
      "camera_framing": "<simple framing, usually full-body at eye level>",
      "reason": "<one sentence: how this scene conveys the text meaning>"
    }}
  ]
}}

Rules:
- Exactly one item per block. All items in English.
- subject must start with "the main character" and describe pose/expression only (do NOT describe clothing or art style).
- Other people, when present, must be blank white figures (family, crowd, accuser, manipulator). Use them when the text is about other people; otherwise set others to "none".
- Prefer a simple SYMBOL when the text is about a hidden feeling, pressure, or being controlled. Use "no symbol" for plain solo moments.
- Keep environments simple and nearly empty (this is a minimalist style). No realistic, cluttered, or photographic settings.
- Action must embody or metaphorically represent what is spoken in current_text.
- Each scene should be visually different from the recent scenes.
- Keep descriptions concise: 4-12 words per field.
""".strip()

    response = api_call(
        lambda: client.chat.completions.create(
            model=PROMPT_MODEL,
            temperature=0.75,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
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
    subject = clean_text(scene.get("subject", "")) or "the main character standing still"
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
    parts.append(environment)
    parts.append(emotion)
    parts.append(framing)
    parts.append(NEGATIVE_SUFFIX)

    prompt = ". ".join(p for p in parts if p)
    return re.sub(r"\s+", " ", prompt).strip()


def normalize_scene(raw: dict, b: VisualBlock, recent: list[dict], salt: int) -> dict:
    scene_type = enforce_scene_diversity(clean_text(raw.get("scene_type", "")), recent, salt)
    environment = pick_non_repeated(
        clean_text(raw.get("environment", "")), ENVIRONMENT_BANK, recent, "environment", MAX_SAME_ENV_IN_RECENT, salt)
    action = pick_non_repeated(
        clean_text(raw.get("action", "")), ACTION_BANK, recent, "action", 1, salt + 1)
    emotion = clean_text(raw.get("emotion", "")) or pick_non_repeated("", EMOTION_BANK, recent, "emotion", 2, salt + 2)
    camera = pick_non_repeated(
        clean_text(raw.get("camera_framing", "")) or DEFAULT_CAMERA_BY_TYPE.get(scene_type, "full-body wide shot at eye level"),
        CAMERA_BANK, recent, "camera_framing", MAX_SAME_CAM_IN_RECENT, salt + 3)

    subject = clean_text(raw.get("subject", "")) or "the main character"
    if not subject.lower().startswith("the main character"):
        subject = f"the main character, {subject}"

    others = clean_text(raw.get("others", ""))
    if others.lower() in {"none", "no one", "nobody", ""}:
        others = "none"

    # Symbol with light anti-repetition (kept optional)
    symbol = clean_text(raw.get("symbol", ""))
    if symbol.lower() in {"no symbol", "none", ""}:
        symbol = "no symbol"
    else:
        sym_counts = count_recent(recent, "symbol")
        if sym_counts[symbol.lower()] >= MAX_SAME_SYM_IN_RECENT:
            symbol = pick_non_repeated(symbol, SYMBOL_BANK, recent, "symbol", MAX_SAME_SYM_IN_RECENT, salt + 4)

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
    {"scene_type": "alone_centered", "subject": "the main character standing still with a quiet face",
     "others": "none", "symbol": "no symbol",
     "environment": "bare cream room with two walls meeting in a corner and a grey-green floor",
     "action": "clasping both hands tightly together in front of the chest",
     "emotion": "quiet loneliness", "camera_framing": "full-body wide shot at eye level, character centered"},
    {"scene_type": "surrounded_by_others", "subject": "the main character standing in the middle",
     "others": "several blank white figures crowd in and laugh", "symbol": "no symbol",
     "environment": "empty room with a plain wooden table and simple chairs",
     "action": "standing stiffly while forcing a small smile",
     "emotion": "forced cheerfulness hiding hurt", "camera_framing": "full-body wide shot, character centered among the white figures"},
    {"scene_type": "inner_symbol", "subject": "the main character looking down at his chest",
     "others": "none", "symbol": "a small glowing red heart",
     "environment": "bare beige room with nothing but the floor shadow",
     "action": "holding both hands over his chest where a small heart shows",
     "emotion": "suppressed pain behind a calm face", "camera_framing": "medium full-body shot, the floating symbol clearly visible"},
    {"scene_type": "being_manipulated", "subject": "the main character standing passively",
     "others": "two blank white figures wrap him in chains", "symbol": "heavy dark iron chains",
     "environment": "stage-like empty room with a soft round spotlight on the floor",
     "action": "being wrapped in heavy chains by two blank white figures",
     "emotion": "helpless compliance", "camera_framing": "full-body wide shot showing the character and the white figures"},
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
    avg = dur / len(rows) if rows else 0
    summary = (
        f"Input: {input_path.name}\n"
        f"Language: {LANGUAGES.get(lang_code, {}).get('name', lang_code)} ({lang_code})\n"
        f"Duration: {dur:.2f}s ({dur/60:.2f} min)\n"
        f"Whisper segments: {len(segments)}\n"
        f"Image blocks / prompts: {len(rows)}\n"
        f"Target seconds/image: {target:.2f} | Max: {MAX_BLOCK_SECONDS:.2f}\n"
        f"Average seconds/image: {avg:.2f}\n"
        f"Scene type distribution: {dict(sc_counts.most_common())}\n"
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
    parser = argparse.ArgumentParser(description="RU/PL/DE voiceover -> timecoded flat-cartoon image prompts")
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
    parser.add_argument("--force",          action="store_true", help="Regenerate even if outputs exist")
    parser.add_argument("--prompt-workers", type=int, default=DEFAULT_PROMPT_WORKERS,
                        help="Parallel GPT workers (default 4)")
    args = parser.parse_args()

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
