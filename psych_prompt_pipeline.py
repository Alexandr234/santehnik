#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline: RU / PL / DE voiceovers -> Whisper transcription -> visual blocks with timecodes -> diverse image prompts.

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
    python3 psych_prompt_pipeline.py
    python3 psych_prompt_pipeline.py --all
    python3 psych_prompt_pipeline.py --file-ru "/path/RU.mp3"
    python3 psych_prompt_pipeline.py --target-seconds 2.35
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
PROMPT_MODEL     = os.getenv("PROMPT_MODEL", "gpt-4.5")

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
# 2) SCENE DIVERSITY BANKS
# ============================================================
SCENE_TYPES: dict[str, dict[str, str]] = {
    "inner_portrait": {
        "description": "Close or medium-close emotional portrait of one character.",
        "template": "emotion-focused portrait, subtle facial micro-expressions",
    },
    "body_language": {
        "description": "Character reveals state through posture and gesture.",
        "template": "expressive posture, restrained movement, clear emotional readability",
    },
    "room_interaction": {
        "description": "Character interacts with a meaningful object in a grounded interior.",
        "template": "meaningful object contact, psychological realism",
    },
    "walking_reflection": {
        "description": "Character walks, stops, or pauses with emotional intent.",
        "template": "reflective movement, walking or pausing with emotional intent",
    },
    "window_moment": {
        "description": "Character near a window, balcony, or outside light.",
        "template": "inward emotion contrasted with outside space, contemplative stillness",
    },
    "symbolic_object": {
        "description": "Character and one meaningful object connected to the theme.",
        "template": "object and gesture carry psychological meaning, tactile detail",
    },
    "domestic_ritual": {
        "description": "Ordinary domestic routine reveals inner pressure.",
        "template": "ordinary action revealing emotional pressure",
    },
    "threshold_moment": {
        "description": "Doorway, stairs, or exit as a decision point.",
        "template": "decision point shown through space and posture",
    },
    "public_isolation": {
        "description": "Character alone in an empty public or semi-public space.",
        "template": "empty public space emphasizing inner distance",
    },
    "close_detail_action": {
        "description": "Tight detail of hands, face, shoulders, or breathing.",
        "template": "tight psychological detail, hands face breath or object action",
    },
    "environmental_pressure": {
        "description": "Space visually presses on the character.",
        "template": "space composition expresses psychological tension",
    },
    "object_decision": {
        "description": "Character chooses, hides, or releases an object.",
        "template": "clear choice expressed through gesture and object",
    },
}

DEFAULT_CAMERA_BY_TYPE = {
    "inner_portrait":         "medium close-up at eye level",
    "body_language":          "medium shot with full upper-body posture visible",
    "room_interaction":       "medium shot close to the action",
    "walking_reflection":     "tracking medium shot or side profile walk",
    "window_moment":          "three-quarter medium shot near the window",
    "symbolic_object":        "medium close-up with object in hands or foreground",
    "domestic_ritual":        "medium side-profile shot close to the domestic action",
    "threshold_moment":       "wide shot framed through a doorway or corridor",
    "public_isolation":       "wide shot with the character small in the frame",
    "close_detail_action":    "tight close-up on hands face shoulders or object",
    "environmental_pressure": "wide or high-angle shot emphasizing the surrounding space",
    "object_decision":        "over-the-shoulder shot focused on the object and hands",
}

ENVIRONMENT_BANK = [
    "small kitchen at night with a sink and one cup",
    "narrow hallway with shoes and a half-open door",
    "bathroom sink with fogged mirror",
    "quiet bedroom with an unmade bed",
    "empty stairwell between floors",
    "parked car interior in a quiet parking lot",
    "office desk after everyone has left",
    "balcony with railings and distant city lights",
    "empty train platform",
    "messy living room with scattered papers",
    "dim elevator with reflective walls",
    "plain waiting room with one chair and a wall clock",
    "bus stop shelter after rain with wet pavement",
    "long corridor with numbered doors",
    "kitchen table covered with unopened letters",
]

ACTION_BANK = [
    "folding and unfolding a small note",
    "washing the same cup longer than necessary",
    "standing still with keys in hand",
    "slowly packing a bag and then stopping",
    "opening a drawer and closing it without taking anything",
    "holding a cup with both hands without drinking",
    "leaning against a door after closing it",
    "turning a phone face down on the table",
    "straightening objects on a table too carefully",
    "stopping in front of a door but not knocking",
    "pressing both palms on the sink edge",
    "holding a coat but not putting it on",
    "sitting inside a parked car with hands on the steering wheel",
    "counting coins or small objects mechanically",
]

CAMERA_BANK = [
    "wide shot with the character small in the frame",
    "medium side-profile shot",
    "over-the-shoulder shot focused on the object",
    "static frontal medium shot",
    "tight close-up on hands and lower face",
    "high angle showing the character isolated in space",
    "medium shot from behind the character",
    "close profile shot with background depth",
]

EMOTION_BANK = [
    "suppressed anxiety", "quiet resistance", "painful hesitation",
    "emotional numbness", "private shame", "guarded vulnerability",
    "controlled anger", "tired acceptance", "fragile hope",
    "inner conflict", "reluctant release", "loneliness without drama",
]

DIVERSITY_MEMORY = 12
MAX_SAME_TYPE_IN_RECENT = 2
MAX_SAME_ENV_IN_RECENT  = 1
MAX_SAME_CAM_IN_RECENT  = 2

GENERIC_ENV_PATTERNS = [r"\broom\b", r"\binterior\b", r"\bspace\b", r"\bsimple interior\b"]
GENERIC_ACT_PATTERNS = [r"holding still", r"looking away", r"standing quietly", r"sitting quietly"]

# ============================================================
# 3) DATACLASSES
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
    camera_framing: str
    image_prompt: str


@dataclass
class ProcessingJob:
    language_code: str
    input_path: Path


# ============================================================
# 4) UTILITIES
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
# 5) AUDIO / TRANSCRIPTION
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
# 6) VISUAL BLOCK BUILDING
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
# 7) DIVERSITY HELPERS
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
        scene_type = "inner_portrait"
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
# 8) PROMPT GENERATION (GPT)
# ============================================================
SCENE_TYPE_REFERENCE = "\n".join(f"- {k}: {v['description']}" for k, v in SCENE_TYPES.items())

SYSTEM_PROMPT = """
You plan visual scenes for a psychological video. ONE recurring character throughout.
Return JSON only. No prose outside JSON.
No style words: no cinematic, film grain, HDR, painting, anime, hyperrealistic, artist names, lens brand.
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
         "action": s.get("action"), "camera_framing": s.get("camera_framing")}
        for s in recent_scenes[-8:]
    ]
    rnd = random.Random(len(recent_scenes) * 7 + blocks[0].index)
    env_pool   = rnd.sample(ENVIRONMENT_BANK, min(8, len(ENVIRONMENT_BANK)))
    act_pool   = rnd.sample(ACTION_BANK, min(8, len(ACTION_BANK)))
    cam_pool   = rnd.sample(CAMERA_BANK, min(6, len(CAMERA_BANK)))
    emo_pool   = rnd.sample(EMOTION_BANK, min(6, len(EMOTION_BANK)))

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

VOICEOVER BLOCKS:
{json.dumps(build_blocks_payload(blocks), ensure_ascii=False, indent=2)}

PRIMARY TASK: each scene must VISUALLY COMMUNICATE the meaning of its block's spoken text.
Ask yourself: if a viewer watches without sound, will they feel the idea of the text?

Return JSON:
{{
  "items": [
    {{
      "index": <block index>,
      "scene_type": "<one from available scene types>",
      "subject": "<what is visible, must start with 'this character'>",
      "environment": "<specific real place, not generic 'room'>",
      "action": "<one clear visible action that embodies the text meaning>",
      "emotion": "<dominant inner state>",
      "camera_framing": "<specific framing>",
      "reason": "<one sentence: how this scene conveys the text meaning>"
    }}
  ]
}}

Rules:
- Exactly one item per block. All items in English.
- subject must start with "this character".
- One character only, no crowds, no extra people.
- No generic environments (room, interior). No generic actions (looking away, standing quietly).
- Each scene physically different from recent scenes.
- Action must embody or metaphorically represent what is spoken in current_text.
- Keep descriptions concise: 5-12 words per field.
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
    parts = [
        clean_text(scene.get("subject", "")),
        clean_text(scene.get("action", "")),
        clean_text(scene.get("environment", "")),
        clean_text(scene.get("emotion", "")),
        clean_text(scene.get("camera_framing", "")),
        "single character only, no extra people, no crowd, no text, no watermark",
    ]
    prompt = ", ".join(p for p in parts if p)
    return re.sub(r"\s+", " ", prompt).strip()


def normalize_scene(raw: dict, b: VisualBlock, recent: list[dict], salt: int) -> dict:
    scene_type = enforce_scene_diversity(clean_text(raw.get("scene_type", "")), recent, salt)
    environment = pick_non_repeated(
        clean_text(raw.get("environment", "")), ENVIRONMENT_BANK, recent, "environment", MAX_SAME_ENV_IN_RECENT, salt)
    action = pick_non_repeated(
        clean_text(raw.get("action", "")), ACTION_BANK, recent, "action", 1, salt + 1)
    emotion = clean_text(raw.get("emotion", "")) or pick_non_repeated("", EMOTION_BANK, recent, "emotion", 2, salt + 2)
    camera = pick_non_repeated(
        clean_text(raw.get("camera_framing", "")) or DEFAULT_CAMERA_BY_TYPE.get(scene_type, "medium close-up"),
        CAMERA_BANK, recent, "camera_framing", MAX_SAME_CAM_IN_RECENT, salt + 3)
    subject = clean_text(raw.get("subject", "")) or "this character"
    if not subject.lower().startswith("this character"):
        subject = f"this character, {subject}"
    return {
        "scene_type": scene_type,
        "subject": subject,
        "environment": environment,
        "action": action,
        "emotion": emotion,
        "camera_framing": camera,
        "reason": clean_text(raw.get("reason", "")),
    }


FALLBACK_SCENES = [
    {"scene_type": "domestic_ritual", "subject": "this character beside a kitchen sink",
     "environment": "small kitchen at night with a sink and one cup",
     "action": "washing the same cup longer than necessary",
     "emotion": "suppressed anxiety", "camera_framing": "medium side-profile shot"},
    {"scene_type": "threshold_moment", "subject": "this character at a half-open door",
     "environment": "narrow hallway with shoes and a half-open door",
     "action": "holding the door handle but not leaving",
     "emotion": "painful hesitation", "camera_framing": "wide shot framed through a doorway"},
    {"scene_type": "object_decision", "subject": "this character at a desk",
     "environment": "office desk after everyone has left",
     "action": "placing a small object into a drawer and closing it slowly",
     "emotion": "tired acceptance", "camera_framing": "over-the-shoulder shot focused on the object"},
    {"scene_type": "public_isolation", "subject": "this character under a bus stop shelter",
     "environment": "bus stop shelter after rain with wet pavement",
     "action": "holding a coat closed while looking at the empty street",
     "emotion": "loneliness without drama", "camera_framing": "wide shot with the character small in the frame"},
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
            camera_framing=scene.get("camera_framing", ""),
            image_prompt=build_final_prompt(scene),
        ))
    return rows


# ============================================================
# 9) OUTPUT WRITERS
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
                  "text", "scene_type", "environment", "action", "emotion", "camera_framing", "image_prompt"]
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
# 10) JOB DETECTION
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
# 11) MAIN
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
    parser = argparse.ArgumentParser(description="RU/PL/DE voiceover -> timecoded diverse image prompts")
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
