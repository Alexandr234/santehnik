#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline: RU/EN voiceovers -> Whisper transcription -> image-block SRT -> dynamic 1-3 second visual blocks -> prompts + image timing files.

Default folders:
    Voiceovers:
        /Users/aleksandrtomilov/Desktop/ЯПОНИЯ АВТОМАТИЗАЦИЯ/ОЗВУЧКА

    Prompts and timings:
        /Users/aleksandrtomilov/Desktop/ЯПОНИЯ АВТОМАТИЗАЦИЯ/ПРОМПТЫ

The script is made for 2 language versions of the same Japanese-habits scenario:
    ru = Russian
    en = English

What it creates for each voiceover:
    1) raw Whisper transcript SRT
    2) image-block SRT (each subtitle block = one future image, normally 1-3 seconds)
    3) visual_blocks.json
    4) prompts.txt / prompts.csv / prompts.json
    5) image_times.txt / image_times.csv / image_times.json

Recommended file names in the ОЗВУЧКА folder:
    scenario_ru.mp3, RU.mp3, русский.mp3
    scenario_en.mp3, EN.mp3, english.mp3

Install:
    pip install openai
    brew install ffmpeg   # macOS

Run:
    export OPENAI_API_KEY="sk-..."
    python3 prompt_pipeline.py

Options:
    python3 prompt_pipeline.py --all
    python3 prompt_pipeline.py --target-seconds 2.35
    python3 prompt_pipeline.py --file-ru "/path/RU.mp3" --file-en "/path/EN.mp3"
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
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from openai import OpenAI, APIConnectionError, APITimeoutError, RateLimitError, APIError

# === DEFAULT PATHS ===
DEFAULT_BASE_FOLDER = "/Users/aleksandrtomilov/Desktop/ЯПОНИЯ АВТОМАТИЗАЦИЯ"
DEFAULT_VOICEOVER_FOLDER = f"{DEFAULT_BASE_FOLDER}/ОЗВУЧКА"
DEFAULT_PROMPTS_FOLDER = f"{DEFAULT_BASE_FOLDER}/ПРОМПТЫ"

# === OPENAI API KEY ===
# Paste your key here or use: export OPENAI_API_KEY="sk-..."
OPENAI_API_KEY = ""

TRANSCRIBE_MODEL = os.getenv("TRANSCRIBE_MODEL", "whisper-1")
PROMPT_MODEL = os.getenv("PROMPT_MODEL", "gpt-5.4")


def get_openai_api_key() -> str:
    key_from_script = (OPENAI_API_KEY or "").strip()
    key_from_env = (os.getenv("OPENAI_API_KEY") or "").strip()
    return key_from_script or key_from_env


DEFAULT_TARGET_SECONDS_PER_IMAGE = 2.35
MIN_BLOCK_SECONDS = 1.0
MAX_BLOCK_SECONDS = 3.0
SOFT_MIN_BLOCK_SECONDS = 1.15
SOFT_TARGET_SECONDS = 2.35
MAX_WORDS_PER_IMAGE = 18
DEFAULT_PROMPT_WORKERS = int(os.getenv("PROMPT_WORKERS", "8"))

SUPPORTED_INPUTS = {
    ".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".mp4", ".mov", ".mkv", ".webm"
}

LANGUAGE_ORDER = ["ru", "en"]
LANGUAGES: dict[str, dict[str, Any]] = {
    "ru": {
        "name": "Russian",
        "native_name": "русский",
        "folder": "RU_русский",
        "whisper": "ru",
        "cues": ["_ru", "-ru", " ru", "ru_", "ru-", "рус", "русский", "russian", "rus", "озвучка_ru", "ru озвучка"],
    },
    "en": {
        "name": "English",
        "native_name": "English",
        "folder": "EN_английский",
        "whisper": "en",
        "cues": ["_en", "-en", " en", "en_", "en-", "англ", "английский", "english", "eng", "озвучка_en", "en voice"],
    },
}

# Short style prefix prepended to every generated image prompt.
# Full style description lives in the system prompt — not duplicated here.
STYLE_PREFIX_SHORT = {
    "ru": (
        "Тёплая уютная 2D редакторская иллюстрация в духе премиального ролика об осознанной жизни и японском минимализме: "
        "мягкие плавные формы, тонкий мягкий тёмно-коричневый контур или без жёсткого контура, "
        "аккуратная плоская заливка с мягкими нежными тенями и деликатным тёплым свето-градиентом, "
        "тёплая землистая приглушённая палитра (кремово-бежевый, тёплая охра, терракота, шалфейно-зелёный, тёплое дерево, пыльно-бирюзовый). "
        "Люди — реалистичные, но упрощённые обаятельные персонажи (нормальные пропорции, спокойное лицо, мягкие волнистые волосы), никогда не стикмены; "
        "предметы и растения — узнаваемые мягкие формы с такой же деликатной заливкой и тенью. "
        "Уютный минималистичный интерьер дома в японском духе, мягкий свет, много воздуха. "
        "Спокойно и аккуратно, не неон, не жёсткие чёрные контуры, не фотореализм, не 3D, не аниме"
    ),
    "en": (
        "Warm cozy 2D editorial illustration in the spirit of a premium mindful-living / Japanese-minimalism explainer: "
        "soft rounded shapes, a thin soft dark-brown outline or no hard outline, "
        "clean flat fills with soft delicate shadows and a subtle warm light gradient, "
        "warm earthy muted palette (creamy beige, warm ochre, terracotta, sage green, warm wood, dusty teal). "
        "Humans are realistic but simplified, likeable characters (natural proportions, calm face, soft wavy hair), never stick figures; "
        "objects and plants are recognizable soft shapes with the same delicate fill and shadow. "
        "A cozy minimalist Japanese-inspired home interior, soft light, lots of breathing room. "
        "Calm and clean, not neon, not hard black outlines, not photorealistic, not 3D, not anime"
    ),
}


# Names of famous real people that image generators often refuse or distort.
# scene_prompt is sanitized against this list before building the final prompt.
FAMOUS_NAMES: set[str] = {
    # English / international
    "albert einstein", "einstein", "isaac newton", "newton", "charles darwin", "darwin",
    "galileo galilei", "galileo", "nikola tesla", "tesla", "marie curie", "curie",
    "stephen hawking", "hawking", "richard feynman", "feynman",
    "julius caesar", "caesar", "napoleon", "napoleon bonaparte", "bonaparte",
    "alexander the great", "genghis khan", "attila", "hannibal",
    "cleopatra", "tutankhamun", "ramesses", "ramses",
    "christopher columbus", "columbus", "marco polo", "magellan",
    "leonardo da vinci", "da vinci", "michelangelo", "raphael", "botticelli",
    "plato", "aristotle", "socrates", "pythagoras", "archimedes", "euclid",
    "confucius", "buddha", "zoroaster",
    "abraham lincoln", "lincoln", "george washington", "washington",
    "thomas jefferson", "jefferson", "benjamin franklin", "franklin",
    "winston churchill", "churchill", "franklin roosevelt", "roosevelt",
    "adolf hitler", "hitler", "joseph stalin", "stalin", "mussolini",
    "karl marx", "marx", "friedrich engels", "engels", "vladimir lenin", "lenin",
    "mao zedong", "mao", "che guevara", "guevara", "fidel castro", "castro",
    "nelson mandela", "mandela", "mahatma gandhi", "gandhi",
    "martin luther king", "martin luther",
    "william shakespeare", "shakespeare", "dante alighieri", "dante",
    "homer", "virgil", "cervantes",
    "sigmund freud", "freud", "carl jung", "jung",
    "adam smith", "john maynard keynes", "keynes",
    "elon musk", "musk", "steve jobs", "jeff bezos", "bezos",
    "bill gates", "gates", "mark zuckerberg", "zuckerberg",
    "donald trump", "trump", "joe biden", "biden",
    "barack obama", "obama", "vladimir putin", "putin",
    "angela merkel", "merkel", "emmanuel macron", "macron",
    "queen elizabeth", "king charles", "princess diana", "diana",
    # Russian historical figures (in transliteration — appear in EN prompts)
    "ivan the terrible", "peter the great", "catherine the great",
    "nicholas ii", "alexander ii", "alexander iii",
    "yuri gagarin", "gagarin", "mikhail gorbachev", "gorbachev",
    "boris yeltsin", "yeltsin", "nikita khrushchev", "khrushchev",
    # Кириллица — для RU-промптов
    "эйнштейн", "ньютон", "дарвин", "галилей", "тесла", "кюри",
    "наполеон", "бонапарт", "цезарь", "александр македонский",
    "чингисхан", "аттила", "ганнибал", "клеопатра",
    "колумб", "магеллан", "леонардо да винчи", "да винчи",
    "платон", "аристотель", "сократ", "пифагор", "архимед",
    "маркс", "энгельс", "ленин", "сталин", "гитлер", "муссолини",
    "пётр первый", "пётр i", "пётр великий",
    "екатерина великая", "екатерина вторая",
    "иван грозный", "николай второй", "александр второй",
    "гагарин", "горбачёв", "горбачев", "ельцин", "хрущёв", "хрущев",
    "путин", "трамп", "байден", "обама", "меркель", "макрон",
    "шекспир", "данте", "гомер", "сервантес",
    "фрейд", "юнг", "маркс", "ганди", "мандела",
    "маск", "джобс", "гейтс", "цукерберг",
}


def sanitize_real_names(scene_prompt: str, language_code: str) -> str:
    """Remove real people's names from scene_prompt, replacing with generic role descriptor."""
    if not scene_prompt:
        return scene_prompt

    generic_ru = "исторический деятель"
    generic_en = "historical figure"
    replacement = generic_ru if language_code == "ru" else generic_en

    # Check against known names list (longest first to avoid partial replacements)
    t_low = scene_prompt.lower()
    for name in sorted(FAMOUS_NAMES, key=len, reverse=True):
        if name in t_low:
            scene_prompt = re.sub(re.escape(name), replacement, scene_prompt, flags=re.IGNORECASE)
            t_low = scene_prompt.lower()

    # Catch remaining "Имя Фамилия" patterns (two consecutive Cyrillic capitalised words)
    scene_prompt = re.sub(
        r'\b[А-ЯЁ][а-яё]{2,}\s+[А-ЯЁ][а-яё]{2,}\b',
        replacement,
        scene_prompt,
    )

    # Catch remaining "Firstname Lastname" patterns (two consecutive Latin capitalised words)
    # Guard: skip if it looks like a place name by checking the word after the pattern.
    scene_prompt = re.sub(
        r'\b([A-Z][a-z]{2,})\s+([A-Z][a-z]{2,})\b',
        lambda m: replacement if (m.group(1).lower() + " " + m.group(2).lower()) not in {
            "new york", "los angeles", "san francisco", "las vegas", "new orleans",
            "north america", "south america", "middle east", "red square",
            "black sea", "dead sea", "pacific ocean", "atlantic ocean",
        } else m.group(0),
        scene_prompt,
    )

    return clean_text(scene_prompt)


# ── Dataclasses ─────────────────────────────────────────────────────────────

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
    language_name: str
    start: float
    end: float
    duration: float
    text: str
    frame_type: str   # scene | object | concept | contrast
    scene_prompt: str
    labels: str
    image_prompt: str


@dataclass
class ProcessingJob:
    language_code: str
    input_path: Path


@dataclass
class ContinuityMemory:
    total_frames_generated: int = 0
    non_scene_frames_generated: int = 0
    last_frame_type: str = ""


@dataclass
class SemanticGroup:
    group_id: int
    start_index: int
    end_index: int
    start: float
    end: float
    text: str
    block_indices: list[int]


# ── Utilities ────────────────────────────────────────────────────────────────

def run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    print("$", " ".join(cmd))
    return subprocess.run(cmd, check=check, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise RuntimeError("ffmpeg/ffprobe not found. Install with: brew install ffmpeg")


def clean_text(text: Any) -> str:
    if text is None:
        return ""
    if isinstance(text, str):
        value = text
    elif isinstance(text, (list, tuple, set)):
        value = " ".join(clean_text(item) for item in text if item is not None)
    elif isinstance(text, dict):
        value = " ".join(clean_text(v) for v in text.values() if v is not None)
    else:
        value = str(text)
    return re.sub(r"\s+", " ", value).strip()


def _is_retryable_openai_error(error: Exception) -> bool:
    status_code = getattr(error, "status_code", None)
    if status_code is None:
        return True
    return status_code in {408, 409, 429} or status_code >= 500


def openai_call_with_retries(make_call, description: str, max_retries: int = 5):
    for attempt in range(1, max_retries + 1):
        try:
            return make_call()
        except (APIConnectionError, APITimeoutError, RateLimitError, APIError) as error:
            if isinstance(error, APIError) and not _is_retryable_openai_error(error):
                raise
            if attempt >= max_retries:
                print(f"ERROR: {description} failed after {max_retries} attempts: {error}")
                raise
            delay = min(60, 5 * attempt)
            print(f"WARNING: {description} attempt {attempt}/{max_retries} failed: {error}. Retry in {delay}s...")
            time.sleep(delay)


def word_count(text: str) -> int:
    return len(re.findall(r"[\w']+", text, flags=re.UNICODE))


def safe_stem(path: Path) -> str:
    stem = path.stem.strip()
    stem = re.sub(r"[\\/:*?\"<>|]+", "_", stem)
    stem = re.sub(r"\s+", "_", stem)
    return stem or "audio"


def find_input_files(folder: Path) -> list[Path]:
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_INPUTS]
    files = [p for p in files if "_transcribe_tmp" not in p.name]
    return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)


def normalize_for_language_detection(path: Path) -> str:
    return f" {path.stem.lower()} ".replace(".", " ")


def detect_language_from_filename(path: Path) -> str | None:
    text = normalize_for_language_detection(path)
    for code in LANGUAGE_ORDER:
        for cue in LANGUAGES[code]["cues"]:
            if cue.lower() in text:
                return code
    return None


def seconds_to_srt_time(seconds: float) -> str:
    if seconds < 0:
        seconds = 0
    ms = int(round((seconds - int(seconds)) * 1000))
    whole = int(seconds)
    if ms == 1000:
        whole += 1
        ms = 0
    h = whole // 3600
    m = (whole % 3600) // 60
    s = whole % 60
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def seconds_to_label(seconds: float) -> str:
    whole = int(seconds)
    h = whole // 3600
    m = (whole % 3600) // 60
    s = whole % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def write_srt(segments: list[Segment] | list[VisualBlock], path: Path) -> None:
    lines: list[str] = []
    for i, seg in enumerate(segments, 1):
        text = clean_text(seg.text)
        if not text:
            continue
        lines.append(str(i))
        lines.append(f"{seconds_to_srt_time(seg.start)} --> {seconds_to_srt_time(seg.end)}")
        lines.append(text)
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


# ── Audio / Transcription ────────────────────────────────────────────────────

def get_media_duration(path: Path) -> float:
    result = run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path)
    ])
    return float(result.stdout.strip())


def make_transcription_audio(input_path: Path, work_dir: Path) -> Path:
    require_ffmpeg()
    out = work_dir / f"{safe_stem(input_path)}_transcribe_tmp.mp3"
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(input_path), "-vn", "-ac", "1", "-ar", "16000", "-b:a", "64k", str(out)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return out


def split_audio_if_needed(audio_path: Path, work_dir: Path, chunk_seconds: int = 600, max_mb: int = 24) -> list[Path]:
    if audio_path.stat().st_size / (1024 * 1024) <= max_mb:
        return [audio_path]
    out_dir = work_dir / f"{audio_path.stem}_chunks"
    out_dir.mkdir(exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(audio_path), "-f", "segment", "-segment_time", str(chunk_seconds),
         "-c", "copy", str(out_dir / "chunk_%03d.mp3")],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return sorted(out_dir.glob("chunk_*.mp3"))


def response_to_dict(obj: Any) -> dict[str, Any]:
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    return json.loads(obj.model_dump_json())


def transcribe_audio(client: OpenAI, input_path: Path, output_dir: Path, language_code: str | None) -> list[Segment]:
    tmp_audio = make_transcription_audio(input_path, output_dir)
    chunks = split_audio_if_needed(tmp_audio, output_dir)

    all_segments: list[Segment] = []
    time_offset = 0.0
    whisper_language = LANGUAGES.get(language_code or "", {}).get("whisper")

    for chunk in chunks:
        print(f"Transcribing: {chunk.name} | language={whisper_language or 'auto'}")
        request: dict[str, Any] = {"model": TRANSCRIBE_MODEL, "response_format": "verbose_json", "temperature": 0}
        if whisper_language:
            request["language"] = whisper_language

        def make_call():
            with chunk.open("rb") as f:
                req = dict(request)
                req["file"] = f
                return client.audio.transcriptions.create(**req)

        transcription = openai_call_with_retries(make_call, description=f"transcription of {chunk.name}")
        data = response_to_dict(transcription)
        segs = data.get("segments") or []

        if not segs and data.get("text"):
            duration = get_media_duration(chunk)
            segs = [{"start": 0.0, "end": duration, "text": data["text"]}]

        for seg in segs:
            start = float(seg.get("start", 0)) + time_offset
            end = float(seg.get("end", start + 1)) + time_offset
            text = clean_text(seg.get("text", ""))
            if text and end > start:
                all_segments.append(Segment(start=start, end=end, text=text))

        time_offset += get_media_duration(chunk)

    try:
        tmp_audio.unlink(missing_ok=True)
    except Exception:
        pass

    return all_segments


# ── Visual block building ────────────────────────────────────────────────────

def split_text_into_sentenceish_units(text: str) -> list[str]:
    text = clean_text(text)
    if not text:
        return []
    units = re.split(r"(?<=[.!?…])\s+", text)
    out: list[str] = []
    for unit in units:
        unit = clean_text(unit)
        if not unit:
            continue
        if word_count(unit) > 16:
            parts = re.split(r"(?<=[,;:—-])\s+", unit)
            out.extend(clean_text(p) for p in parts if clean_text(p))
        else:
            out.append(unit)
    return out


def group_words_evenly(words: list[str], groups_count: int) -> list[str]:
    groups_count = max(1, min(groups_count, len(words)))
    out: list[str] = []
    for i in range(groups_count):
        start = round(i * len(words) / groups_count)
        end = round((i + 1) * len(words) / groups_count)
        part = " ".join(words[start:end]).strip()
        if part:
            out.append(part)
    return out


def split_long_segment_for_visuals(seg: Segment, max_seconds: float = MAX_BLOCK_SECONDS) -> list[Segment]:
    duration = max(0.05, seg.end - seg.start)
    text = clean_text(seg.text)
    if not text:
        return []
    if duration <= max_seconds and word_count(text) <= MAX_WORDS_PER_IMAGE:
        return [seg]

    target_count = max(1, math.ceil(duration / max_seconds), math.ceil(word_count(text) / MAX_WORDS_PER_IMAGE))
    sentence_units = split_text_into_sentenceish_units(text)

    if len(sentence_units) >= target_count:
        groups: list[list[str]] = [[] for _ in range(target_count)]
        group_word_counts = [0] * target_count
        for unit in sentence_units:
            avg_words = max(1, math.ceil(word_count(text) / target_count))
            idx = min(len([g for g in groups if g]), target_count - 1)
            while idx < target_count - 1 and group_word_counts[idx] >= avg_words:
                idx += 1
            groups[idx].append(unit)
            group_word_counts[idx] += word_count(unit)
        pieces = [clean_text(" ".join(g)) for g in groups if clean_text(" ".join(g))]
    else:
        pieces = group_words_evenly(text.split(), target_count)

    safe_pieces: list[str] = []
    for piece in pieces:
        if word_count(piece) > MAX_WORDS_PER_IMAGE + 4:
            extra = math.ceil(len(piece.split()) / MAX_WORDS_PER_IMAGE)
            safe_pieces.extend(group_words_evenly(piece.split(), extra))
        else:
            safe_pieces.append(piece)
    pieces = [p for p in safe_pieces if clean_text(p)] or [text]

    total_words = sum(max(1, word_count(p)) for p in pieces)
    out: list[Segment] = []
    cursor = seg.start
    for p in pieces:
        frac = max(1, word_count(p)) / total_words
        end = cursor + duration * frac
        out.append(Segment(start=cursor, end=end, text=clean_text(p)))
        cursor = end
    out[-1].end = seg.end

    final: list[Segment] = []
    for piece_seg in out:
        if piece_seg.end - piece_seg.start <= max_seconds + 0.01:
            final.append(piece_seg)
            continue
        pd = piece_seg.end - piece_seg.start
        words = piece_seg.text.split()
        n = max(1, math.ceil(pd / max_seconds))
        subtexts = group_words_evenly(words, n)
        cursor = piece_seg.start
        for subtext in subtexts:
            sub_end = cursor + pd * (len(subtext.split()) / max(1, len(words)))
            final.append(Segment(start=cursor, end=sub_end, text=clean_text(subtext)))
            cursor = sub_end
        final[-1].end = piece_seg.end
    return final


def normalize_segments_for_visuals(segments: list[Segment]) -> list[Segment]:
    out: list[Segment] = []
    for seg in segments:
        out.extend(split_long_segment_for_visuals(seg))
    return out


def is_strong_visual_break(text: str) -> bool:
    t = clean_text(text).lower()
    markers = [
        "но ", "однако ", "зато ", "вместо этого", "на самом деле", "представьте", "например",
        "первое", "во-первых", "второе", "в-третьих", "главное", "ошибка", "миф", "правда",
        "секрет", "ответ", "теперь", "а вот",
        "but ", "however", "instead", "in fact", "imagine", "for example", "first", "second",
        "third", "the truth", "the answer", "the secret", "myth", "wrong", "now", "yet ",
    ]
    return any(t.startswith(m) for m in markers)


def build_visual_blocks(
    segments: list[Segment],
    target_seconds: float = DEFAULT_TARGET_SECONDS_PER_IMAGE,
) -> list[VisualBlock]:
    target_seconds = max(MIN_BLOCK_SECONDS, min(target_seconds, MAX_BLOCK_SECONDS))
    units = normalize_segments_for_visuals(segments)
    blocks: list[VisualBlock] = []
    cur: list[Segment] = []

    def flush() -> None:
        nonlocal cur
        if not cur:
            return
        text = clean_text(" ".join(s.text for s in cur))
        if text:
            blocks.append(VisualBlock(index=len(blocks) + 1, start=cur[0].start, end=cur[-1].end, text=text))
        cur = []

    for seg in units:
        seg_text = clean_text(seg.text)
        if not seg_text:
            continue
        if not cur:
            cur.append(seg)
            continue

        candidate_duration = seg.end - cur[0].start
        candidate_text = clean_text(" ".join(s.text for s in cur + [seg]))
        current_duration = cur[-1].end - cur[0].start
        current_text = clean_text(" ".join(s.text for s in cur))
        previous_ends_sentence = bool(re.search(r"[.!?…]$", current_text))

        should_flush = (
            candidate_duration > MAX_BLOCK_SECONDS + 0.01
            or (word_count(candidate_text) > MAX_WORDS_PER_IMAGE and current_duration >= MIN_BLOCK_SECONDS)
            or (current_duration >= target_seconds and (previous_ends_sentence or is_strong_visual_break(seg_text)))
            or (current_duration >= SOFT_TARGET_SECONDS and word_count(current_text) >= 8)
        )
        if should_flush:
            flush()
        cur.append(seg)

    flush()

    merged: list[VisualBlock] = []
    for block in blocks:
        if merged:
            prev = merged[-1]
            if (block.end - block.start) < MIN_BLOCK_SECONDS and (block.end - prev.start) <= MAX_BLOCK_SECONDS + 0.01:
                merged[-1] = VisualBlock(
                    index=prev.index, start=prev.start, end=block.end,
                    text=clean_text(prev.text + " " + block.text),
                )
                continue
        merged.append(block)

    hard_fixed: list[VisualBlock] = []
    for block in merged:
        if block.end - block.start <= MAX_BLOCK_SECONDS + 0.01:
            hard_fixed.append(block)
            continue
        pieces = split_long_segment_for_visuals(Segment(block.start, block.end, block.text), max_seconds=MAX_BLOCK_SECONDS)
        for p in pieces:
            hard_fixed.append(VisualBlock(index=0, start=p.start, end=p.end, text=p.text))

    for i, block in enumerate(hard_fixed, 1):
        block.index = i
    return hard_fixed


# ── Semantic grouping ────────────────────────────────────────────────────────

def text_closes_semantic_group(text: str) -> bool:
    t = clean_text(text)
    return bool(t) and t[-1] in ".!?…"


def starts_new_semantic_topic(text: str) -> bool:
    t = clean_text(text).lower()
    starters = [
        "но ", "однако ", "зато ", "теперь ", "так ", "ответ ", "когда ", "результат ",
        "главное ", "и вот ", "а теперь ",
        "but ", "however ", "now ", "the answer ", "when ", "the result ",
    ]
    return any(t.startswith(x) for x in starters)


def build_semantic_groups(
    blocks: list[VisualBlock],
    max_group_seconds: float = 9.0,
    max_group_blocks: int = 6,
) -> list[SemanticGroup]:
    groups: list[SemanticGroup] = []
    current: list[VisualBlock] = []

    def flush() -> None:
        nonlocal current
        if not current:
            return
        gid = len(groups) + 1
        groups.append(SemanticGroup(
            group_id=gid,
            start_index=current[0].index,
            end_index=current[-1].index,
            start=current[0].start,
            end=current[-1].end,
            text=clean_text(" ".join(b.text for b in current)),
            block_indices=[b.index for b in current],
        ))
        current = []

    for b in blocks:
        if current:
            cur_duration = current[-1].end - current[0].start
            if text_closes_semantic_group(current[-1].text) and starts_new_semantic_topic(b.text):
                flush()
            elif cur_duration >= max_group_seconds or len(current) >= max_group_blocks:
                flush()
        current.append(b)
        group_duration = current[-1].end - current[0].start
        if text_closes_semantic_group(b.text) and (group_duration >= 2.0 or len(current) >= 2):
            flush()

    flush()
    return groups


def build_semantic_group_map(groups: list[SemanticGroup]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for group in groups:
        for pos, index in enumerate(group.block_indices):
            if len(group.block_indices) == 1:
                position = "only"
            elif pos == 0:
                position = "first"
            elif pos == len(group.block_indices) - 1:
                position = "last"
            else:
                position = "middle"
            out[index] = {
                "semantic_group_id": group.group_id,
                "semantic_group_text": group.text,
                "semantic_group_position": position,
                "semantic_group_block_indices": group.block_indices,
            }
    return out


def pack_batches_by_semantic_groups(
    blocks: list[VisualBlock],
    groups: list[SemanticGroup],
    batch_size: int,
) -> list[list[VisualBlock]]:
    by_index = {b.index: b for b in blocks}
    batches: list[list[VisualBlock]] = []
    current: list[VisualBlock] = []
    for group in groups:
        group_blocks = [by_index[i] for i in group.block_indices if i in by_index]
        if not group_blocks:
            continue
        if current and len(current) + len(group_blocks) > batch_size:
            batches.append(current)
            current = []
        current.extend(group_blocks)
    if current:
        batches.append(current)
    return batches


# ── Frame type heuristic ─────────────────────────────────────────────────────

def infer_frame_type(current_text: str) -> str:
    """Return recommended frame_type based on text content heuristics.

    Frame types for the calm Japanese-habits style:
      scene    — уютная сюжетная сцена (по умолчанию, ~80%)
      object   — спокойный крупный план ОДНОГО предмета/детали на чистом фоне
      concept  — концептуальная метафора (разум как дом, спокойствие, свет) с мягким тёплым свечением
      contrast — спокойное «до/после»: беспорядок -> порядок, минимализм
    """
    t = (current_text or "").lower()

    # contrast: беспорядок против порядка, избавление от лишнего
    contrast_ru = [
        "беспорядок", "бардак", "хлам", "завал", "загроможд", "лишн", "избавь", "избавит",
        "выброс", "выкинь", "выкид", "расхлам", "меньше вещей", "порядок", "разбер", "уберёшь", "уберешь",
    ]
    contrast_en = [
        "clutter", "mess", "messy", "declutter", "get rid", "throw away", "throw out",
        "tidy", "fewer things", "less stuff", "without ", "no more",
    ]
    if any(x in t for x in contrast_ru) or any(x in t for x in contrast_en):
        return "contrast"

    # concept: метафоры разума/дома, «твой дом это...» (только явные метафоры, не любое спокойное слово)
    concept_ru = [
        "разум", "в голове", "внутри тебя", "внутри нас", "дом это", "дом — это",
        "твой дом", "отражает", "символ", "метафора", "энергия дома", "осознанност",
    ]
    concept_en = [
        "your mind", "in your head", "inside you", "your home is", "reflects",
        "a symbol", "metaphor", "represents", "mindfulness", "state of mind",
    ]
    if any(x in t for x in concept_ru) or any(x in t for x in concept_en):
        return "concept"

    # object: акцент на одном конкретном предмете (кружка, провод, стопка бумаг и т.п.)
    object_ru = [
        "кружка", "чашка", "провод", "кабель", "стопка", "бумаг", "коробка", "ящик",
        "вещь", "предмет", "полка", "чайник", "тарелк", "одна вещь", "один предмет",
    ]
    object_en = [
        "mug", "cup", "cable", "cord", "wire", "stack", "papers", "box", "drawer",
        "object", "item", "shelf", "one thing", "single ",
    ]
    if any(x in t for x in object_ru) or any(x in t for x in object_en):
        return "object"

    return "scene"


# ── Prompt templates ─────────────────────────────────────────────────────────

PROMPT_SYSTEM_BY_LANG = {
    "ru": """
Ты создаёшь промпты для генератора изображений по коротким блокам русской озвучки.
Тема ролика: японские привычки, осознанная жизнь, минимализм, порядок в доме и в голове.

СТИЛЬ (обязателен для каждого кадра):
- Тёплая уютная 2D редакторская иллюстрация в духе премиального ролика об осознанной жизни и японском минимализме (спокойный lo-fi cozy explainer)
- Мягкие плавные формы; тонкий мягкий тёмно-коричневый контур ровной толщины ИЛИ вовсе без жёсткого контура (НЕ жирные чёрные контуры, НЕ каракуль)
- Аккуратная плоская заливка с мягкими нежными тенями и деликатным тёплым свето-градиентом (НЕ жёсткая плоская заливка, но и НЕ фотореализм)
- Тёплая землистая приглушённая палитра: кремово-бежевые и молочные стены, тёплая охра и горчичный, терракота и мягкий ржаво-оранжевый, шалфейный и оливковый зелёный, тёплое дерево, пыльно-бирюзовый и приглушённо-синий акцент (НЕ неон, НЕ кислота)
- Человек — реалистичный, но упрощённый и обаятельный персонаж: нормальные человеческие пропорции, спокойное выразительное лицо, мягкие волнистые волосы, простая уютная одежда (свитер, домашняя одежда). НИКОГДА не стикмен и не мультяшный шарж. По возможности один и тот же повторяющийся герой
- Предметы и растения — узнаваемые, с правильными пропорциями мягкие формы с такой же деликатной заливкой и мягкой тенью
- Уютный минималистичный ФОН — дом в японском духе: чистое спокойное пространство, деревянная мебель, немного предметов, комнатные растения, мягкий естественный свет, много воздуха и свободного места
- Мягкое рассеянное освещение, деликатные тени, спокойная дышащая композиция. Не фотореализм, не 3D, не CGI, не аниме, не манга

ЗАПРЕТ НА ИМЕНА РЕАЛЬНЫХ ЛЮДЕЙ:
- Никогда не упоминай имена реальных людей в scene_prompt
- Вместо имени используй роль: человек, женщина, мужчина, житель, хозяин дома и т.п.

ЧЕТЫРЕ ТИПА КАДРОВ:
1. scene — обычная уютная сюжетная сцена в интерьере дома (по умолчанию, ~80% кадров)
2. object — спокойный крупный план ОДНОГО значимого предмета или детали на чистом, незагромождённом фоне (кружка с отколотой ручкой, спутанные провода, стопка бумаг, растение). Минимум деталей, акцент на одном объекте
3. concept — концептуальная метафора (разум как дом, спокойствие, свет, поток мыслей) с мягким тёплым свечением-акцентом на приглушённом фоне
4. contrast — спокойное «до/после»: беспорядок против порядка, захламлённый угол против того же угла в минимализме. Мягко, БЕЗ жёстких крестов и стрелок

ТЕКСТ ВНУТРИ ИЗОБРАЖЕНИЯ:
- По умолчанию БЕЗ текста (спокойный стиль важнее надписей)
- Если совсем нужно — короткая деликатная подпись кириллицей, аккуратным шрифтом, не кричащая
- labels: максимум 1-2 слова кириллицей; если не нужен — пустая строка

ПРИНЦИП:
- Каждый кадр иллюстрирует ТОЛЬКО текущую фразу, не следующую
- Держи единый уютный мир: один и тот же дом, палитра, герой; меняй только действие, предмет или ракурс
- scene_prompt: 5-15 слов, конкретно ЧТО нарисовать и в каком спокойном окружении, без инструкций по стилю

Верни только валидный JSON.
""".strip(),

    "en": """
You create image prompts for short English voiceover blocks.
Video topic: Japanese habits, mindful living, minimalism, order in the home and in the mind.

MANDATORY STYLE for every frame:
- Warm cozy 2D editorial illustration in the spirit of a premium mindful-living / Japanese-minimalism explainer (calm lo-fi cozy explainer)
- Soft rounded shapes; a thin soft dark-brown outline of even weight OR no hard outline at all (NOT bold black outlines, NOT scribble)
- Clean flat fills with soft delicate shadows and a subtle warm light gradient (NOT harsh flat fill, but NOT photoreal either)
- Warm earthy muted palette: creamy beige and milky walls, warm ochre and mustard, terracotta and soft rust-orange, sage and olive green, warm wood, dusty teal and muted blue accents (NOT neon, NOT acid)
- The human is a realistic but simplified, likeable character: natural human proportions, a calm expressive face, soft wavy hair, simple cozy clothing (sweater, home clothes). NEVER a stick figure and never a cartoon caricature. Prefer the same recurring character
- Objects and plants are recognizable, correctly proportioned soft shapes with the same delicate fill and soft shadow
- A cozy minimalist BACKGROUND — a Japanese-inspired home: clean calm space, wooden furniture, few objects, houseplants, soft natural light, lots of breathing room and negative space
- Soft diffuse lighting, delicate shadows, calm breathable composition. Not photorealistic, not 3D, not CGI, not anime, not manga

NO REAL PERSON NAMES:
- Never mention real people's names in scene_prompt
- Use a role descriptor instead: a person, a woman, a man, a resident, the homeowner, etc.

FOUR FRAME TYPES:
1. scene — a regular cozy narrative scene in a home interior (default, ~80% of frames)
2. object — a calm close-up of ONE meaningful object or detail on a clean, uncluttered background (a chipped mug, tangled cables, a stack of papers, a plant). Minimal detail, focus on the single object
3. concept — a conceptual metaphor (the mind as a home, calm, light, a flow of thoughts) with a soft warm glow accent on a muted background
4. contrast — a calm "before/after": clutter vs order, a cluttered corner vs the same corner minimalist and tidy. Gentle, NO hard X marks or arrows

TEXT INSIDE IMAGE:
- By default NO text (the calm style matters more than captions)
- Only if truly needed — a short, delicate caption in a neat, non-shouting font
- labels: max 1-2 words; if not needed — empty string

PRINCIPLE:
- One frame = only the current phrase, not the next one
- Keep one cozy world: the same home, palette, character; change only the action, object or angle
- scene_prompt: 5-15 words, concretely WHAT to draw and in what calm setting, no style instructions

Return valid JSON only.
""".strip(),
}


PROMPT_USER_TEMPLATE = """
Создай промпт для каждого блока озвучки.
Верни только JSON в точном формате:
{
  "items": [
    {
      "index": 1,
      "frame_type": "scene",
      "scene_prompt": "a woman looking at a drawer of tangled cables, cozy minimalist kitchen",
      "labels": ""
    }
  ]
}

Правила:
- Ровно один элемент на каждый блок
- frame_type: "scene" (по умолчанию), "object" (крупный план одного предмета), "concept" (метафора со свечением), "contrast" (спокойное до/после, порядок против беспорядка)
- scene_prompt: 5-15 слов, конкретно что нарисовать, без инструкций по стилю
- labels: пустая строка если текст внутри не нужен (по умолчанию без текста); если нужен — max 2 слова кириллицей (RU) или по-английски (EN)
- Кадр иллюстрирует ТОЛЬКО текущую фразу, не следующую
- Совет по типу кадра (hint_frame_type) — ориентир, но не обязаловка
- ВСЕ кадры происходят в одном уютном мире из STORY BRIEF: тот же дом, та же палитра, тот же герой, та же атмосфера спокойствия. Не добавляй чужеродные объекты, которых нет в брифе
- Понимай неоднозначные слова по STORY BRIEF, а не буквально (например «сеть» = сетка/шнур в доме, а не интернет; «провод» = кабель зарядки, а не абстракция)
- Держи визуальную непрерывность с соседними блоками (previous_text/next_text/semantic_group_text): если это та же сцена — сохраняй ту же локацию, того же персонажа и окружение, меняй только действие или ракурс

Язык: __LANGUAGE_NAME__ (__LANGUAGE_CODE__)

STORY BRIEF (единый мир всего ролика — обязателен для каждого кадра):
__STORY_BRIEF__

Контекст предыдущего батча:
__CONTINUITY_MEMORY__

Блоки:
__BLOCKS_JSON__
""".strip()


# ── Prompt composition ───────────────────────────────────────────────────────

def label_matches_language(label: str, language_code: str) -> bool:
    label = clean_text(label)
    if not label or len(label.split()) > 3 or len(label) > 18:
        return False
    if language_code == "ru":
        if re.search(r"[A-Za-z]", label):
            return False
        if re.search(r"\d", label):
            return True
        return bool(re.search(r"[А-Яа-яЁё]", label))
    if language_code == "en":
        return not bool(re.search(r"[А-Яа-яЁё]", label))
    return True


def limit_labels_text(labels: str, language_code: str) -> str:
    labels = clean_text(labels)
    if not labels:
        return ""
    parts = [p.strip() for p in re.split(r"[;,|\n]+", labels) if p.strip()]
    safe = [p for p in parts if label_matches_language(p, language_code)]
    return "; ".join(safe[:2])


def compose_final_prompt(
    scene_prompt: str,
    frame_type: str,
    language_code: str,
    labels: str = "",
) -> str:
    scene_prompt = clean_text(scene_prompt).rstrip(". ")
    labels = limit_labels_text(labels, language_code)
    style = STYLE_PREFIX_SHORT.get(language_code, STYLE_PREFIX_SHORT["en"])

    if frame_type == "object":
        if language_code == "ru":
            return (
                f"{style}. "
                f"Спокойный крупный план ОДНОГО предмета или детали по центру, на чистом, незагромождённом "
                f"мягком фоне из землистой палитры, много воздуха вокруг, без персонажей и лишних деталей, "
                f"мягкая деликатная тень под предметом. "
                f"{scene_prompt}."
            )
        return (
            f"{style}. "
            f"A calm close-up of ONE object or detail, centered, on a clean uncluttered soft earthy-colored "
            f"background with lots of breathing room, no characters, no extra detail, "
            f"a soft delicate shadow under the object. "
            f"{scene_prompt}."
        )

    if frame_type == "concept":
        label_part = (
            (f" Деликатная подпись кириллицей: {labels}." if labels else " Без текста внутри изображения.")
            if language_code == "ru"
            else (f" A delicate label: {labels}." if labels else " No text inside the image.")
        )
        if language_code == "ru":
            return (
                f"{style}. "
                f"Концептуальная метафора, одна ясная идея, простая символичная композиция с мягким тёплым "
                f"свечением-акцентом на приглушённом фоне, спокойно и минималистично. "
                f"{scene_prompt}.{label_part}"
            )
        return (
            f"{style}. "
            f"A conceptual metaphor, one clear idea, a simple symbolic composition with a soft warm glow accent "
            f"on a muted background, calm and minimalist. "
            f"{scene_prompt}.{label_part}"
        )

    if frame_type == "contrast":
        label_part = (
            (f" Деликатные подписи кириллицей: {labels}." if labels else " Без текста внутри изображения.")
            if language_code == "ru"
            else (f" Delicate labels: {labels}." if labels else " No text inside the image.")
        )
        if language_code == "ru":
            return (
                f"{style}. "
                f"Спокойное сравнение «до и после»: слева беспорядок и лишние вещи, справа тот же уголок "
                f"в порядке и минимализме, мягко и аккуратно, без жёстких крестов и стрелок. "
                f"{scene_prompt}.{label_part}"
            )
        return (
            f"{style}. "
            f"A calm 'before and after' comparison: clutter and excess on the left, the same corner tidy and "
            f"minimalist on the right, gentle and clean, no hard X marks or arrows. "
            f"{scene_prompt}.{label_part}"
        )

    # Default: scene
    scene_bg = (
        " Уютный минималистичный интерьер/окружение по смыслу сцены, мягкий свет, много воздуха и свободного места."
        if language_code == "ru"
        else " A cozy minimalist interior/setting that fits the scene, soft light, lots of breathing room and negative space."
    )
    no_text = (
        (f" Деликатная подпись кириллицей: {labels}." if labels else " Без текста внутри изображения.")
        if language_code == "ru"
        else (f" A delicate label: {labels}." if labels else " No text inside the image.")
    )
    return f"{style}. {scene_prompt}.{scene_bg}{no_text}"


def fallback_scene_prompt(text: str, language_code: str) -> str:
    text = clean_text(text)
    if language_code == "ru":
        return f"Простая сцена: {text}"
    return f"Simple scene: {text}"


# ── Prompt generation ────────────────────────────────────────────────────────

def build_blocks_payload(
    blocks: list[VisualBlock],
    semantic_group_map: dict[int, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    payload = []
    semantic_group_map = semantic_group_map or {}
    local_by_index = {b.index: b for b in blocks}

    for b in blocks:
        group_meta = semantic_group_map.get(b.index, {})
        group_indices = [idx for idx in group_meta.get("semantic_group_block_indices", []) if idx in local_by_index]
        current_pos = group_indices.index(b.index) if b.index in group_indices else -1

        if current_pos >= 0:
            prev_text = clean_text(" ".join(local_by_index[idx].text for idx in group_indices[max(0, current_pos - 2):current_pos]))
            next_text = clean_text(" ".join(local_by_index[idx].text for idx in group_indices[current_pos + 1:current_pos + 3]))
        else:
            idx = blocks.index(b)
            prev_text = clean_text(" ".join(x.text for x in blocks[max(0, idx - 2):idx]))
            next_text = clean_text(" ".join(x.text for x in blocks[idx + 1:idx + 2]))

        payload.append({
            "index": b.index,
            "start": seconds_to_label(b.start),
            "end": seconds_to_label(b.end),
            "duration": round(b.end - b.start, 2),
            "previous_text": prev_text,
            "current_text": b.text,
            "next_text": next_text,
            "semantic_group_text": group_meta.get("semantic_group_text", ""),
            "hint_frame_type": infer_frame_type(b.text),
        })
    return payload


def continuity_memory_to_text(memory: ContinuityMemory) -> str:
    if memory.total_frames_generated == 0:
        return "none"
    ratio = round(memory.non_scene_frames_generated / memory.total_frames_generated, 3)
    return json.dumps({
        "total_frames": memory.total_frames_generated,
        "non_scene_frames": memory.non_scene_frames_generated,
        "non_scene_ratio": ratio,
        "last_frame_type": memory.last_frame_type,
    }, ensure_ascii=False)


def enforce_non_scene_balance(rows: list[PromptRow], language_code: str) -> list[PromptRow]:
    """Cap non-scene frames at 20% of total."""
    non_scene_indices = [i for i, r in enumerate(rows) if r.frame_type != "scene"]
    max_non_scene = max(1, int(len(rows) * 0.20))
    if len(non_scene_indices) <= max_non_scene:
        return rows
    to_demote = set(non_scene_indices[max_non_scene:])
    new_rows = []
    for i, r in enumerate(rows):
        if i in to_demote:
            new_prompt = fallback_scene_prompt(r.text, language_code)
            new_rows.append(PromptRow(
                index=r.index, language_code=r.language_code, language_name=r.language_name,
                start=r.start, end=r.end, duration=r.duration, text=r.text,
                frame_type="scene", scene_prompt=new_prompt, labels="",
                image_prompt=compose_final_prompt(new_prompt, "scene", language_code),
            ))
        else:
            new_rows.append(r)
    return new_rows


STORY_BRIEF_SYSTEM_BY_LANG = {
    "ru": """
Ты — сценарист-редактор ролика про японские привычки, осознанную жизнь и минимализм.
По полному тексту озвучки составь короткую «библию мира» для художника, который будет рисовать кадры.
Верни ТОЛЬКО валидный JSON:
{
  "era_setting": "место и атмосфера одной короткой строкой (например: современный уютный минималистичный дом в японском духе, спокойный тёплый свет)",
  "world": "как выглядит мир: интерьер, мебель, предметы, растения, свет, палитра (1-2 предложения)",
  "characters": "главный повторяющийся персонаж (роль и внешность в общих чертах, без имён реальных людей)",
  "glossary": [{"word": "слово из текста", "meaning": "что оно значит именно в этом ролике"}],
  "avoid": "что НЕЛЬЗЯ рисовать: неон, кислотные цвета, жёсткие чёрные контуры, стикмены, хаотичные детали и т.п."
}
Мир должен быть единым для всех кадров: тот же дом, та же тёплая приглушённая палитра, тот же спокойный герой.
Особое внимание — неоднозначным словам, которые художник может понять буквально или неверно
(например: «сеть» = сетка или спутанные шнуры в доме, а не интернет; «провод» = кабель зарядки, а не абстракция).
В glossary добавь 3-8 таких слов из текста. Никаких имён реальных людей. Пиши кратко.
""".strip(),
    "en": """
You are a script editor for a video about Japanese habits, mindful living and minimalism.
From the full voiceover text, build a short "world bible" for the artist who will draw the frames.
Return ONLY valid JSON:
{
  "era_setting": "place and mood in one short line (e.g. a modern cozy minimalist Japanese-inspired home, calm warm light)",
  "world": "how the world looks: interior, furniture, objects, plants, light, palette (1-2 sentences)",
  "characters": "the main recurring character (role and rough look, no real person names)",
  "glossary": [{"word": "word from the text", "meaning": "what it means specifically in this video"}],
  "avoid": "what must NOT be drawn: neon, acid colors, hard black outlines, stick figures, chaotic detail, etc."
}
The world must be consistent across all frames: the same home, the same warm muted palette, the same calm character.
Pay special attention to ambiguous words the artist might read literally or wrongly
(e.g. "net" = a mesh or tangled cords in the home, not the internet; "cord" = a charging cable, not an abstraction).
Add 3-8 such words from the text to glossary. No real person names. Be concise.
""".strip(),
}


def build_story_brief(client: OpenAI, blocks: list[VisualBlock], language_code: str) -> dict[str, str]:
    """One-shot 'world bible' for the whole video: keeps every frame in the same era/world and
    disambiguates tricky words (e.g. 'стоянка' = prehistoric campsite, not a car park).
    Returns {"brief": <text injected into every batch>, "setting": <short SCENE_LOCK line>}."""
    full_text = clean_text(" ".join(b.text for b in blocks))
    if not full_text:
        return {"brief": "", "setting": ""}
    full_text = full_text[:12000]

    try:
        response = openai_call_with_retries(
            lambda: client.chat.completions.create(
                model=PROMPT_MODEL,
                temperature=0.2,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": STORY_BRIEF_SYSTEM_BY_LANG.get(language_code, STORY_BRIEF_SYSTEM_BY_LANG["en"])},
                    {"role": "user", "content": f"Полный текст озвучки / Full voiceover text:\n\n{full_text}"},
                ],
            ),
            description=f"story brief for {language_code.upper()}",
        )
        data = json.loads(response.choices[0].message.content or "{}")
    except Exception as error:
        print(f"WARNING: story brief for {language_code.upper()} failed: {error}. Continuing without it.")
        return {"brief": "", "setting": ""}

    era_setting = clean_text(data.get("era_setting", ""))
    world = clean_text(data.get("world", ""))
    characters = clean_text(data.get("characters", ""))
    avoid = clean_text(data.get("avoid", ""))
    glossary_items = data.get("glossary", []) or []
    glossary_pairs = []
    for item in glossary_items:
        if isinstance(item, dict):
            word = clean_text(item.get("word", ""))
            meaning = clean_text(item.get("meaning", ""))
            if word and meaning:
                glossary_pairs.append(f"{word} = {meaning}")
    glossary_text = "; ".join(glossary_pairs)

    is_ru = language_code == "ru"
    lines = []
    if era_setting:
        lines.append((f"- Эпоха/место: {era_setting}" if is_ru else f"- Era/place: {era_setting}"))
    if world:
        lines.append((f"- Мир: {world}" if is_ru else f"- World: {world}"))
    if characters:
        lines.append((f"- Персонажи: {characters}" if is_ru else f"- Characters: {characters}"))
    if glossary_text:
        lines.append((f"- Значения слов: {glossary_text}" if is_ru else f"- Word meanings: {glossary_text}"))
    if avoid:
        lines.append((f"- НЕ рисовать: {avoid}" if is_ru else f"- Do NOT draw: {avoid}"))

    brief = "\n".join(lines) if lines else ("нет" if is_ru else "none")
    print(f"Story brief for {language_code.upper()}: {era_setting or '(empty)'}")
    return {"brief": brief, "setting": era_setting}


def generate_prompts_for_batch(
    client: OpenAI,
    blocks: list[VisualBlock],
    language_code: str,
    memory: ContinuityMemory,
    semantic_group_map: dict[int, dict[str, Any]] | None = None,
    story_brief: str = "",
) -> tuple[list[PromptRow], ContinuityMemory]:
    language = LANGUAGES.get(language_code, LANGUAGES["en"])
    language_name = language.get("name", language_code.upper())
    blocks_payload = build_blocks_payload(blocks, semantic_group_map=semantic_group_map)

    user_prompt = (
        PROMPT_USER_TEMPLATE
        .replace("__LANGUAGE_CODE__", language_code)
        .replace("__LANGUAGE_NAME__", language_name)
        .replace("__STORY_BRIEF__", story_brief or ("нет" if language_code == "ru" else "none"))
        .replace("__CONTINUITY_MEMORY__", continuity_memory_to_text(memory))
        .replace("__BLOCKS_JSON__", json.dumps(blocks_payload, ensure_ascii=False, indent=2))
    )

    response = openai_call_with_retries(
        lambda: client.chat.completions.create(
            model=PROMPT_MODEL,
            temperature=0.35,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": PROMPT_SYSTEM_BY_LANG.get(language_code, PROMPT_SYSTEM_BY_LANG["en"])},
                {"role": "user", "content": user_prompt},
            ],
        ),
        description=f"prompt generation for {language_code.upper()} blocks {blocks[0].index}-{blocks[-1].index}",
    )

    content = response.choices[0].message.content or "{}"
    data = json.loads(content)
    items = data.get("items", [])
    by_index = {int(item["index"]): item for item in items if "index" in item}
    heuristics = {int(p["index"]): p.get("hint_frame_type", "scene") for p in blocks_payload if "index" in p}

    rows: list[PromptRow] = []
    for b in blocks:
        item = by_index.get(b.index, {})
        scene_prompt = clean_text(item.get("scene_prompt", "")) or fallback_scene_prompt(b.text, language_code)
        scene_prompt = sanitize_real_names(scene_prompt, language_code)
        labels = limit_labels_text(clean_text(item.get("labels", "")), language_code)
        frame_type = clean_text(item.get("frame_type", "scene")) or "scene"

        if frame_type not in {"scene", "object", "concept", "contrast"}:
            frame_type = "scene"

        # Apply strong heuristic if model defaulted to scene but hint says object/concept/contrast
        if frame_type == "scene" and heuristics.get(b.index) in {"object", "concept", "contrast"}:
            frame_type = heuristics[b.index]

        rows.append(PromptRow(
            index=b.index,
            language_code=language_code,
            language_name=language_name,
            start=b.start,
            end=b.end,
            duration=b.end - b.start,
            text=b.text,
            frame_type=frame_type,
            scene_prompt=scene_prompt,
            labels=labels,
            image_prompt=compose_final_prompt(scene_prompt, frame_type, language_code, labels),
        ))

    rows = enforce_non_scene_balance(rows, language_code)

    if rows:
        last = rows[-1]
        batch_non_scene = sum(1 for r in rows if r.frame_type != "scene")
        memory = ContinuityMemory(
            total_frames_generated=memory.total_frames_generated + len(rows),
            non_scene_frames_generated=memory.non_scene_frames_generated + batch_non_scene,
            last_frame_type=last.frame_type,
        )
    return rows, memory


def generate_prompts(
    client: OpenAI,
    blocks: list[VisualBlock],
    language_code: str,
    batch_size: int = 18,
    prompt_workers: int = DEFAULT_PROMPT_WORKERS,
    story_brief: str = "",
) -> list[PromptRow]:
    groups = build_semantic_groups(blocks)
    semantic_group_map = build_semantic_group_map(groups)
    batches = pack_batches_by_semantic_groups(blocks, groups, batch_size=batch_size)

    print(f"Semantic groups for {language_code.upper()}: {len(groups)}")
    print(f"Prompt batches for {language_code.upper()}: {len(batches)}")
    print(f"Prompt workers: {prompt_workers}")

    if prompt_workers <= 1:
        rows: list[PromptRow] = []
        memory = ContinuityMemory()
        for batch in batches:
            print(f"Generating {language_code.upper()} prompts {batch[0].index}-{batch[-1].index}...")
            batch_rows, memory = generate_prompts_for_batch(
                client, batch, language_code, memory, semantic_group_map, story_brief,
            )
            rows.extend(batch_rows)
        return sorted(rows, key=lambda r: r.index)

    rows: list[PromptRow] = []
    max_workers = max(1, min(prompt_workers, len(batches)))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for batch in batches:
            print(f"Queue {language_code.upper()} prompts {batch[0].index}-{batch[-1].index}...")
            futures[executor.submit(
                generate_prompts_for_batch, client, batch, language_code, ContinuityMemory(), semantic_group_map, story_brief,
            )] = (batch[0].index, batch[-1].index)
        for future in as_completed(futures):
            s, e = futures[future]
            print(f"Finished {language_code.upper()} prompts {s}-{e}.")
            batch_rows, _ = future.result()
            rows.extend(batch_rows)

    return sorted(rows, key=lambda r: r.index)


# ── File output ──────────────────────────────────────────────────────────────

def write_image_times(rows: list[PromptRow], txt_path: Path, csv_path: Path, json_path: Path) -> None:
    lines: list[str] = []
    items: list[dict[str, Any]] = []
    for r in rows:
        start = seconds_to_srt_time(r.start)
        end = seconds_to_srt_time(r.end)
        lines.append(f"{r.index:03d} | {start} --> {end} | duration: {r.duration:.2f}s | image: {r.index:03d}.png")
        items.append({
            "index": r.index,
            "image_filename": f"{r.index:03d}.png",
            "start": start,
            "end": end,
            "start_seconds": round(r.start, 3),
            "end_seconds": round(r.end, 3),
            "duration": round(r.duration, 3),
            "text": r.text,
        })
    txt_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["index", "image_filename", "start", "end", "start_seconds", "end_seconds", "duration", "text"])
        writer.writeheader()
        for item in items:
            writer.writerow(item)
    json_path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")


def expected_output_paths(input_path: Path, language_code: str, language_dir: Path, root_output_dir: Path) -> dict[str, Path]:
    stem = f"{language_code}_{safe_stem(input_path)}"
    return {
        "stem": Path(stem),
        "raw_srt": language_dir / f"{stem}_raw_whisper.srt",
        "image_srt": language_dir / f"{stem}_image_blocks.srt",
        "blocks_json": language_dir / f"{stem}_visual_blocks.json",
        "prompts_json": language_dir / f"{stem}_prompts.json",
        "prompts_csv": language_dir / f"{stem}_prompts.csv",
        "prompts_txt": language_dir / f"{stem}_prompts.txt",
        "image_times_txt": language_dir / f"{stem}_image_times.txt",
        "image_times_csv": language_dir / f"{stem}_image_times.csv",
        "image_times_json": language_dir / f"{stem}_image_times.json",
        "summary": language_dir / f"{stem}_summary.txt",
    }


def _read_json_list(path: Path) -> list[Any]:
    if not path.exists() or path.stat().st_size <= 0:
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    return data if isinstance(data, list) else []


def ready_prompts_exist(input_path: Path, language_code: str, language_dir: Path, root_output_dir: Path) -> bool:
    paths = expected_output_paths(input_path, language_code, language_dir, root_output_dir)
    required = [paths["prompts_txt"], paths["prompts_json"], paths["prompts_csv"],
                paths["image_times_txt"], paths["image_times_json"], paths["image_srt"]]
    if not all(p.exists() and p.stat().st_size > 0 for p in required):
        return False
    prompts = _read_json_list(paths["prompts_json"])
    image_times = _read_json_list(paths["image_times_json"])
    if not prompts or not image_times or len(prompts) != len(image_times):
        return False
    if [i.get("index") for i in prompts if isinstance(i, dict)] != [i.get("index") for i in image_times if isinstance(i, dict)]:
        return False
    txt = paths["prompts_txt"].read_text(encoding="utf-8", errors="ignore")
    return "001 |" in txt and len(txt.strip()) > 50


def copy_existing_root_outputs(input_path: Path, language_code: str, language_dir: Path, root_output_dir: Path) -> None:
    paths = expected_output_paths(input_path, language_code, language_dir, root_output_dir)
    stem = str(paths["stem"])
    for src, dst in [
        (paths["prompts_txt"], root_output_dir / f"{stem}_prompts.txt"),
        (paths["image_times_txt"], root_output_dir / f"{stem}_image_times.txt"),
        (paths["image_times_json"], root_output_dir / f"{stem}_image_times.json"),
        (paths["image_srt"], root_output_dir / f"{stem}_image_blocks.srt"),
        (paths["prompts_txt"], root_output_dir / f"prompts_{language_code}.txt"),
        (paths["image_times_txt"], root_output_dir / f"image_times_{language_code}.txt"),
        (paths["image_times_json"], root_output_dir / f"image_times_{language_code}.json"),
        (paths["image_srt"], root_output_dir / f"image_blocks_{language_code}.srt"),
    ]:
        if src.exists() and src.stat().st_size > 0:
            shutil.copyfile(src, dst)


def write_outputs(
    input_path: Path,
    language_code: str,
    segments: list[Segment],
    blocks: list[VisualBlock],
    rows: list[PromptRow],
    language_dir: Path,
    root_output_dir: Path,
    target_seconds: float,
    write_root_copies: bool = True,
    story_setting: str = "",
) -> None:
    language_name = LANGUAGES.get(language_code, {}).get("name", language_code.upper())
    paths = expected_output_paths(input_path, language_code, language_dir, root_output_dir)
    stem = str(paths["stem"])

    write_srt(segments, paths["raw_srt"])
    write_srt(blocks, paths["image_srt"])
    paths["blocks_json"].write_text(json.dumps([asdict(b) for b in blocks], ensure_ascii=False, indent=2), encoding="utf-8")
    paths["prompts_json"].write_text(json.dumps([asdict(r) for r in rows], ensure_ascii=False, indent=2), encoding="utf-8")

    with paths["prompts_csv"].open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "index", "language_code", "language_name", "start", "end", "duration",
            "text", "frame_type", "scene_prompt", "labels", "image_prompt",
        ])
        writer.writeheader()
        for r in rows:
            writer.writerow({
                "index": r.index, "language_code": r.language_code, "language_name": r.language_name,
                "start": seconds_to_srt_time(r.start), "end": seconds_to_srt_time(r.end),
                "duration": round(r.duration, 2), "text": r.text, "frame_type": r.frame_type,
                "scene_prompt": r.scene_prompt, "labels": r.labels, "image_prompt": r.image_prompt,
            })

    story_setting = clean_text(story_setting)
    txt_lines: list[str] = []
    for r in rows:
        txt_lines.append(f"{r.index:03d} | {seconds_to_srt_time(r.start)} --> {seconds_to_srt_time(r.end)} | {r.duration:.2f}s | {r.frame_type}")
        txt_lines.append(f"TEXT: {r.text}")
        # SCENE_LOCK carries the whole video's era/world into the image generator,
        # so it keeps the same setting/palette across frames instead of jumping context.
        if story_setting:
            txt_lines.append(f"SCENE_LOCK: {story_setting}")
        if r.labels:
            txt_lines.append(f"LABELS: {r.labels}")
        txt_lines.append(r.image_prompt)
        txt_lines.append("")
    paths["prompts_txt"].write_text("\n".join(txt_lines), encoding="utf-8")

    write_image_times(rows, paths["image_times_txt"], paths["image_times_csv"], paths["image_times_json"])

    if write_root_copies:
        for src, dst in [
            (paths["prompts_txt"], root_output_dir / f"{stem}_prompts.txt"),
            (paths["image_times_txt"], root_output_dir / f"{stem}_image_times.txt"),
            (paths["image_times_json"], root_output_dir / f"{stem}_image_times.json"),
            (paths["image_srt"], root_output_dir / f"{stem}_image_blocks.srt"),
            (paths["prompts_txt"], root_output_dir / f"prompts_{language_code}.txt"),
            (paths["image_times_txt"], root_output_dir / f"image_times_{language_code}.txt"),
            (paths["image_times_json"], root_output_dir / f"image_times_{language_code}.json"),
            (paths["image_srt"], root_output_dir / f"image_blocks_{language_code}.srt"),
        ]:
            shutil.copyfile(src, dst)

        lang_upper = language_code.upper()
        legacy = {
            "RU": [("ru_RU_prompts.txt", "prompts_txt"), ("ru_RU_image_times.txt", "image_times_txt"),
                   ("ru_RU_image_times.json", "image_times_json"), ("ru_RU_image_blocks.srt", "image_srt")],
            "EN": [("en_EN_prompts.txt", "prompts_txt"), ("en_EN_image_times.txt", "image_times_txt"),
                   ("en_EN_image_times.json", "image_times_json"), ("en_EN_image_blocks.srt", "image_srt")],
        }
        for dst_name, src_key in legacy.get(lang_upper, []):
            shutil.copyfile(paths[src_key], root_output_dir / dst_name)

    duration = max((s.end for s in segments), default=0.0)
    theoretical = math.ceil(duration / target_seconds) if duration else len(rows)
    frame_counts = {}
    for r in rows:
        frame_counts[r.frame_type] = frame_counts.get(r.frame_type, 0) + 1
    avg = (duration / len(rows)) if rows else 0
    too_long = [r for r in rows if r.duration > MAX_BLOCK_SECONDS + 0.05]
    too_short = [r for r in rows if r.duration < MIN_BLOCK_SECONDS - 0.05]

    summary = (
        f"Input: {input_path.name}\n"
        f"Language: {language_name} ({language_code})\n"
        f"Duration: {duration:.2f}s ({duration / 60:.2f} min)\n"
        f"Raw Whisper segments: {len(segments)}\n"
        f"Image blocks / prompts: {len(rows)}\n"
        f"Target seconds/image: {target_seconds:.2f} | Max: {MAX_BLOCK_SECONDS:.2f}\n"
        f"Theoretical count at {target_seconds:.2f}s: {theoretical}\n"
        f"Average seconds/image: {avg:.2f}\n"
        f"Blocks >3s: {len(too_long)} | Blocks <1s: {len(too_short)}\n"
        f"Frame types: {frame_counts}\n"
    )
    paths["summary"].write_text(summary, encoding="utf-8")
    print(summary)


# ── Job building & entry point ───────────────────────────────────────────────

def build_jobs(args: argparse.Namespace) -> list[ProcessingJob]:
    voiceover_folder = Path(args.voiceover_folder).expanduser()
    if not voiceover_folder.exists():
        raise RuntimeError(f"Voiceover folder not found: {voiceover_folder}")

    manual_jobs: list[ProcessingJob] = []
    for code in LANGUAGE_ORDER:
        file_value = getattr(args, f"file_{code}")
        if file_value:
            manual_jobs.append(ProcessingJob(language_code=code, input_path=Path(file_value).expanduser()))

    if args.file:
        file_path = Path(args.file).expanduser()
        code = args.language or detect_language_from_filename(file_path) or "unknown"
        return manual_jobs + [ProcessingJob(language_code=code, input_path=file_path)]

    if manual_jobs:
        return manual_jobs

    files = find_input_files(voiceover_folder)
    if not files:
        raise RuntimeError(f"No supported audio/video files found in: {voiceover_folder}")

    if args.all:
        return [ProcessingJob(language_code=detect_language_from_filename(p) or "unknown", input_path=p) for p in files]

    selected: dict[str, Path] = {}
    unknown_files: list[Path] = []
    for file_path in files:
        code = detect_language_from_filename(file_path)
        if code in LANGUAGE_ORDER and code not in selected:
            selected[code] = file_path
        elif code is None:
            unknown_files.append(file_path)

    if not selected and len(files) == 2:
        cyrillic = [p for p in files if re.search(r"[а-яё]", p.stem.lower())]
        latin = [p for p in files if not re.search(r"[а-яё]", p.stem.lower())]
        if len(cyrillic) == 1 and len(latin) == 1:
            selected["ru"] = cyrillic[0]
            selected["en"] = latin[0]
        else:
            sf = sorted(files, key=lambda p: p.name.lower())
            selected["en"], selected["ru"] = sf[0], sf[1]
            print("WARNING: Could not detect language markers. Assigned by filename order: first=EN, second=RU.")

    missing = [code for code in LANGUAGE_ORDER if code not in selected]
    if len(missing) == 1 and len(unknown_files) == 1:
        selected[missing[0]] = unknown_files[0]
        print(f"INFO: Auto-assigned '{unknown_files[0].name}' to language: {missing[0].upper()}")
        missing = []

    if missing:
        print("WARNING: Could not auto-detect files for:", ", ".join(missing))
        for file_path in files:
            print(f"  - {file_path.name} -> {detect_language_from_filename(file_path) or 'unknown'}")
        print("Use filename markers: _ru, _en, RU, EN, or pass --file-ru/--file-en")

    jobs = [ProcessingJob(language_code=code, input_path=selected[code]) for code in LANGUAGE_ORDER if code in selected]
    if not jobs:
        raise RuntimeError("No jobs selected. Use --file-ru/--file-en or rename files with _ru/_en.")
    return jobs


def process_file(
    client: OpenAI,
    job: ProcessingJob,
    root_output_dir: Path,
    target_seconds: float,
    write_root_copies: bool = True,
    force: bool = False,
    prompt_workers: int = DEFAULT_PROMPT_WORKERS,
) -> None:
    input_path = job.input_path.expanduser()
    if not input_path.exists():
        raise RuntimeError(f"Input file not found: {input_path}")

    language_code = job.language_code
    language_dir = root_output_dir / LANGUAGES.get(language_code, {}).get("folder", f"UNKNOWN_{language_code}")
    language_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Processing {input_path.name} | language={language_code.upper()} ===")
    if not force and ready_prompts_exist(input_path, language_code, language_dir, root_output_dir):
        print(f"SKIP: Ready prompts already exist for {input_path.name} ({language_code.upper()}).")
        if write_root_copies:
            copy_existing_root_outputs(input_path, language_code, language_dir, root_output_dir)
        return

    segments = transcribe_audio(client, input_path, language_dir, language_code)
    if not segments:
        raise RuntimeError(f"No transcription segments for {input_path}")

    blocks = build_visual_blocks(segments, target_seconds=target_seconds)
    story = build_story_brief(client, blocks, language_code)
    rows = generate_prompts(
        client, blocks, language_code, prompt_workers=prompt_workers, story_brief=story["brief"],
    )
    write_outputs(
        input_path=input_path, language_code=language_code, segments=segments,
        blocks=blocks, rows=rows, language_dir=language_dir, root_output_dir=root_output_dir,
        target_seconds=target_seconds, write_root_copies=write_root_copies,
        story_setting=story["setting"],
    )


def write_master_summary(root_output_dir: Path, jobs: list[ProcessingJob]) -> None:
    lines = ["RU/EN prompt generation summary", "", "Processed jobs:"]
    for job in jobs:
        name = LANGUAGES.get(job.language_code, {}).get("name", job.language_code.upper())
        lines.append(f"- {name} ({job.language_code}): {job.input_path.name}")
    lines += [
        "", "Root files:",
        "- prompts_ru.txt / image_times_ru.txt / image_times_ru.json / image_blocks_ru.srt",
        "- prompts_en.txt / image_times_en.txt / image_times_en.json / image_blocks_en.srt",
    ]
    (root_output_dir / "_ru_en_history_summary.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transcribe RU/EN voiceovers and generate 1-3 second image prompts with montage timings."
    )
    parser.add_argument("--voiceover-folder", "--folder", default=DEFAULT_VOICEOVER_FOLDER)
    parser.add_argument("--output-folder", default=DEFAULT_PROMPTS_FOLDER)
    parser.add_argument("--all", action="store_true", help="Process all files in voiceover folder.")
    parser.add_argument("--file", default=None, help="Process a single specific file.")
    parser.add_argument("--language", choices=LANGUAGE_ORDER + ["unknown"], default=None)
    parser.add_argument("--file-ru", default=None, help="Russian voiceover file.")
    parser.add_argument("--file-en", default=None, help="English voiceover file.")
    parser.add_argument("--target-seconds", type=float, default=DEFAULT_TARGET_SECONDS_PER_IMAGE,
                        help="Target seconds per image. Recommended: 2.0-2.6. Hard max: 3s.")
    parser.add_argument("--no-root-copies", action="store_true")
    parser.add_argument("--force", action="store_true", help="Regenerate even if outputs exist.")
    parser.add_argument("--prompt-workers", type=int, default=DEFAULT_PROMPT_WORKERS,
                        help="Parallel OpenAI workers. Default: 8. Try 4-12.")
    args = parser.parse_args()

    api_key = get_openai_api_key()
    if not api_key:
        raise RuntimeError(
            "Add your OpenAI API key: OPENAI_API_KEY = 'sk-...' in this script, "
            "or: export OPENAI_API_KEY='sk-...'"
        )

    root_output_dir = Path(args.output_folder).expanduser()
    root_output_dir.mkdir(parents=True, exist_ok=True)

    jobs = build_jobs(args)
    client = OpenAI(api_key=api_key)

    for job in jobs:
        process_file(
            client=client, job=job, root_output_dir=root_output_dir,
            target_seconds=args.target_seconds, write_root_copies=not args.no_root_copies,
            force=args.force, prompt_workers=args.prompt_workers,
        )

    write_master_summary(root_output_dir, jobs)
    print(f"\nDone. Output saved to: {root_output_dir}")


if __name__ == "__main__":
    main()
