#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline: 4 language voiceovers -> Whisper transcription -> SRT -> visual blocks -> image prompts + image timing files.

УНИВЕРСАЛЬНАЯ ВЕРСИЯ (ЛЮБАЯ ТЕМА ДОКУМЕНТАЛКИ).
Главное отличие от старой версии ("ПТИЦЫ РЕАЛИСТИЧНЫЕ"):
    - скрипт БОЛЬШЕ НЕ ЗАШИТ под птиц/океан. Тема ролика определяется АВТОМАТИЧЕСКИ
      из текста озвучки (шелкопряд, птицы, история, техника, растения — что угодно);
    - перед генерацией сцен строится краткий "визуальный бриф" (главный герой, эпоха/место,
      ключевые объекты), чтобы весь ролик держался одной темы, как настоящая документалка;
    - стиль остаётся ФОТОРЕАЛИСТИЧНЫЙ кинематографичный документальный (BBC Earth /
      National Geographic), но нейтральный по содержанию — он подходит под любой сюжет;
    - один визуал примерно на каждые ~6.5 секунд озвучки.

ДОКУМЕНТАЛЬНЫЕ ПАУЗЫ (новое):
    - как в настоящем документальном фильме, МЕЖДУ ФРАЗАМИ диктора вставляются ПАУЗЫ,
      и на каждую паузу генерируется отдельный B-ROLL кадр с животными/природой
      (перебивка без слов: только картинка и живые звуки природы);
    - пауза ставится ТОЛЬКО после завершённого предложения (. ! ? …): если визуальный
      блок оборвался посреди мысли, речь продолжается без разрыва, а пауза уезжает
      к концу ближайшего законченного предложения — никаких «обрывистых» пауз;
    - итоговые файлы таймингов содержат РАСТЯНУТУЮ временную шкалу:
        * строки `type: speech` — фразы диктора, с полем `src: ... --> ...`
          (положение этой фразы в ИСХОДНОЙ озвучке);
        * строки `type: broll` — паузы-перебивки (в это время диктор молчит);
    - финальный сборщик видео читает эти поля, режет озвучку по фразам и вставляет
      тишину на время b-roll перебивок (см. video_creator_times_autovenv_no_subs_realistic.py);
    - управление: --no-pauses (отключить), --pause-seconds 2.2 (средняя длина паузы),
      --pause-every 1 (пауза после каждой N-й фразы).

ЗВУК (новое):
    - в КАЖДЫЙ финальный промпт добавляется аудио-описание: в кадре могут быть ТОЛЬКО
      оригинальные звуки природы (ветер, вода, крики животных на экране), СТРОГО без
      человеческой речи, без закадрового голоса и без музыки. Это описание уходит и в
      генерацию видео, чтобы клипы рождались сразу с правильным натуральным звуком.

Default folders (можно переопределить env-переменной BASE_FOLDER или аргументами):
    Voiceovers:
        <BASE_FOLDER>/ОЗВУЧКА
    Prompts and timings:
        <BASE_FOLDER>/ПРОМПТЫ

The script is made for 4 language versions of the same scenario:
    ru = Russian
    de = German
    es = Spanish
    pl = Polish

Important:
    Timings are calculated separately from each language voiceover.
    This means image blocks in Russian, German, Spanish, and Polish can have different start/end times,
    because the same phrase can be longer or shorter in different languages.

Recommended file names in the ОЗВУЧКА folder:
    scenario_ru.mp3
    scenario_de.mp3
    scenario_es.mp3
    scenario_pl.mp3

Install:
    pip install openai
    brew install ffmpeg   # macOS, if ffmpeg is not installed

Run:
    export OPENAI_API_KEY="sk-..."
    # опционально задать корневую папку проекта (в ней ищутся ОЗВУЧКА и ПРОМПТЫ):
    export BASE_FOLDER="/Users/aleksandrtomilov/Desktop/ШЕЛКОПРЯД"
    python3 doc_prompt_pipeline_universal_ru_de_es_pl.py

Useful options:
    python3 doc_prompt_pipeline_universal_ru_de_es_pl.py --all
    python3 doc_prompt_pipeline_universal_ru_de_es_pl.py --target-seconds 6.5
    python3 doc_prompt_pipeline_universal_ru_de_es_pl.py --voiceover-folder "/path/ОЗВУЧКА" --output-folder "/path/ПРОМПТЫ"
    python3 doc_prompt_pipeline_universal_ru_de_es_pl.py --file-ru "/path/ru.mp3" --file-de "/path/de.mp3" --file-es "/path/es.mp3" --file-pl "/path/pl.mp3"
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
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from openai import OpenAI

# === DEFAULT PATHS ===
# Корневую папку проекта можно задать через BASE_FOLDER; тема ролика при этом любая.
DEFAULT_BASE_FOLDER = os.getenv("BASE_FOLDER", "/Users/aleksandrtomilov/Desktop/ПТИЦЫ РЕАЛИСТИЧНЫЕ")
DEFAULT_VOICEOVER_FOLDER = f"{DEFAULT_BASE_FOLDER}/ОЗВУЧКА"
DEFAULT_PROMPTS_FOLDER = f"{DEFAULT_BASE_FOLDER}/ПРОМПТЫ"

TRANSCRIBE_MODEL = os.getenv("TRANSCRIBE_MODEL", "whisper-1")
PROMPT_MODEL = os.getenv("PROMPT_MODEL", "gpt-4o-mini")

# Practical default: one image every ~6.5 seconds (reference video cuts every ~6.47s).
DEFAULT_TARGET_SECONDS_PER_IMAGE = 6.5
MIN_BLOCK_SECONDS = 4.0
MAX_BLOCK_SECONDS = 9.0
TARGET_WORDS_PER_IMAGE = 20
MAX_WORDS_PER_IMAGE = 30

# --- ДОКУМЕНТАЛЬНЫЕ ПАУЗЫ МЕЖДУ ФРАЗАМИ ---
# После каждой N-й фразы диктора вставляется пауза-перебивка (b-roll кадр с животными,
# диктор молчит, звучит только природа). Длительность паузы слегка "дышит" вокруг
# среднего значения (детерминированно, без random — чтобы повторные прогоны совпадали).
DEFAULT_PAUSES_ENABLED = os.getenv("DOC_PAUSES", "1").strip().lower() not in ("0", "false", "no", "нет")
DEFAULT_PAUSE_SECONDS = float(os.getenv("DOC_PAUSE_SECONDS", "2.2"))
DEFAULT_PAUSE_EVERY = int(os.getenv("DOC_PAUSE_EVERY", "1"))
PAUSE_MIN_SECONDS = 1.2
PAUSE_MAX_SECONDS = 4.0

SUPPORTED_INPUTS = {
    ".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".mp4", ".mov", ".mkv", ".webm"
}

LANGUAGE_ORDER = ["ru", "de", "es", "pl"]
LANGUAGES: dict[str, dict[str, Any]] = {
    "ru": {
        "name": "Russian",
        "folder": "RU_русский",
        "whisper": "ru",
        "cues": ["_ru", "-ru", " ru", "рус", "русский", "russian", "rus"],
    },
    "de": {
        "name": "German",
        "folder": "DE_немецкий",
        "whisper": "de",
        # Добавлены частые варианты названий: Германия / германский / DEU / Deutschland.
        "cues": [
            "_de", "-de", " de", "deu",
            "_ge", "-ge", " ge",  # поддержка файла GE.mp3 как немецкого
            "нем", "немец", "немецкий",
            "герм", "герман", "германия", "german", "deutsch", "deutschland"
        ],
    },
    "es": {
        "name": "Spanish",
        "folder": "ES_испанский",
        "whisper": "es",
        "cues": ["_es", "-es", " es", "исп", "испанский", "spanish", "espanol", "español"],
    },
    "pl": {
        "name": "Polish",
        "folder": "PL_польский",
        "whisper": "pl",
        "cues": ["_pl", "-pl", " pl", "пол", "польский", "polish", "polski"],
    },
}

# This style block is automatically prepended to EVERY final prompt.
# НЕЙТРАЛЬНЫЙ фотореалистичный кинодокументальный стиль: подходит под ЛЮБУЮ тему.
# Никакой привязки к птицам/океану — конкретный сюжет задаёт SCENE-часть промпта.
STYLE_PREFIX = (
    "photorealistic cinematic documentary frame, ultra-detailed realistic photography, "
    "shot on a professional camera with a high-quality lens, shallow depth of field with natural bokeh, "
    "razor-sharp focus on the main subject, lifelike textures and materials, "
    "true-to-life anatomy, proportions and fine detail, "
    "BBC Earth and National Geographic documentary style, "
    "dramatic natural lighting, warm golden-hour sun or soft moody overcast light, "
    "real atmosphere and real weather, rich cinematic color grading, high dynamic range, "
    "subtle film-like grain, crisp 4K UHD clarity, believable real-world environment, "
    "clean cinematic composition with a clear focal point, emotionally engaging real-world scene, "
    "photoreal only, not an illustration, not a painting, not watercolor, not a drawing, not a cartoon, "
    "not 3D, not CGI, not a render, not vector art, not flat design, no text, no subtitles, no labels, no watermark, no logo"
)

# Аудио-требование, добавляемое в КАЖДЫЙ финальный промпт (важно для генерации видео со звуком):
# только оригинальные звуки природы, как они звучат в реальности; никакой человеческой речи и музыки.
AUDIO_SUFFIX = (
    "Audio: only authentic natural ambient sound captured on location, exactly as it sounds in the wild — "
    "wind, air, water, rustling vegetation, distant weather, and the real calls, cries and movement sounds "
    "of the animals visible on screen. Strictly NO human voice, NO speech, NO narration, NO voice-over, "
    "NO singing, NO music, NO soundtrack, NO score, NO added artificial sound effects"
)


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
    scene_prompt: str
    image_prompt: str


@dataclass
class TimelineEntry:
    """Одна позиция ФИНАЛЬНОЙ временной шкалы ролика (с уже вставленными паузами).

    kind == "speech": фраза диктора; src_start/src_end — положение фразы в исходной озвучке.
    kind == "broll":  пауза-перебивка (кадры животных, диктор молчит); src_* = None.
    """
    index: int
    kind: str            # "speech" | "broll"
    start: float         # финальная шкала (с паузами)
    end: float
    src_start: float | None
    src_end: float | None
    text: str
    scene_prompt: str
    image_prompt: str

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class ProcessingJob:
    language_code: str
    input_path: Path


def run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    print("$", " ".join(cmd))
    return subprocess.run(cmd, check=check, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise RuntimeError("ffmpeg/ffprobe not found. Install with: brew install ffmpeg")


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def word_count(text: str) -> int:
    return len(re.findall(r"[\w']+", text, flags=re.UNICODE))


def safe_stem(path: Path) -> str:
    # macOS supports Cyrillic names, but this prevents slashes/odd separators in output names.
    stem = path.stem.strip()
    stem = re.sub(r"[\\/:*?\"<>|]+", "_", stem)
    stem = re.sub(r"\s+", "_", stem)
    return stem or "audio"


def find_input_files(folder: Path) -> list[Path]:
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_INPUTS]
    files = [p for p in files if "_transcribe_tmp" not in p.name]
    return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)


def normalize_for_language_detection(path: Path) -> str:
    text = f" {path.stem.lower()} "
    text = text.replace(".", " ").replace("_", "_").replace("-", "-")
    return text


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


def write_srt(segments: list[Segment], path: Path) -> None:
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


def get_media_duration(path: Path) -> float:
    result = run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path)
    ])
    return float(result.stdout.strip())


def make_transcription_audio(input_path: Path, work_dir: Path) -> Path:
    require_ffmpeg()
    out = work_dir / f"{safe_stem(input_path)}_transcribe_tmp.mp3"
    cmd = [
        "ffmpeg", "-y", "-i", str(input_path),
        "-vn", "-ac", "1", "-ar", "16000", "-b:a", "64k", str(out)
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out


def split_audio_if_needed(audio_path: Path, work_dir: Path, chunk_seconds: int = 600, max_mb: int = 24) -> list[Path]:
    size_mb = audio_path.stat().st_size / (1024 * 1024)
    if size_mb <= max_mb:
        return [audio_path]

    out_dir = work_dir / f"{audio_path.stem}_chunks"
    out_dir.mkdir(exist_ok=True)
    pattern = out_dir / "chunk_%03d.mp3"
    cmd = [
        "ffmpeg", "-y", "-i", str(audio_path),
        "-f", "segment", "-segment_time", str(chunk_seconds),
        "-c", "copy", str(pattern)
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
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
        request: dict[str, Any] = {
            "model": TRANSCRIBE_MODEL,
            "file": None,
            "response_format": "verbose_json",
            "temperature": 0,
        }
        if whisper_language:
            request["language"] = whisper_language

        with chunk.open("rb") as f:
            request["file"] = f
            transcription = client.audio.transcriptions.create(**request)

        data = response_to_dict(transcription)
        segs = data.get("segments") or []

        if not segs and data.get("text"):
            duration = get_media_duration(chunk)
            segs = [{"start": 0.0, "end": duration, "text": data["text"]}]

        for seg in segs:
            start = float(seg.get("start", 0)) + time_offset
            end = float(seg.get("end", start + 1)) + time_offset
            text = clean_text(seg.get("text", ""))
            if text:
                all_segments.append(Segment(start=start, end=end, text=text))

        time_offset += get_media_duration(chunk)

    try:
        tmp_audio.unlink(missing_ok=True)
    except Exception:
        pass

    return all_segments


def split_long_segment(seg: Segment, max_words: int = 22) -> list[Segment]:
    text = clean_text(seg.text)
    words = text.split()
    if len(words) <= max_words:
        return [seg]

    pieces: list[str] = []
    current: list[str] = []
    for w in words:
        current.append(w)
        if len(current) >= max_words and re.search(r"[.!?,;:…]$", w):
            pieces.append(" ".join(current))
            current = []
    if current:
        pieces.append(" ".join(current))

    if len(pieces) == 1:
        pieces = [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]

    total_words = sum(len(p.split()) for p in pieces) or 1
    duration = max(0.1, seg.end - seg.start)
    out: list[Segment] = []
    cursor = seg.start
    for p in pieces:
        frac = len(p.split()) / total_words
        end = cursor + duration * frac
        out.append(Segment(start=cursor, end=end, text=p))
        cursor = end
    out[-1].end = seg.end
    return out


def normalize_segments(segments: list[Segment]) -> list[Segment]:
    out: list[Segment] = []
    for seg in segments:
        out.extend(split_long_segment(seg))
    return out


def build_visual_blocks(
    segments: list[Segment],
    target_seconds: float = DEFAULT_TARGET_SECONDS_PER_IMAGE,
) -> list[VisualBlock]:
    segments = normalize_segments(segments)
    blocks: list[VisualBlock] = []
    cur: list[Segment] = []

    def flush() -> None:
        nonlocal cur
        if not cur:
            return
        text = clean_text(" ".join(s.text for s in cur))
        blocks.append(VisualBlock(index=len(blocks) + 1, start=cur[0].start, end=cur[-1].end, text=text))
        cur = []

    for seg in segments:
        seg_text = clean_text(seg.text)
        if not seg_text:
            continue

        if not cur:
            cur.append(seg)
            continue

        candidate_text = clean_text(" ".join(s.text for s in cur + [seg]))
        candidate_seconds = seg.end - cur[0].start
        candidate_words = word_count(candidate_text)
        previous_text = clean_text(" ".join(s.text for s in cur))

        topic_break_markers = [
            # English
            "No.1", "No. 1", "number one", "first", "second", "third", "the answer", "welcome back",
            "subscribe", "stay until the end",
            # Russian
            "номер один", "во-первых", "первое", "первый", "второе", "второй", "третье", "третий",
            "ответ", "добро пожаловать", "подпишись", "подписывайтесь", "досмотрите до конца",
            # German
            "nummer eins", "erstens", "erste", "zweite", "zweitens", "dritte", "drittens", "die antwort",
            "willkommen zurück", "abonnieren", "bleib bis zum ende",
            # Spanish
            "número uno", "numero uno", "primero", "primer", "segundo", "tercero", "la respuesta",
            "bienvenido", "suscríbete", "suscribete", "quédate hasta el final", "quedate hasta el final",
            # Polish
            "numer jeden", "po pierwsze", "pierwszy", "pierwsza", "drugi", "po drugie", "trzeci", "po trzecie",
            "odpowiedź", "odpowiedz", "witaj ponownie", "zasubskrybuj", "zostań do końca", "zostan do konca",
        ]
        starts_new_topic = any(seg_text.lower().startswith(m.lower()) for m in topic_break_markers)
        previous_ends_sentence = bool(re.search(r"[.!?…]$", previous_text))

        should_flush = False
        if candidate_seconds >= MAX_BLOCK_SECONDS:
            should_flush = True
        elif candidate_words >= MAX_WORDS_PER_IMAGE:
            should_flush = True
        elif candidate_seconds >= target_seconds and candidate_words >= TARGET_WORDS_PER_IMAGE:
            should_flush = True
        elif starts_new_topic and (cur[-1].end - cur[0].start) >= MIN_BLOCK_SECONDS:
            should_flush = True
        elif previous_ends_sentence and (cur[-1].end - cur[0].start) >= target_seconds:
            should_flush = True

        if should_flush:
            flush()
        cur.append(seg)

    flush()

    merged: list[VisualBlock] = []
    for block in blocks:
        duration = block.end - block.start
        if merged and duration < 2.8 and word_count(block.text) < 10:
            prev = merged[-1]
            merged[-1] = VisualBlock(
                index=prev.index,
                start=prev.start,
                end=block.end,
                text=clean_text(prev.text + " " + block.text),
            )
        else:
            merged.append(block)

    for i, block in enumerate(merged, 1):
        block.index = i
    return merged


# ---------------------------------------------------------------------------
# THEME BRIEF: автоматически определяем, О ЧЁМ документалка, по тексту озвучки.
# Это заменяет старую жёсткую привязку к птицам/океану: теперь скрипт универсален
# и держит весь ролик в реальной теме сценария (насекомое, история, техника и т.д.).
# ---------------------------------------------------------------------------

THEME_SYSTEM_PROMPT = """
You are the art director of a photorealistic cinematic documentary.
You will read the FULL narration (voiceover) of a short documentary video.
The narration can be in Russian, German, Spanish, or Polish.

Your job: figure out what the documentary is REALLY about and produce a compact visual brief in English
that will guide image generation for the WHOLE video, so every shot feels like it belongs to ONE
coherent real documentary about this exact topic (like BBC Earth or National Geographic).

Return valid JSON only, with this exact shape:
{
  "subject": "the main recurring real-world subject of the documentary (a specific animal, plant, insect, object, technology, place, or type of person)",
  "setting": "the real places, habitats, historical era(s), countries and environments where the story happens",
  "look": "short guidance on the realistic visual style and mood that fits THIS topic (typical locations, lighting, kinds of shots)",
  "keywords": ["8 to 15 concrete visual nouns specific to THIS topic that should recur across shots"]
}

Rules:
- Everything must be REAL and photographable. No fantasy, no surreal elements.
- Be specific to the ACTUAL narration. If it is about an insect, name the insect and its real habitat and life stages.
  If it is about history, name the era, the real places, the clothing, the objects, the technology.
- Do NOT default to birds, the ocean, or any topic that is not actually in the narration.
- Write everything in English.
""".strip()


THEME_USER_PROMPT_TEMPLATE = """
Read this full documentary narration and produce the visual brief as described.

Language: __LANGUAGE_NAME__ (__LANGUAGE_CODE__)

Full narration:
__FULL_TEXT__
""".strip()


def format_theme_brief(brief: dict[str, Any]) -> str:
    """Human-readable brief that is injected into every scene-generation batch."""
    if not brief:
        return (
            "No explicit brief available. Infer the real subject and setting directly from each block's meaning, "
            "and keep all shots consistent with one real documentary topic."
        )
    subject = clean_text(str(brief.get("subject", "")))
    setting = clean_text(str(brief.get("setting", "")))
    look = clean_text(str(brief.get("look", "")))
    keywords = brief.get("keywords", [])
    if isinstance(keywords, list):
        keywords_text = ", ".join(clean_text(str(k)) for k in keywords if str(k).strip())
    else:
        keywords_text = clean_text(str(keywords))

    lines: list[str] = []
    if subject:
        lines.append(f"Main recurring subject: {subject}")
    if setting:
        lines.append(f"Real setting / habitat / era: {setting}")
    if look:
        lines.append(f"Realistic look and mood: {look}")
    if keywords_text:
        lines.append(f"Recurring concrete visual elements: {keywords_text}")
    return "\n".join(lines) if lines else "No explicit brief available."


def generate_theme_brief(client: OpenAI, blocks: list[VisualBlock], language_code: str) -> dict[str, Any]:
    full_text = clean_text(" ".join(b.text for b in blocks))
    if not full_text:
        return {}

    # Ограничиваем длину, чтобы не раздувать запрос на очень длинных сценариях.
    max_chars = 12000
    if len(full_text) > max_chars:
        full_text = full_text[:max_chars]

    language_name = LANGUAGES.get(language_code, {}).get("name", language_code.upper())
    user_prompt = THEME_USER_PROMPT_TEMPLATE
    user_prompt = user_prompt.replace("__LANGUAGE_CODE__", language_code)
    user_prompt = user_prompt.replace("__LANGUAGE_NAME__", language_name)
    user_prompt = user_prompt.replace("__FULL_TEXT__", full_text)

    try:
        response = client.chat.completions.create(
            model=PROMPT_MODEL,
            temperature=0.2,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": THEME_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        data = json.loads(response.choices[0].message.content or "{}")
        if isinstance(data, dict):
            return data
    except Exception as exc:  # noqa: BLE001 - брифа может не быть, генерация всё равно продолжится
        print(f"WARNING: theme brief generation failed ({exc}). Falling back to per-block inference.")
    return {}


SCENE_SYSTEM_PROMPT = """
You create SCENE DESCRIPTIONS for photorealistic image-generation prompts of a cinematic documentary.
The documentary can be about ANY topic (an animal, an insect, a plant, an object, a technology, a place, a historical event).
The voiceover blocks can be in Russian, German, Spanish, or Polish.
Understand the meaning of each block and write the scene descriptions in English.
The final photoreal cinematic style will be added automatically by the script, so you must focus on content, action, composition, and mood.

You are given a VISUAL BRIEF that describes the documentary's real subject, setting and recurring elements.
Use it to keep the WHOLE video coherent: the same real subject and world should recur across shots,
exactly like a real BBC Earth / National Geographic / historical documentary about this exact topic.

Your scene descriptions must follow these rules:
- They are for STATIC IMAGES only.
- Each scene must directly and literally visualize the exact meaning of the spoken block.
- Convert abstract narration into concrete, real-world visual imagery that is consistent with the brief.
- Prefer visible actions, real places, real objects, and believable real-world behavior.
- Follow the real habitat, setting, country or historical era implied by the narration and the brief.
  Do NOT invent a different topic and do NOT default to birds or the ocean unless the narration is actually about them.
- Vary framing strongly across blocks, like a real documentary edit: some extreme close-up / macro shots of the subject and its fine detail,
  some tight detail shots, some medium shots, some wide establishing shots of the environment, some super-wide landscapes with a tiny distant subject,
  some low-angle shots, some dramatic silhouettes against a glowing sky, some over-the-shoulder or hands-in-frame shots when people or crafts are involved.
- Name the specific subject (species, object, place, or type of person) when it is known from the text or the brief; otherwise describe a plausible real one that fits the topic.
- Keep it realistic and physically plausible — no fantasy, no surreal elements, no anthropomorphism.
- Do not add style words like photorealistic, cinematic, telephoto, 4K, etc. The script adds the style automatically.
- Do not include negative prompts.
- Write in English.
- Each scene description should be 30-65 words.
""".strip()


SCENE_USER_PROMPT_TEMPLATE = """
Create scene descriptions for these timed voiceover blocks.
Return valid JSON only, with this exact shape:
{
  "items": [
    {
      "index": 1,
      "scene_prompt": "..."
    }
  ]
}

Important:
- One scene description per block.
- The scene description should describe what the photo-real picture must show.
- Keep every shot consistent with the VISUAL BRIEF below (same real subject and world).
- No text inside image.
- No negative prompt.
- The script will automatically prepend a fixed photorealistic cinematic style to every final prompt.
- Use the timing only for context; do not mention the timecode inside the scene prompt.

Language: __LANGUAGE_NAME__ (__LANGUAGE_CODE__)

VISUAL BRIEF (applies to the whole video):
__THEME_BRIEF__

Timed blocks:
__BLOCKS_JSON__
""".strip()


def compose_final_prompt(scene_prompt: str) -> str:
    scene_prompt = clean_text(scene_prompt).rstrip(". ")
    return f"{STYLE_PREFIX}. Scene: {scene_prompt}. {AUDIO_SUFFIX}."


def fallback_scene_prompt(text: str) -> str:
    return (
        "A clear, specific, real documentary scene that directly and literally shows what this narration describes: "
        f"{clean_text(text)}, with a readable main subject, visible action, a real environment, real lighting, "
        "and a clean cinematic composition"
    )


def generate_prompts_for_batch(
    client: OpenAI,
    blocks: list[VisualBlock],
    language_code: str,
    theme_brief_text: str,
) -> list[PromptRow]:
    language_name = LANGUAGES.get(language_code, {}).get("name", language_code.upper())
    blocks_payload = [
        {
            "index": b.index,
            "start": seconds_to_label(b.start),
            "end": seconds_to_label(b.end),
            "text": b.text,
        }
        for b in blocks
    ]

    user_prompt = SCENE_USER_PROMPT_TEMPLATE
    user_prompt = user_prompt.replace("__LANGUAGE_CODE__", language_code)
    user_prompt = user_prompt.replace("__LANGUAGE_NAME__", language_name)
    user_prompt = user_prompt.replace("__THEME_BRIEF__", theme_brief_text)
    user_prompt = user_prompt.replace("__BLOCKS_JSON__", json.dumps(blocks_payload, ensure_ascii=False, indent=2))

    response = client.chat.completions.create(
        model=PROMPT_MODEL,
        temperature=0.5,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SCENE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    content = response.choices[0].message.content or "{}"
    data = json.loads(content)
    items = data.get("items", [])
    by_index = {int(item["index"]): item for item in items if "index" in item}

    rows: list[PromptRow] = []
    for b in blocks:
        item = by_index.get(b.index, {})
        scene_prompt = clean_text(item.get("scene_prompt", ""))
        if not scene_prompt:
            scene_prompt = fallback_scene_prompt(b.text)
        image_prompt = compose_final_prompt(scene_prompt)
        rows.append(PromptRow(
            index=b.index,
            language_code=language_code,
            language_name=language_name,
            start=b.start,
            end=b.end,
            duration=b.end - b.start,
            text=b.text,
            scene_prompt=scene_prompt,
            image_prompt=image_prompt,
        ))
    return rows


# ---------------------------------------------------------------------------
# B-ROLL ПЕРЕБИВКИ ДЛЯ ПАУЗ: отдельные кадры с животными/природой без слов.
# ---------------------------------------------------------------------------

BROLL_SYSTEM_PROMPT = """
You create SILENT CUTAWAY (b-roll) scene descriptions for a photorealistic wildlife/nature documentary.
These shots are shown during PAUSES between narration phrases, exactly like in real BBC Earth /
National Geographic documentaries: no words are spoken, the viewer simply watches the animals and
listens to the natural ambient sound of the location.

You are given a VISUAL BRIEF of the documentary. Produce the requested number of varied cutaway shots
that all belong to the SAME real subject and world as the brief.

Rules:
- Every shot must feature the documentary's real animals/subject or its immediate natural habitat.
- Prefer intimate observational moments: an animal breathing, blinking, grooming, feeding calmly,
  turning its head, a slow detail of fur/feathers/skin, wind moving the habitat, water, light.
- Vary framing strongly: extreme close-up / macro, tight detail, medium, wide establishing, silhouette.
- Realistic and physically plausible only. No fantasy, no anthropomorphism, no people, no text.
- Do not add style words (photorealistic, cinematic, 4K...) — they are added automatically.
- Write in English. Each description 25-55 words.
Return valid JSON only: {"items": [{"index": 1, "scene_prompt": "..."}]}
""".strip()


def fallback_broll_scene(brief: dict[str, Any], i: int) -> str:
    subject = clean_text(str(brief.get("subject", ""))) if brief else ""
    setting = clean_text(str(brief.get("setting", ""))) if brief else ""
    subject = subject or "the documentary's main animal"
    setting = setting or "its real natural habitat"
    variants = [
        f"Calm observational close-up of {subject} at rest in {setting}, slow natural breathing, eyes and fine surface detail clearly visible, soft natural light",
        f"Wide establishing shot of {setting}, {subject} small in the frame, wind gently moving the environment, natural daylight",
        f"Extreme macro detail of {subject} — texture of its body surface, tiny natural movements, shallow depth of field in {setting}",
        f"Medium shot of {subject} feeding or grooming calmly in {setting}, unhurried natural behavior, believable real-world moment",
        f"Low-angle shot of {subject} against the sky or landscape of {setting}, quiet dramatic silhouette, natural atmosphere",
    ]
    return variants[i % len(variants)]


def generate_broll_prompts(
    client: OpenAI,
    brief: dict[str, Any],
    theme_brief_text: str,
    count: int,
    language_code: str,
) -> list[str]:
    """Возвращает count сцен-перебивок (scene prompts) для пауз между фразами."""
    if count <= 0:
        return []

    scenes: list[str] = []
    try:
        user_prompt = (
            f"Create exactly {count} cutaway shot descriptions.\n\n"
            f"VISUAL BRIEF (applies to the whole video):\n{theme_brief_text}\n"
        )
        response = client.chat.completions.create(
            model=PROMPT_MODEL,
            temperature=0.6,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": BROLL_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        data = json.loads(response.choices[0].message.content or "{}")
        items = data.get("items", [])
        for item in items:
            text = clean_text(str(item.get("scene_prompt", "")))
            if text:
                scenes.append(text)
    except Exception as exc:  # noqa: BLE001 - при сбое LLM работаем на фолбэках
        print(f"WARNING: b-roll prompt generation failed ({exc}). Using fallback cutaway scenes.")

    while len(scenes) < count:
        scenes.append(fallback_broll_scene(brief, len(scenes)))
    return scenes[:count]


def pause_duration_for(i: int, base_seconds: float) -> float:
    """Детерминированная 'дышащая' длительность паузы: 0.75x..1.25x от базовой."""
    factor = 0.75 + 0.5 * ((i * 3) % 5) / 4.0
    return max(PAUSE_MIN_SECONDS, min(PAUSE_MAX_SECONDS, base_seconds * factor))


# Фраза считается ЗАВЕРШЁННОЙ, только если заканчивается точкой/!/?/…
# (возможно с закрывающей кавычкой/скобкой после знака).
SENTENCE_END_RE = re.compile(r"[.!?…]+[»\"'\)\]]*\s*$")


def phrase_is_complete(text: str) -> bool:
    return bool(SENTENCE_END_RE.search(clean_text(text)))


def choose_pause_positions(rows: list[PromptRow], pause_every: int) -> list[int]:
    """Выбирает, ПОСЛЕ каких фраз можно ставить паузу-перебивку.

    Ключевое правило (исправление «обрывистых» пауз): пауза допустима ТОЛЬКО после
    фразы, которая заканчивается завершённым предложением (. ! ? …). Если блок
    оборвался посреди мысли («...об одном из самых редких» -> «союзов»), пауза после
    него НЕ ставится — речь продолжается без разрыва, а пауза сдвигается к концу
    ближайшего завершённого предложения.

    pause_every применяется к ЗАВЕРШЁННЫМ фразам: пауза после каждой N-й из них.
    После самой последней фразы пауза тоже допускается (финальный кадр природы).
    """
    positions: list[int] = []
    eligible_seen = 0
    for i, r in enumerate(rows):
        if not phrase_is_complete(r.text):
            continue
        eligible_seen += 1
        if pause_every > 0 and eligible_seen % pause_every == 0:
            positions.append(i)
    return positions


def build_timeline_with_pauses(
    rows: list[PromptRow],
    broll_scenes: list[str],
    pause_after: list[int],
    base_pause_seconds: float,
) -> list[TimelineEntry]:
    """Строит финальную шкалу: фразы диктора + паузы-перебивки между ними.

    Пауза вставляется только после фраз из pause_after (индексы rows) — то есть
    только на границах завершённых предложений. Времена фраз сдвигаются на
    суммарную длительность уже вставленных пауз, а исходное положение фразы
    в озвучке сохраняется в src_start/src_end.
    """
    entries: list[TimelineEntry] = []
    shift = 0.0
    broll_used = 0
    pause_set = set(pause_after)

    for i, r in enumerate(rows):
        entries.append(TimelineEntry(
            index=0,
            kind="speech",
            start=r.start + shift,
            end=r.end + shift,
            src_start=r.start,
            src_end=r.end,
            text=r.text,
            scene_prompt=r.scene_prompt,
            image_prompt=r.image_prompt,
        ))

        insert_pause = i in pause_set
        if insert_pause and broll_used < len(broll_scenes):
            dur = pause_duration_for(i, base_pause_seconds)
            scene = broll_scenes[broll_used]
            broll_used += 1
            pause_start = r.end + shift
            entries.append(TimelineEntry(
                index=0,
                kind="broll",
                start=pause_start,
                end=pause_start + dur,
                src_start=None,
                src_end=None,
                text="",
                scene_prompt=scene,
                image_prompt=compose_final_prompt(scene),
            ))
            shift += dur

    for n, e in enumerate(entries, 1):
        e.index = n
    return entries




def generate_prompts(
    client: OpenAI,
    blocks: list[VisualBlock],
    language_code: str,
    batch_size: int = 18,
) -> tuple[list[PromptRow], dict[str, Any], str]:
    # 1) Один раз определяем тему всего ролика по полному тексту озвучки.
    brief = generate_theme_brief(client, blocks, language_code)
    theme_brief_text = format_theme_brief(brief)
    detected_subject = clean_text(str(brief.get("subject", ""))) if brief else ""
    print(f"Theme for {language_code.upper()}: {detected_subject or 'auto-inferred per block'}")

    # 2) Генерируем сцены пачками, каждая пачка знает общий бриф темы.
    rows: list[PromptRow] = []
    for i in range(0, len(blocks), batch_size):
        batch = blocks[i:i + batch_size]
        print(f"Generating {language_code.upper()} prompts {batch[0].index}-{batch[-1].index}...")
        rows.extend(generate_prompts_for_batch(client, batch, language_code, theme_brief_text))
    return rows, brief, theme_brief_text


def timing_header_line(e: TimelineEntry) -> str:
    """Строка тайминга финальной шкалы.

    speech: 001 | 00:00:00,000 --> 00:00:06,640 | duration: 6.64s | type: speech | src: 00:00:00,000 --> 00:00:06,640
    broll:  002 | 00:00:06,640 --> 00:00:08,840 | duration: 2.20s | type: broll
    Поле src говорит сборщику видео, откуда вырезать эту фразу из исходной озвучки.
    """
    base = (
        f"{e.index:03d} | {seconds_to_srt_time(e.start)} --> {seconds_to_srt_time(e.end)} "
        f"| duration: {e.duration:.2f}s | type: {e.kind}"
    )
    if e.kind == "speech" and e.src_start is not None and e.src_end is not None:
        base += f" | src: {seconds_to_srt_time(e.src_start)} --> {seconds_to_srt_time(e.src_end)}"
    return base


def write_image_times(entries: list[TimelineEntry], txt_path: Path, csv_path: Path) -> None:
    lines: list[str] = [timing_header_line(e) for e in entries]
    txt_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["index", "start", "end", "duration", "type", "src_start", "src_end"])
        writer.writeheader()
        for e in entries:
            writer.writerow({
                "index": e.index,
                "start": seconds_to_srt_time(e.start),
                "end": seconds_to_srt_time(e.end),
                "duration": round(e.duration, 2),
                "type": e.kind,
                "src_start": seconds_to_srt_time(e.src_start) if e.src_start is not None else "",
                "src_end": seconds_to_srt_time(e.src_end) if e.src_end is not None else "",
            })


def write_outputs(
    input_path: Path,
    language_code: str,
    segments: list[Segment],
    blocks: list[VisualBlock],
    rows: list[PromptRow],
    entries: list[TimelineEntry],
    language_dir: Path,
    root_output_dir: Path,
    target_seconds: float,
    write_root_copies: bool = True,
) -> None:
    language_name = LANGUAGES.get(language_code, {}).get("name", language_code.upper())
    stem = f"{language_code}_{safe_stem(input_path)}"

    srt_path = language_dir / f"{stem}.srt"
    blocks_path = language_dir / f"{stem}_visual_blocks.json"
    prompts_json_path = language_dir / f"{stem}_prompts.json"
    prompts_csv_path = language_dir / f"{stem}_prompts.csv"
    prompts_txt_path = language_dir / f"{stem}_prompts.txt"
    image_times_txt_path = language_dir / f"{stem}_image_times.txt"
    image_times_csv_path = language_dir / f"{stem}_image_times.csv"
    summary_path = language_dir / f"{stem}_summary.txt"

    write_srt(segments, srt_path)
    blocks_path.write_text(json.dumps([asdict(b) for b in blocks], ensure_ascii=False, indent=2), encoding="utf-8")
    prompts_json_path.write_text(json.dumps([asdict(e) for e in entries], ensure_ascii=False, indent=2), encoding="utf-8")

    with prompts_csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["index", "type", "language_code", "language_name", "start", "end", "duration", "text", "scene_prompt", "image_prompt"],
        )
        writer.writeheader()
        for e in entries:
            writer.writerow({
                "index": e.index,
                "type": e.kind,
                "language_code": language_code,
                "language_name": language_name,
                "start": seconds_to_srt_time(e.start),
                "end": seconds_to_srt_time(e.end),
                "duration": round(e.duration, 2),
                "text": e.text,
                "scene_prompt": e.scene_prompt,
                "image_prompt": e.image_prompt,
            })

    # TXT format:
    # line 1 = index + timecode + duration + type (+ src for speech)
    # line 2 = final prompt (у broll-перебивок — свой отдельный промпт с кадрами животных)
    # blank line between entries
    # no negative prompt beyond the fixed style restrictions already included in STYLE_PREFIX
    txt_content_lines: list[str] = []
    for e in entries:
        txt_content_lines.append(timing_header_line(e))
        txt_content_lines.append(e.image_prompt)
        txt_content_lines.append("")

    txt_content = "\n".join(txt_content_lines)
    prompts_txt_path.write_text(txt_content, encoding="utf-8")
    write_image_times(entries, image_times_txt_path, image_times_csv_path)

    if write_root_copies:
        # Copies directly in the ПРОМПТЫ folder, so they are easy to grab without opening language subfolders.
        shutil.copyfile(prompts_txt_path, root_output_dir / f"{stem}_prompts.txt")
        shutil.copyfile(image_times_txt_path, root_output_dir / f"{stem}_image_times.txt")

        # Convenience fixed names for the latest processed file of each language.
        shutil.copyfile(prompts_txt_path, root_output_dir / f"prompts_{language_code}.txt")
        shutil.copyfile(image_times_txt_path, root_output_dir / f"image_times_{language_code}.txt")

    duration = max((s.end for s in segments), default=0.0)
    theoretical = math.ceil(duration / target_seconds) if duration else len(rows)
    broll_count = sum(1 for e in entries if e.kind == "broll")
    pause_total = sum(e.duration for e in entries if e.kind == "broll")
    final_duration = entries[-1].end if entries else duration
    summary = f"""Input: {input_path.name}
Language: {language_name} ({language_code})
Voiceover duration: {duration:.2f} seconds ({duration / 60:.2f} minutes)
Final timeline duration (with documentary pauses): {final_duration:.2f} seconds ({final_duration / 60:.2f} minutes)
SRT segments: {len(segments)}
Speech blocks / narration prompts: {len(rows)}
B-roll pause cutaways inserted: {broll_count} (total pause time: {pause_total:.2f}s)
Total timeline entries (speech + broll): {len(entries)}
Theoretical count by target {target_seconds:.2f}s/image: {theoretical}
Average seconds per speech block: {(duration / len(rows)) if rows else 0:.2f}
Average words per speech block: {(sum(word_count(r.text) for r in rows) / len(rows)) if rows else 0:.2f}

Files in language folder:
- {srt_path.name}
- {blocks_path.name}
- {prompts_json_path.name}
- {prompts_csv_path.name}
- {prompts_txt_path.name}
- {image_times_txt_path.name}
- {image_times_csv_path.name}

Root copies in ПРОМПТЫ:
- {stem}_prompts.txt
- {stem}_image_times.txt
- prompts_{language_code}.txt
- image_times_{language_code}.txt
"""
    summary_path.write_text(summary, encoding="utf-8")
    print(summary)


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
        jobs: list[ProcessingJob] = []
        for file_path in files:
            code = detect_language_from_filename(file_path) or "unknown"
            jobs.append(ProcessingJob(language_code=code, input_path=file_path))
        return jobs

    # Default mode: choose one newest detected file for each of the 4 target languages.
    selected: dict[str, Path] = {}
    unknown_files: list[Path] = []

    for file_path in files:
        code = detect_language_from_filename(file_path)
        if code in LANGUAGE_ORDER and code not in selected:
            selected[code] = file_path
        elif code is None:
            unknown_files.append(file_path)

    # If filenames have no language markers but there are exactly 4 files, assign them by name order.
    # This is a fallback only. Recommended: use _ru, _de, _es, _pl in filenames or pass --file-ru etc.
    if not selected and len(files) == 4:
        for code, file_path in zip(LANGUAGE_ORDER, sorted(files, key=lambda p: p.name.lower())):
            selected[code] = file_path

    missing = [code for code in LANGUAGE_ORDER if code not in selected]

    # Если 3 языка определились, а остался ровно 1 неопознанный файл,
    # автоматически считаем его недостающим языком.
    # Это спасает ситуацию, когда немецкая озвучка названа, например, "Германия.mp3"
    # или любым другим нестандартным именем.
    if len(missing) == 1 and len(unknown_files) == 1:
        selected[missing[0]] = unknown_files[0]
        print(f"INFO: Auto-assigned unknown file '{unknown_files[0].name}' to missing language: {missing[0].upper()}")
        missing = []

    if missing:
        print("WARNING: Could not auto-detect files for languages:", ", ".join(missing))
        print("Files found in voiceover folder:")
        for file_path in files:
            detected = detect_language_from_filename(file_path) or "unknown"
            print(f"- {file_path.name} -> {detected}")
        print("Recommended filename markers: _ru, _de, _es, _pl")
        print("German markers also supported: немецкий, Германия, German, Deutsch, Deutschland, DEU")
        print("Or pass files manually: --file-ru ... --file-de ... --file-es ... --file-pl ...")

    jobs = [ProcessingJob(language_code=code, input_path=selected[code]) for code in LANGUAGE_ORDER if code in selected]
    if not jobs:
        raise RuntimeError("No language jobs were selected. Rename files with _ru/_de/_es/_pl or pass --file-ru/--file-de/--file-es/--file-pl.")
    return jobs


def process_file(
    client: OpenAI,
    job: ProcessingJob,
    root_output_dir: Path,
    target_seconds: float,
    write_root_copies: bool = True,
    pauses_enabled: bool = DEFAULT_PAUSES_ENABLED,
    pause_seconds: float = DEFAULT_PAUSE_SECONDS,
    pause_every: int = DEFAULT_PAUSE_EVERY,
) -> None:
    input_path = job.input_path.expanduser()
    if not input_path.exists():
        raise RuntimeError(f"Input file not found: {input_path}")

    language_code = job.language_code
    language_folder_name = LANGUAGES.get(language_code, {}).get("folder", f"UNKNOWN_{language_code}")
    language_dir = root_output_dir / language_folder_name
    language_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Processing {input_path.name} | language={language_code.upper()} ===")
    segments = transcribe_audio(client, input_path, language_dir, language_code)
    if not segments:
        raise RuntimeError(f"No transcription segments returned for {input_path}")

    blocks = build_visual_blocks(segments, target_seconds=target_seconds)
    rows, brief, theme_brief_text = generate_prompts(client, blocks, language_code)

    # Документальные паузы: b-roll перебивки с животными — ТОЛЬКО на границах
    # завершённых предложений, чтобы речь не обрывалась посреди мысли.
    pause_positions: list[int] = []
    broll_scenes: list[str] = []
    if pauses_enabled and pause_every > 0:
        pause_positions = choose_pause_positions(rows, pause_every)
        incomplete = sum(1 for r in rows if not phrase_is_complete(r.text))
        if incomplete:
            print(f"INFO: {incomplete} block(s) end mid-sentence — no pause will be placed after them.")
        if not pause_positions:
            print("WARNING: no complete-sentence boundaries found (transcript without punctuation?). "
                  "No documentary pauses will be inserted.")
        else:
            print(f"Inserting documentary pauses for {language_code.upper()}: "
                  f"{len(pause_positions)} b-roll cutaways (~{pause_seconds:.1f}s each) "
                  f"at sentence boundaries only")
            broll_scenes = generate_broll_prompts(client, brief, theme_brief_text, len(pause_positions), language_code)
    else:
        print(f"Documentary pauses disabled for {language_code.upper()}")

    entries = build_timeline_with_pauses(
        rows=rows,
        broll_scenes=broll_scenes,
        pause_after=pause_positions if broll_scenes else [],
        base_pause_seconds=pause_seconds,
    )

    write_outputs(
        input_path=input_path,
        language_code=language_code,
        segments=segments,
        blocks=blocks,
        rows=rows,
        entries=entries,
        language_dir=language_dir,
        root_output_dir=root_output_dir,
        target_seconds=target_seconds,
        write_root_copies=write_root_copies,
    )


def write_master_summary(root_output_dir: Path, jobs: list[ProcessingJob]) -> None:
    lines = [
        "Multilanguage prompt generation summary",
        "",
        "Processed jobs:",
    ]
    for job in jobs:
        language_name = LANGUAGES.get(job.language_code, {}).get("name", job.language_code.upper())
        lines.append(f"- {language_name} ({job.language_code}): {job.input_path.name}")
    lines.extend([
        "",
        "Main root files:",
        "- prompts_ru.txt / image_times_ru.txt",
        "- prompts_de.txt / image_times_de.txt",
        "- prompts_es.txt / image_times_es.txt",
        "- prompts_pl.txt / image_times_pl.txt",
        "",
        "Each language also has its own subfolder with SRT, JSON, CSV, prompts, and image timing files.",
    ])
    (root_output_dir / "_multilang_summary.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transcribe RU/DE/ES/PL voiceovers and generate photorealistic documentary prompts (ANY topic) with separate image timings."
    )
    parser.add_argument(
        "--voiceover-folder",
        "--folder",
        default=DEFAULT_VOICEOVER_FOLDER,
        help="Folder containing voiceover audio/video files. Default: ОЗВУЧКА folder.",
    )
    parser.add_argument(
        "--output-folder",
        default=DEFAULT_PROMPTS_FOLDER,
        help="Folder where prompts and image timings are saved. Default: ПРОМПТЫ folder.",
    )
    parser.add_argument("--all", action="store_true", help="Process all supported files in the voiceover folder.")
    parser.add_argument("--file", default=None, help="Process one specific audio/video file.")
    parser.add_argument("--language", choices=LANGUAGE_ORDER + ["unknown"], default=None, help="Language for --file mode.")
    parser.add_argument("--file-ru", default=None, help="Specific Russian voiceover file.")
    parser.add_argument("--file-de", default=None, help="Specific German voiceover file.")
    parser.add_argument("--file-es", default=None, help="Specific Spanish voiceover file.")
    parser.add_argument("--file-pl", default=None, help="Specific Polish voiceover file.")
    parser.add_argument(
        "--target-seconds",
        type=float,
        default=DEFAULT_TARGET_SECONDS_PER_IMAGE,
        help="Target seconds per image prompt.",
    )
    parser.add_argument(
        "--no-root-copies",
        action="store_true",
        help="Do not create convenience prompt/timing copies directly in the ПРОМПТЫ root folder.",
    )
    parser.add_argument(
        "--no-pauses",
        action="store_true",
        help="Отключить документальные паузы (b-roll перебивки между фразами).",
    )
    parser.add_argument(
        "--pause-seconds",
        type=float,
        default=DEFAULT_PAUSE_SECONDS,
        help=f"Средняя длительность паузы-перебивки в секундах (default: {DEFAULT_PAUSE_SECONDS}).",
    )
    parser.add_argument(
        "--pause-every",
        type=int,
        default=DEFAULT_PAUSE_EVERY,
        help=f"Вставлять паузу после каждой N-й фразы (default: {DEFAULT_PAUSE_EVERY}).",
    )
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Set OPENAI_API_KEY first: export OPENAI_API_KEY='sk-...'")

    root_output_dir = Path(args.output_folder).expanduser()
    root_output_dir.mkdir(parents=True, exist_ok=True)

    jobs = build_jobs(args)
    client = OpenAI()

    for job in jobs:
        process_file(
            client=client,
            job=job,
            root_output_dir=root_output_dir,
            target_seconds=args.target_seconds,
            write_root_copies=not args.no_root_copies,
            pauses_enabled=DEFAULT_PAUSES_ENABLED and not args.no_pauses,
            pause_seconds=args.pause_seconds,
            pause_every=args.pause_every,
        )

    write_master_summary(root_output_dir, jobs)
    print(f"\nDone. Prompts and image timings saved to: {root_output_dir}")


if __name__ == "__main__":
    main()
