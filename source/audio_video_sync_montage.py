# -*- coding: utf-8 -*-
"""
Синхронизация аудио и 8-секундных роликов: тайм-коды по смыслу + автомонтаж.

ИДЕЯ
====
Это 4-й скрипт к уже существующему конвейеру:

    scenario.txt
        -> master_prompt_pipeline...py   -> generated_prompts.json (сцены + sentence_indexes + block_text)
        -> media_gen_..._flower.py        -> КАРТИНКИ/0001.png, 0002.png, ...
        -> video_from_local_images...py   -> ВИДЕО/0001.mp4, 0002.mp4, ...  (каждый ~8 сек)

Нумерация сквозная: global_scene_index=1 -> 0001.png -> 0001.mp4.

Проблема, которую решает этот скрипт: в master-конвейере длительности сцен
ОЦЕНОЧНЫЕ (по количеству символов), а не по реальному аудио. Из-за этого ролик
может встать не в тот момент, где о нём говорится.

ЧТО ДЕЛАЕТ ЭТОТ СКРИПТ
======================
ЭТАП 1 — ТАЙМ-КОДЫ (по смыслу):
  1) Транскрибирует аудио через OpenAI Whisper с таймкодами слов/сегментов
     (реальное "когда какое слово произносится").
  2) Выравнивает scenario.txt на транскрипт (difflib) -> у каждого предложения
     появляется реальный start/end на таймлайне аудио.
  3) Каждая сцена из generated_prompts.json знает свои sentence_indexes и block_id,
     значит получает реальный таймкод. Несколько сцен внутри блока распределяются
     по времени блока. Привязку "сцена <-> смысл текста" сделал генератор промптов,
     поэтому монтаж получается ПО СМЫСЛУ.
  4) Пишет video_timecodes.json + читаемый video_timecodes.txt.

ЭТАП 2 — МОНТАЖ:
  5) Через ffmpeg подгоняет каждый ролик под его слот (короче -> обрезка,
     длиннее -> фриз последнего кадра / замедление / луп), клеит по порядку и
     накладывает оригинальное аудио. Итоговая длина == длине аудио.
     Во время сборки печатается живая шкала готовности монтажа.

  Недостающие ролики (ещё не сгенерированные) по умолчанию (MISSING_MODE=stretch)
  НЕ дают чёрных вставок: их время делится поровну между соседними готовыми
  роликами — сосед показывает больше своего материала, а если слот превысит длину
  клипа, сосед замедляется (setpts, с ограничением MAX_SLOW_FACTOR, дальше фриз).
  MISSING_MODE=black возвращает старое поведение с чёрными слотами.

МУЛЬТИЯЗЫЧНОСТЬ (en / ru / es / pt)
===================================
Видео-ролики одни и те же для всех языков (визуал не зависит от языка) — меняются
только ОЗВУЧКА и ТАЙМИНГИ (одна фраза на разных языках звучит разное время).

Аудиофайлы в папке ПРОМПТЫ:
  - английский  — единственный аудиофайл, чьё имя НЕ начинается с "scenario"
                  (напр. paul_third_heaven@...mp3);
  - scenario_ru.mp3 / scenario_es.mp3 / scenario_pt.mp3 — адаптации.

Для каждого найденного языка скрипт считает СВОИ тайм-коды под его озвучку и
собирает отдельный монтаж:
  en -> final_montage.mp4 / video_timecodes.json
  ru -> final_montage_ru.mp4 / video_timecodes_ru.json   (и так для es, pt)

Как английские сцены ложатся под иностранную озвучку: иностранное аудио
переводится Whisper'ом в АНГЛИЙСКИЙ текст с таймкодами (audio.translations) и
выравнивается на английский scenario.txt тем же механизмом. Таким образом каждый
ролик встаёт под смысл нужной фразы именно с темпом речи этого языка.

Если final_montage.mp4 уже существует — английский монтаж считается готовым и
пропускается (собираются только недостающие языки). Пересобрать всё: --force.

РЕЖИМЫ ДЕГРАДАЦИИ (всё graceful):
  - нет OpenAI ключа / нет аудио для Whisper -> тайм-коды по символам (как в master);
  - scenario.txt расходится с аудио (или перевод не выровнялся) -> пропорциональный
    откат под длину именно этой озвучки;
  - нет scenario.txt -> раскидываем ролики равномерно по сегментам Whisper;
  - нет ffmpeg -> считаем и сохраняем только тайм-коды (этап 1), монтаж пропускаем.

ЗАПУСК
======
    export OPENAI_API_KEY="sk-..."          # или положить ключ в ПРОМПТЫ/openai_key.txt
    python audio_video_sync_montage.py                  # все языки: тайм-коды + монтаж
    python audio_video_sync_montage.py --langs ru,es    # только выбранные языки
    python audio_video_sync_montage.py --preview        # тест первой минуты каждого языка
    python audio_video_sync_montage.py --timecodes-only # только этап 1
    python audio_video_sync_montage.py --render-only     # только монтаж по готовым тайм-кодам
    python audio_video_sync_montage.py --force           # пересобрать даже готовые монтажи

Настройки таймингов и режима заполнения — через переменные окружения (см. CONFIG).
"""

from __future__ import annotations

import argparse
import difflib
import json
import math
import os
import re
import subprocess
import sys
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# =========================================================
# CONFIG
# =========================================================

# Та же базовая папка, что и в остальных скриптах конвейера.
BASE_DIR = Path(os.getenv("PROMPTS_BASE_DIR", "/Users/aleksandrtomilov/Desktop/ПРОМПТЫ"))
VIDEOS_DIR = BASE_DIR / "ВИДЕО"

SCENARIO_FILE = BASE_DIR / "scenario.txt"
PROMPTS_JSON = BASE_DIR / "generated_prompts.json"          # из master-скрипта

# Выходные файлы этапа 1.
TIMECODES_JSON = BASE_DIR / "video_timecodes.json"
TIMECODES_TXT = BASE_DIR / "video_timecodes.txt"
TRANSCRIPT_JSON = BASE_DIR / "audio_transcript.json"       # кэш транскрипта

# Выходные файлы этапа 2.
FINAL_VIDEO = BASE_DIR / "final_montage.mp4"
PREVIEW_VIDEO = BASE_DIR / "final_montage_preview.mp4"

# Режим предпросмотра: смонтировать только первые N секунд (чтобы быстро проверить
# результат, не рендеря весь ролик и не нагружая машину). 0/пусто -> выключено.
# Можно задать через env PREVIEW_SECONDS или флагом --preview [СЕКУНДЫ].
_PREVIEW_ENV = os.getenv("PREVIEW_SECONDS", "").strip()
PREVIEW_SECONDS_DEFAULT = 60.0

# OpenAI ключ: сначала env, потом локальный файл. НИКОГДА не хардкодим.
OPENAI_API_KEY = (os.getenv("OPENAI_API_KEY") or "").strip()
OPENAI_KEY_FILE = BASE_DIR / "openai_key.txt"
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "whisper-1")

# Оценка длительности по символам (fallback, идентична master-скрипту).
SECONDS_PER_100_CHARS = 7.8

# Каждый ролик по документации video-скрипта ~8 секунд.
CLIP_SECONDS = float(os.getenv("CLIP_SECONDS", "8"))

# Минимальная длительность слота под один ролик (чтобы кадры не мелькали).
MIN_SLOT_SECONDS = float(os.getenv("MIN_SLOT_SECONDS", "1.2"))

# Что делать, когда слот ДЛИННЕЕ ролика: freeze | slow | loop.
FILL_MODE = os.getenv("FILL_MODE", "freeze").strip().lower()

# Что делать с недостающими роликами (которые ещё не сгенерированы):
#   stretch — убрать дырку, а её время поделить поровну между соседними готовыми
#             роликами (сосед показывает больше своего материала; если слот
#             превысит длину клипа — сосед замедляется). ПО УМОЛЧАНИЮ.
#   black   — вставлять чёрный слот на месте недостающего ролика (старое поведение).
MISSING_MODE = os.getenv("MISSING_MODE", "stretch").strip().lower()

# Максимальный коэффициент замедления соседа при растягивании. Выше — уже
# слишком «слоу-мо», поэтому остаток добивается фризом последнего кадра.
MAX_SLOW_FACTOR = float(os.getenv("MAX_SLOW_FACTOR", "3.0"))

# Параметры финального рендера.
TARGET_FPS = int(os.getenv("TARGET_FPS", "30"))
# Пусто -> взять разрешение первого ролика; иначе, напр., "1920x1080".
TARGET_RESOLUTION = os.getenv("TARGET_RESOLUTION", "").strip()

# Порог качества выравнивания: доля слов сценария, нашедших место в транскрипте.
# Ниже порога считаем, что аудио не совпадает со сценарием -> пропорциональный откат.
ALIGN_MIN_COVERAGE = float(os.getenv("ALIGN_MIN_COVERAGE", "0.45"))

AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg"}
PREFERRED_AUDIO_NAMES = [
    "voice.mp3", "voice.wav", "voice.m4a", "voice.aac",
    "audio.mp3", "audio.wav", "audio.m4a", "audio.aac",
]


# =========================================================
# ЛОГ / УТИЛИТЫ
# =========================================================

def info(msg: str) -> None:
    print(f"[INFO] {msg}", flush=True)


def warn(msg: str) -> None:
    print(f"[WARN] {msg}", flush=True)


def fail(msg: str) -> None:
    print(f"[ERROR] {msg}", flush=True)
    sys.exit(1)


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def save_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def resolve_openai_key() -> str:
    if OPENAI_API_KEY:
        return OPENAI_API_KEY
    if OPENAI_KEY_FILE.exists():
        key = OPENAI_KEY_FILE.read_text(encoding="utf-8").strip()
        if key:
            return key
    return ""


def which(program: str) -> Optional[str]:
    from shutil import which as _which
    return _which(program)


def format_tc(seconds: float) -> str:
    """Секунды -> MM:SS.mmm для читаемых тайм-кодов."""
    if seconds < 0:
        seconds = 0.0
    m = int(seconds // 60)
    s = seconds - m * 60
    return f"{m:02d}:{s:06.3f}"


def progress_bar(done: int, total: int, width: int = 32) -> str:
    """ASCII-шкала готовности: [██████░░░░] 62.5% (done/total)."""
    total = max(1, total)
    frac = max(0.0, min(1.0, done / total))
    filled = int(round(frac * width))
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {frac * 100:5.1f}%  ({done}/{total} готово)"


# =========================================================
# АУДИО
# =========================================================

def _audio_files() -> List[Path]:
    if not BASE_DIR.exists():
        return []
    return sorted(
        [p for p in BASE_DIR.iterdir() if p.is_file() and p.suffix.lower() in AUDIO_EXTS],
        key=lambda x: x.name.lower(),
    )


def find_audio_file() -> Optional[Path]:
    """Совместимость: любое аудио (предпочитая voice/audio.*)."""
    for name in PREFERRED_AUDIO_NAMES:
        p = BASE_DIR / name
        if p.exists():
            return p
    files = _audio_files()
    return files[0] if files else None


# ---- Мультиязычные локали -----------------------------------------------

# Языки-адаптации: код -> ожидаемый stem аудиофайла (scenario_<code>).
LOCALE_AUDIO_STEMS = {"ru": "scenario_ru", "es": "scenario_es", "pt": "scenario_pt"}
# Порядок обработки.
LOCALE_ORDER = ["en", "ru", "es", "pt"]


@dataclass
class Locale:
    code: str
    audio: Path
    translate: bool           # переводить озвучку в EN для выравнивания на scenario.txt
    transcript_json: Path
    timecodes_json: Path
    timecodes_txt: Path
    out_video: Path
    preview_video: Path


def _suffixed(base_path: Path, code: str) -> Path:
    """final_montage.mp4 -> final_montage_ru.mp4 (для en путь без изменений)."""
    if code == "en":
        return base_path
    return base_path.with_name(f"{base_path.stem}_{code}{base_path.suffix}")


def find_english_audio() -> Optional[Path]:
    """Английская озвучка — единственный аудиофайл, чьё имя НЕ начинается с 'scenario'."""
    for p in _audio_files():
        if not p.stem.lower().startswith("scenario"):
            return p
    return None


def _make_locale(code: str, audio: Path, translate: bool) -> Locale:
    return Locale(
        code=code,
        audio=audio,
        translate=translate,
        transcript_json=_suffixed(TRANSCRIPT_JSON, code),
        timecodes_json=_suffixed(TIMECODES_JSON, code),
        timecodes_txt=_suffixed(TIMECODES_TXT, code),
        out_video=_suffixed(FINAL_VIDEO, code),
        preview_video=_suffixed(PREVIEW_VIDEO, code),
    )


def discover_locales(selected: Optional[List[str]] = None) -> List[Locale]:
    """
    Находит доступные локали по аудиофайлам в папке.
    en  -> аудио без префикса scenario; ru/es/pt -> scenario_<code>.<ext>.
    selected ограничивает набор языков (None -> все найденные).
    """
    locales: List[Locale] = []
    want = set(selected) if selected else None

    for code in LOCALE_ORDER:
        if want is not None and code not in want:
            continue
        if code == "en":
            audio = find_english_audio()
            if audio:
                locales.append(_make_locale("en", audio, translate=False))
        else:
            stem = LOCALE_AUDIO_STEMS[code]
            match = next((p for p in _audio_files() if p.stem.lower() == stem), None)
            if match:
                locales.append(_make_locale(code, match, translate=True))
    return locales


def ffprobe_duration(path: Path) -> Optional[float]:
    if not which("ffprobe"):
        return None
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True, text=True, check=True,
        )
        return float(result.stdout.strip())
    except Exception:
        return None


def ffmpeg_duration(path: Path) -> Optional[float]:
    """Длительность через парсинг вывода ffmpeg (когда ffprobe недоступен)."""
    if not which("ffmpeg"):
        return None
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-i", str(path)],
            capture_output=True, text=True,
        )
        m = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.?\d*)", result.stderr)
        if m:
            h, mi, s = m.groups()
            return int(h) * 3600 + int(mi) * 60 + float(s)
    except Exception:
        pass
    return None


def probe_duration(path: Path) -> Optional[float]:
    return ffprobe_duration(path) or ffmpeg_duration(path)


def ffprobe_resolution(path: Path) -> Optional[Tuple[int, int]]:
    if which("ffprobe"):
        try:
            result = subprocess.run(
                [
                    "ffprobe", "-v", "error",
                    "-select_streams", "v:0",
                    "-show_entries", "stream=width,height",
                    "-of", "csv=p=0:s=x",
                    str(path),
                ],
                capture_output=True, text=True, check=True,
            )
            w, h = result.stdout.strip().split("x")
            return int(w), int(h)
        except Exception:
            pass
    # fallback: парсим вывод ffmpeg
    if which("ffmpeg"):
        try:
            result = subprocess.run(
                ["ffmpeg", "-hide_banner", "-i", str(path)],
                capture_output=True, text=True,
            )
            m = re.search(r"Stream #\d+:\d+.*Video.*?,\s*(\d{2,5})x(\d{2,5})", result.stderr)
            if m:
                return int(m.group(1)), int(m.group(2))
        except Exception:
            pass
    return None


def wave_duration(path: Path) -> Optional[float]:
    if path.suffix.lower() != ".wav":
        return None
    try:
        with wave.open(str(path), "rb") as wf:
            return wf.getnframes() / float(wf.getframerate())
    except Exception:
        return None


def get_audio_duration(path: Optional[Path], scenario_text: str) -> float:
    if path is not None:
        dur = ffprobe_duration(path)
        if dur:
            info(f"Длина аудио (ffprobe): {dur:.2f} сек")
            return dur
        dur = wave_duration(path)
        if dur:
            info(f"Длина аудио (wave): {dur:.2f} сек")
            return dur
        dur = ffmpeg_duration(path)
        if dur:
            info(f"Длина аудио (ffmpeg): {dur:.2f} сек")
            return dur
        warn("Не удалось прочитать длину аудио — оцениваю по символам.")
    chars = len(scenario_text) if scenario_text else 0
    dur = max(1.0, (chars / 100.0) * SECONDS_PER_100_CHARS)
    info(f"Оценочная длина по символам: {dur:.2f} сек")
    return dur


# =========================================================
# СЦЕНАРИЙ -> ПРЕДЛОЖЕНИЯ
# (та же логика, что в master-скрипте, чтобы sentence_indexes совпадали)
# =========================================================

def split_into_sentences(text: str) -> List[str]:
    normalized = re.sub(r"\s+", " ", text).strip()
    parts = re.split(r"(?<=[.!?])\s+(?=[A-ZА-ЯЁ0-9\"'])", normalized)
    sentences = [p.strip() for p in parts if p.strip()]
    return sentences or ([normalized] if normalized else [])


@dataclass
class Sentence:
    index: int          # 1-based, совпадает с sentence_indexes в generated_prompts.json
    text: str
    start: float = 0.0
    end: float = 0.0
    aligned: bool = False


def build_sentences(scenario_text: str) -> List[Sentence]:
    return [
        Sentence(index=i, text=s)
        for i, s in enumerate(split_into_sentences(scenario_text), start=1)
    ]


# =========================================================
# ТРАНСКРИБАЦИЯ (WHISPER)
# =========================================================

@dataclass
class Word:
    text: str
    start: float
    end: float


@dataclass
class Segment:
    text: str
    start: float
    end: float


@dataclass
class Transcript:
    words: List[Word] = field(default_factory=list)
    segments: List[Segment] = field(default_factory=list)
    language: str = ""


# Лимит OpenAI на файл транскрибации — 25 МБ. Берём запас.
WHISPER_MAX_BYTES = 24 * 1024 * 1024
# На сколько секунд резать сжатое аудио, если оно всё ещё больше лимита.
WHISPER_CHUNK_SECONDS = int(os.getenv("WHISPER_CHUNK_SECONDS", "1200"))  # 20 мин


def _run_whisper_file(client: Any, path: Path, translate: bool = False) -> Dict[str, Any]:
    """
    Один вызов Whisper по файлу.
    translate=False: транскрибация на языке оригинала, word+segment (фолбэк на segment).
    translate=True:  перевод озвучки в АНГЛИЙСКИЙ текст с таймкодами (только segment) —
                     нужно, чтобы иностранную озвучку можно было выровнять на английский
                     scenario.txt тем же механизмом.
    """
    if translate:
        # Endpoint переводов даёт только сегментные таймкоды (word-granularity не поддерживает).
        with path.open("rb") as f:
            resp = client.audio.translations.create(
                model=WHISPER_MODEL, file=f, response_format="verbose_json",
            )
        return resp.model_dump() if hasattr(resp, "model_dump") else dict(resp)

    try:
        with path.open("rb") as f:
            resp = client.audio.transcriptions.create(
                model=WHISPER_MODEL, file=f, response_format="verbose_json",
                timestamp_granularities=["word", "segment"],
            )
    except Exception as e:
        warn(f"Word-таймкоды не сработали ({str(e)[:120]}). Пробую только сегменты.")
        with path.open("rb") as f:
            resp = client.audio.transcriptions.create(
                model=WHISPER_MODEL, file=f, response_format="verbose_json",
            )
    return resp.model_dump() if hasattr(resp, "model_dump") else dict(resp)


def _compress_audio_for_whisper(src: Path, out: Path) -> Optional[Path]:
    """Сжимает в моно 16кГц mp3 32кбит — Whisper'у этого достаточно, файл в разы меньше."""
    if not which("ffmpeg"):
        return None
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(src), "-ac", "1", "-ar", "16000",
        "-c:a", "libmp3lame", "-b:a", "32k", str(out),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        warn(f"Сжатие аудио не удалось: {e.stderr[:200]}")
        return None
    return out if out.exists() and out.stat().st_size > 0 else None


def _split_audio_into_chunks(src: Path, chunk_seconds: int, tmp_dir: Path) -> List[Tuple[Path, float]]:
    """Режет аудио на части; возвращает [(файл, смещение_в_секундах), ...]."""
    if not which("ffmpeg"):
        return []
    pattern = str(tmp_dir / "chunk_%03d.mp3")
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(src), "-f", "segment", "-segment_time", str(chunk_seconds),
        "-c", "copy", pattern,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        warn(f"Нарезка аудио не удалась: {e.stderr[:200]}")
        return []
    chunks = sorted(tmp_dir.glob("chunk_*.mp3"))
    result: List[Tuple[Path, float]] = []
    offset = 0.0
    for c in chunks:
        result.append((c, offset))
        offset += probe_duration(c) or chunk_seconds
    return result


def transcribe_audio(audio_path: Path, api_key: str,
                     cache_path: Path = TRANSCRIPT_JSON, translate: bool = False) -> Optional[Transcript]:
    """
    Whisper с таймкодами. Кэшируется в cache_path.
    translate=True переводит иностранную озвучку в английский текст с таймкодами.
    Большие файлы (>25 МБ) автоматически сжимаются, а очень длинные — режутся
    на части, таймкоды которых затем сшиваются со смещением.
    """
    cached = load_json(cache_path, None)
    if isinstance(cached, dict) and cached.get("audio_name") == audio_path.name \
            and bool(cached.get("translated")) == translate:
        info(f"Использую кэш транскрипта: {cache_path.name}")
        return _transcript_from_dict(cached)

    if not api_key:
        warn("Нет OpenAI ключа — транскрибация пропущена.")
        return None

    try:
        from openai import OpenAI
    except Exception as e:
        warn(f"Не установлен пакет openai ({e}) — транскрибация пропущена.")
        return None

    client = OpenAI(api_key=api_key)
    import shutil
    tmp_dir = BASE_DIR / f"_whisper_tmp_{cache_path.stem}"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # 1) Если файл больше лимита — сжимаем.
    src = audio_path
    size_mb = src.stat().st_size / (1024 * 1024)
    if src.stat().st_size > WHISPER_MAX_BYTES:
        info(f"Аудио {size_mb:.1f} МБ > лимита Whisper 25 МБ — сжимаю (моно 16кГц)...")
        compressed = _compress_audio_for_whisper(src, tmp_dir / "whisper_input.mp3")
        if compressed:
            src = compressed
            info(f"Сжато до {src.stat().st_size / (1024 * 1024):.1f} МБ.")
        else:
            warn("Не удалось сжать (нет ffmpeg?) — попробую отправить как есть.")

    # 2) Формируем список файлов для отправки (при необходимости — с нарезкой).
    if src.stat().st_size <= WHISPER_MAX_BYTES:
        files_with_offset: List[Tuple[Path, float]] = [(src, 0.0)]
    else:
        info(f"Всё ещё {src.stat().st_size / (1024*1024):.1f} МБ — режу на части по "
             f"{WHISPER_CHUNK_SECONDS//60} мин...")
        files_with_offset = _split_audio_into_chunks(src, WHISPER_CHUNK_SECONDS, tmp_dir)
        if not files_with_offset:
            warn("Не удалось подготовить аудио для Whisper.")
            shutil.rmtree(tmp_dir, ignore_errors=True)
            return None

    # 3) Транскрибируем/переводим каждый файл и сшиваем таймкоды со смещением.
    mode = "перевод в EN" if translate else "транскрибация"
    info(f"Whisper ({mode}) {audio_path.name}, частей: {len(files_with_offset)}...")
    all_words: List[Word] = []
    all_segments: List[Segment] = []
    language = ""
    ok = False
    for idx, (fpath, offset) in enumerate(files_with_offset, start=1):
        try:
            data = _run_whisper_file(client, fpath, translate=translate)
        except Exception as e:
            warn(f"Часть {idx}/{len(files_with_offset)} не обработана: {str(e)[:160]}")
            continue
        part = _transcript_from_dict(data)
        language = language or part.language
        for w in part.words:
            all_words.append(Word(text=w.text, start=w.start + offset, end=w.end + offset))
        for s in part.segments:
            all_segments.append(Segment(text=s.text, start=s.start + offset, end=s.end + offset))
        ok = True
        if len(files_with_offset) > 1:
            info(f"  часть {idx}/{len(files_with_offset)} готова "
                 f"(+{len(part.words)} слов)")

    shutil.rmtree(tmp_dir, ignore_errors=True)

    if not ok:
        warn("Транскрибация не удалась ни для одной части.")
        return None

    tr = Transcript(words=all_words, segments=all_segments, language=language)
    save_json(cache_path, {
        "audio_name": audio_path.name,
        "translated": translate,
        "language": tr.language,
        "words": [w.__dict__ for w in tr.words],
        "segments": [s.__dict__ for s in tr.segments],
    })
    info(f"Транскрипт: слов={len(tr.words)}, сегментов={len(tr.segments)}, язык={tr.language or '?'}")
    return tr


def _transcript_from_dict(data: Dict[str, Any]) -> Transcript:
    words = []
    for w in (data.get("words") or []):
        try:
            words.append(Word(text=str(w.get("word", w.get("text", ""))),
                              start=float(w["start"]), end=float(w["end"])))
        except Exception:
            continue
    segments = []
    for s in (data.get("segments") or []):
        try:
            segments.append(Segment(text=str(s.get("text", "")),
                                    start=float(s["start"]), end=float(s["end"])))
        except Exception:
            continue
    return Transcript(words=words, segments=segments, language=str(data.get("language", "")))


# =========================================================
# ВЫРАВНИВАНИЕ СЦЕНАРИЯ НА ТРАНСКРИПТ
# =========================================================

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def align_sentences_to_transcript(sentences: List[Sentence], tr: Transcript) -> float:
    """
    Проставляет sentence.start/end по реальному аудио через выравнивание токенов
    сценария на слова транскрипта (difflib). Возвращает coverage (0..1) —
    долю токенов сценария, уверенно привязанных к времени.

    Если у транскрипта нет пословных таймкодов, используем сегменты как
    "псевдо-слова" (текст сегмента с равномерным распределением времени).
    """
    # 1) Поток слов транскрипта с таймингами.
    t_words: List[Word] = []
    if tr.words:
        t_words = tr.words
    elif tr.segments:
        # Разворачиваем каждый сегмент в слова с равномерным временем.
        for seg in tr.segments:
            seg_tokens = _TOKEN_RE.findall(seg.text)
            if not seg_tokens:
                continue
            span = max(0.01, seg.end - seg.start)
            step = span / len(seg_tokens)
            for k, tok in enumerate(seg_tokens):
                ws = seg.start + k * step
                t_words.append(Word(text=tok, start=ws, end=ws + step))
    if not t_words:
        return 0.0

    t_tokens = [w.text.lower().strip() for w in t_words]
    t_tokens = [t for t in t_tokens]  # keep 1:1 with t_words

    # 2) Поток токенов сценария с указателем на предложение.
    s_tokens: List[str] = []
    s_owner: List[int] = []  # индекс предложения в sentences (0-based)
    for si, sent in enumerate(sentences):
        for tok in tokenize(sent.text):
            s_tokens.append(tok)
            s_owner.append(si)
    if not s_tokens:
        return 0.0

    # 3) Выравнивание последовательностей токенов.
    matcher = difflib.SequenceMatcher(a=s_tokens, b=t_tokens, autojunk=False)
    # Для каждого токена сценария — время (start) сопоставленного слова транскрипта.
    s_time: List[Optional[float]] = [None] * len(s_tokens)
    matched = 0
    for block in matcher.get_matching_blocks():
        for off in range(block.size):
            si = block.a + off
            ti = block.b + off
            s_time[si] = t_words[ti].start
            matched += 1

    # coverage = доля произнесённых слов, нашедших место в сценарии.
    # Именно "качество распознавания речи в тексте", а не доля всего сценария —
    # так метрика остаётся честной, даже если транскрибируется лишь часть аудио
    # (например в режиме предпросмотра первой минуты).
    coverage = matched / max(1, len(t_tokens))

    # 4) Заполняем пропуски линейной интерполяцией между известными точками.
    _interpolate_times(s_time)

    # 5) Считаем start/end каждого предложения по его токенам.
    if all(v is None for v in s_time):
        return coverage

    for si, sent in enumerate(sentences):
        times = [s_time[i] for i in range(len(s_tokens)) if s_owner[i] == si and s_time[i] is not None]
        if times:
            sent.start = min(times)
            sent.end = max(times)
            sent.aligned = True

    # 6) Делаем тайминги монотонными и без нулевых длительностей.
    _make_sentences_monotonic(sentences, t_words[-1].end if t_words else 0.0)
    return coverage


def _interpolate_times(times: List[Optional[float]]) -> None:
    n = len(times)
    # первый/последний якорь
    first = next((i for i in range(n) if times[i] is not None), None)
    last = next((i for i in range(n - 1, -1, -1) if times[i] is not None), None)
    if first is None:
        return
    for i in range(first):
        times[i] = times[first]
    for i in range(last + 1, n):
        times[i] = times[last]
    # внутренние дырки
    i = first
    while i <= last:
        if times[i] is not None:
            i += 1
            continue
        prev = i - 1
        nxt = i
        while nxt <= last and times[nxt] is None:
            nxt += 1
        t0 = times[prev]
        t1 = times[nxt]
        gap = nxt - prev
        for k in range(prev + 1, nxt):
            times[k] = t0 + (t1 - t0) * (k - prev) / gap
        i = nxt


def _make_sentences_monotonic(sentences: List[Sentence], audio_end: float) -> None:
    prev_end = 0.0
    for sent in sentences:
        if sent.start < prev_end:
            sent.start = prev_end
        if sent.end <= sent.start:
            sent.end = sent.start + 0.01
        prev_end = sent.end
    if audio_end and sentences and sentences[-1].end > audio_end:
        sentences[-1].end = audio_end


def estimate_sentences_proportional(sentences: List[Sentence], audio_seconds: float) -> None:
    """Fallback: распределяем время предложений пропорционально длине (как master)."""
    total_chars = sum(max(len(s.text), 1) for s in sentences) or 1
    t = 0.0
    for sent in sentences:
        share = max(len(sent.text), 1) / total_chars
        dur = max(0.4, audio_seconds * share)
        sent.start = t
        sent.end = t + dur
        sent.aligned = False
        t += dur
    # нормируем к длине аудио
    if sentences and sentences[-1].end > 0:
        scale = audio_seconds / sentences[-1].end
        for sent in sentences:
            sent.start *= scale
            sent.end *= scale


# =========================================================
# СЦЕНЫ -> ТАЙМ-КОДЫ
# =========================================================

@dataclass
class SceneTC:
    global_scene_index: int
    block_id: int
    scene_index_in_block: int
    sentence_indexes: List[int]
    block_text: str
    subject: str
    video_file: str
    anchor: float = 0.0     # желаемая точка начала (по смыслу)
    start: float = 0.0
    end: float = 0.0
    exists: bool = False
    fill_mode: str = ""     # переопределение FILL_MODE для этого клипа ("slow" у растянутых)
    stretched: bool = False  # клип поглотил время недостающего соседа


def load_scenes() -> List[SceneTC]:
    data = load_json(PROMPTS_JSON, None)
    if not isinstance(data, list) or not data:
        return []
    scenes: List[SceneTC] = []
    for rec in data:
        if not isinstance(rec, dict):
            continue
        gsi = int(rec.get("global_scene_index") or (len(scenes) + 1))
        video_name = f"{gsi:04d}.mp4"
        scenes.append(
            SceneTC(
                global_scene_index=gsi,
                block_id=int(rec.get("block_id") or 0),
                scene_index_in_block=int(rec.get("scene_index_in_block") or 1),
                sentence_indexes=[int(x) for x in (rec.get("sentence_indexes") or []) if str(x).isdigit()],
                block_text=str(rec.get("block_text") or ""),
                subject=str(rec.get("subject") or ""),
                video_file=video_name,
                exists=(VIDEOS_DIR / video_name).exists(),
            )
        )
    scenes.sort(key=lambda s: s.global_scene_index)
    return scenes


def scenes_from_videos_only() -> List[SceneTC]:
    """Если нет generated_prompts.json — берём ролики как есть, по порядку имён."""
    if not VIDEOS_DIR.exists():
        return []
    vids = sorted(
        [p for p in VIDEOS_DIR.iterdir() if p.is_file() and p.suffix.lower() == ".mp4"],
        key=lambda p: (0, int(p.stem)) if p.stem.isdigit() else (1, p.stem.lower()),
    )
    scenes = []
    for i, p in enumerate(vids, start=1):
        gsi = int(p.stem) if p.stem.isdigit() else i
        scenes.append(
            SceneTC(
                global_scene_index=gsi,
                block_id=0,
                scene_index_in_block=1,
                sentence_indexes=[],
                block_text="",
                subject="",
                video_file=p.name,
                exists=True,
            )
        )
    scenes.sort(key=lambda s: s.global_scene_index)
    return scenes


def block_time_range(sentence_indexes: List[int], sentences_by_idx: Dict[int, Sentence]) -> Optional[Tuple[float, float]]:
    times = [(sentences_by_idx[i].start, sentences_by_idx[i].end)
             for i in sentence_indexes if i in sentences_by_idx]
    if not times:
        return None
    return min(t[0] for t in times), max(t[1] for t in times)


def compute_anchors_from_sentences(scenes: List[SceneTC], sentences: List[Sentence]) -> None:
    """
    Якорь сцены = позиция внутри временного диапазона её блока.
    Несколько сцен блока распределяются равномерно по диапазону блока.
    """
    by_idx = {s.index: s for s in sentences}
    # сгруппировать сцены по блоку, чтобы знать N и k
    from collections import defaultdict
    per_block: Dict[int, List[SceneTC]] = defaultdict(list)
    for sc in scenes:
        per_block[sc.block_id].append(sc)

    for sc in scenes:
        rng = block_time_range(sc.sentence_indexes, by_idx)
        if rng is None:
            sc.anchor = -1.0  # заполним позже
            continue
        b_start, b_end = rng
        block_scenes = sorted(per_block[sc.block_id], key=lambda x: (x.scene_index_in_block, x.global_scene_index))
        n = len(block_scenes)
        k = block_scenes.index(sc)
        sc.anchor = b_start + (k / n) * max(0.0, b_end - b_start)


def compute_anchors_from_segments(scenes: List[SceneTC], tr: Optional[Transcript], audio_seconds: float) -> None:
    """Нет привязки к тексту: равномерно раскидываем ролики по сегментам/длине аудио."""
    n = len(scenes)
    if n == 0:
        return
    if tr and tr.segments:
        start = tr.segments[0].start
        end = tr.segments[-1].end
    else:
        start, end = 0.0, audio_seconds
    span = max(0.1, end - start)
    for i, sc in enumerate(scenes):
        sc.anchor = start + (i / n) * span


def finalize_timecodes(scenes: List[SceneTC], audio_seconds: float) -> None:
    """
    Из якорей делаем непрерывные слоты: start = anchor, end = следующий anchor.
    Гарантируем монотонность, минимальную длину и покрытие [0, audio_seconds].
    """
    if not scenes:
        return

    # заполнить отсутствующие якоря (сцены без sentence_indexes) интерполяцией
    anchors: List[Optional[float]] = [sc.anchor if sc.anchor >= 0 else None for sc in scenes]
    # края
    if anchors[0] is None:
        anchors[0] = 0.0
    if anchors[-1] is None:
        anchors[-1] = audio_seconds
    _interpolate_times(anchors)
    for sc, a in zip(scenes, anchors):
        sc.anchor = max(0.0, float(a if a is not None else 0.0))

    # монотонные якоря
    prev = 0.0
    for sc in scenes:
        if sc.anchor < prev:
            sc.anchor = prev
        prev = sc.anchor

    # первый ролик всегда с нуля, чтобы не было чёрного вступления
    scenes[0].anchor = 0.0

    # границы: start_i = anchor_i, end_i = anchor_{i+1}
    for i, sc in enumerate(scenes):
        sc.start = scenes[i].anchor
        sc.end = scenes[i + 1].anchor if i + 1 < len(scenes) else audio_seconds

    # минимальная длительность: если слот слишком короткий — раздвигаем за счёт сдвига
    for i, sc in enumerate(scenes):
        if sc.end - sc.start < MIN_SLOT_SECONDS:
            sc.end = sc.start + MIN_SLOT_SECONDS
            if i + 1 < len(scenes) and scenes[i + 1].start < sc.end:
                scenes[i + 1].start = sc.end

    # финальная зачистка монотонности и хвоста
    prev_end = 0.0
    for sc in scenes:
        if sc.start < prev_end:
            sc.start = prev_end
        if sc.end <= sc.start:
            sc.end = sc.start + MIN_SLOT_SECONDS
        prev_end = sc.end
    scenes[-1].end = max(scenes[-1].start + MIN_SLOT_SECONDS, audio_seconds)


# =========================================================
# ЭТАП 1: ТАЙМ-КОДЫ
# =========================================================

def load_master_scenes() -> List[SceneTC]:
    """Общие для всех языков сцены из generated_prompts.json (или из папки ВИДЕО)."""
    scenes = load_scenes()
    source = "generated_prompts.json"
    if not scenes:
        warn("generated_prompts.json не найден/пуст — беру ролики из папки ВИДЕО по порядку.")
        scenes = scenes_from_videos_only()
        source = "videos-folder"
    if not scenes:
        fail("Нет ни generated_prompts.json, ни роликов в ВИДЕО — нечего синхронизировать.")
    info(f"Сцен/роликов: {len(scenes)} (источник: {source})")
    missing = [sc.video_file for sc in scenes if not sc.exists]
    if missing:
        warn(f"Не хватает {len(missing)} из {len(scenes)} роликов: {', '.join(missing[:8])}"
             f"{'...' if len(missing) > 8 else ''}")
        if MISSING_MODE == "stretch":
            info("Режим MISSING_MODE=stretch: дырки закрою растягиванием соседних роликов.")
        else:
            info(f"Режим MISSING_MODE={MISSING_MODE}: на месте недостающих будет чёрный слот.")
    return scenes


def build_timecodes(loc: Locale, scenario_text: str, scenes_master: List[SceneTC]) -> List[SceneTC]:
    """
    Считает тайм-коды под озвучку конкретной локали. scenes_master — общие сцены
    (визуал одинаков для всех языков). Для иностранных локалей аудио переводится
    Whisper'ом в английский и выравнивается на английский scenario.txt, поэтому
    ролики ложатся под смысл фраз именно ЭТОГО языка с его темпом речи.
    """
    scenes = [_copy_scene(sc) for sc in scenes_master]
    audio_seconds = get_audio_duration(loc.audio, scenario_text)
    info(f"[{loc.code}] аудио: {loc.audio.name} | {audio_seconds:.1f}s"
         f"{' | перевод в EN' if loc.translate else ''}")

    api_key = resolve_openai_key()
    tr = transcribe_audio(loc.audio, api_key, cache_path=loc.transcript_json,
                          translate=loc.translate) if api_key else None

    strategy = "proportional"
    if scenario_text and scenes and scenes[0].sentence_indexes:
        sentences = build_sentences(scenario_text)
        max_needed = max((max(sc.sentence_indexes) for sc in scenes if sc.sentence_indexes), default=0)
        if max_needed > len(sentences):
            warn(f"sentence_indexes ссылаются на {max_needed}, а предложений {len(sentences)} — "
                 f"сценарий мог измениться после генерации промптов.")

        coverage = 0.0
        if tr:
            coverage = align_sentences_to_transcript(sentences, tr)
            info(f"[{loc.code}] выравнивание на аудио: coverage={coverage:.0%}")
        if tr and coverage >= ALIGN_MIN_COVERAGE:
            strategy = "whisper-translate-align" if loc.translate else "whisper-align"
        else:
            if tr:
                warn(f"[{loc.code}] coverage {coverage:.0%} < порога {ALIGN_MIN_COVERAGE:.0%} — "
                     f"откат на пропорциональные тайминги под длину этого аудио.")
            estimate_sentences_proportional(sentences, audio_seconds)
            strategy = "proportional"
        compute_anchors_from_sentences(scenes, sentences)
    else:
        compute_anchors_from_segments(scenes, tr, audio_seconds)
        strategy = "whisper-segments" if (tr and tr.segments) else "uniform"

    info(f"[{loc.code}] стратегия таймингов: {strategy}")
    finalize_timecodes(scenes, audio_seconds)
    save_timecodes(scenes, audio_seconds, strategy, loc)
    return scenes


def save_timecodes(scenes: List[SceneTC], audio_seconds: float, strategy: str, loc: Locale) -> None:
    payload = {
        "locale": loc.code,
        "audio_file": loc.audio.name,
        "audio_seconds": round(audio_seconds, 3),
        "strategy": strategy,
        "clip_seconds": CLIP_SECONDS,
        "fill_mode": FILL_MODE,
        "scenes": [
            {
                "global_scene_index": sc.global_scene_index,
                "video_file": sc.video_file,
                "exists": sc.exists,
                "block_id": sc.block_id,
                "scene_index_in_block": sc.scene_index_in_block,
                "sentence_indexes": sc.sentence_indexes,
                "start": round(sc.start, 3),
                "end": round(sc.end, 3),
                "duration": round(sc.end - sc.start, 3),
                "subject": sc.subject,
                "block_text": sc.block_text,
            }
            for sc in scenes
        ],
    }
    save_json(loc.timecodes_json, payload)

    lines = [
        f"# Тайм-коды [{loc.code}] | аудио={loc.audio.name} | {audio_seconds:.2f}s | стратегия={strategy}",
        f"# файл    старт --> конец  (длина)  | блок/сцена | текст",
        "",
    ]
    for sc in scenes:
        flag = "" if sc.exists else "  [!] нет файла"
        preview = (sc.block_text or sc.subject or "").strip().replace("\n", " ")
        if len(preview) > 70:
            preview = preview[:67] + "..."
        lines.append(
            f"{sc.video_file}  {format_tc(sc.start)} --> {format_tc(sc.end)}  "
            f"({sc.end - sc.start:4.1f}s)  | b{sc.block_id}.s{sc.scene_index_in_block} | {preview}{flag}"
        )
    loc.timecodes_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    info(f"[{loc.code}] сохранено: {loc.timecodes_json.name}, {loc.timecodes_txt.name}")


# =========================================================
# ЭТАП 2: МОНТАЖ (ffmpeg)
# =========================================================

def parse_target_resolution(scenes: List[SceneTC]) -> Tuple[int, int]:
    if TARGET_RESOLUTION:
        try:
            w, h = TARGET_RESOLUTION.lower().split("x")
            return int(w), int(h)
        except Exception:
            warn(f"TARGET_RESOLUTION={TARGET_RESOLUTION!r} некорректно — беру из первого ролика.")
    for sc in scenes:
        if sc.exists:
            res = ffprobe_resolution(VIDEOS_DIR / sc.video_file)
            if res:
                return res
    return 1920, 1080


def build_clip_filter(slot: float, src_dur: Optional[float], w: int, h: int, fill_mode: str) -> str:
    """
    Фильтр видео: масштаб с сохранением пропорций + паддинг до WxH + fps + SAR.
    Заполнение слота, который ДЛИННЕЕ ролика, по fill_mode: slow | loop | freeze.
    """
    base = (
        f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
        f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={TARGET_FPS}"
    )
    src = src_dur if (src_dur and src_dur > 0) else CLIP_SECONDS

    if slot <= src + 0.05:
        # слот короче/равен ролику -> просто обрежем по -t (покажем часть клипа)
        return base

    # слот длиннее ролика
    if fill_mode == "slow":
        factor = slot / src
        if factor <= MAX_SLOW_FACTOR:
            # замедляем весь клип ровно под слот
            return f"setpts={factor:.5f}*PTS," + base
        # замедляем до предела, остаток добиваем фризом последнего кадра
        slowed = src * MAX_SLOW_FACTOR
        extra = slot - slowed
        return (f"setpts={MAX_SLOW_FACTOR:.5f}*PTS," + base +
                f",tpad=stop_mode=clone:stop_duration={extra:.3f}")
    if fill_mode == "loop":
        # луп реализуем на входе (-stream_loop), фильтр обычный
        return base
    # freeze (по умолчанию): доигрываем и замораживаем последний кадр
    extra = slot - src
    return base + f",tpad=stop_mode=clone:stop_duration={extra:.3f}"


def render_clip(sc: SceneTC, w: int, h: int, tmp_dir: Path) -> Optional[Path]:
    src_path = VIDEOS_DIR / sc.video_file
    slot = max(MIN_SLOT_SECONDS, sc.end - sc.start)
    out = tmp_dir / f"seg_{sc.global_scene_index:04d}.mp4"
    src_dur = probe_duration(src_path)
    fill_mode = sc.fill_mode or FILL_MODE

    input_args: List[str] = []
    if fill_mode == "loop" and src_dur and slot > src_dur + 0.05:
        loops = int(math.ceil(slot / src_dur))
        input_args = ["-stream_loop", str(loops)]

    vf = build_clip_filter(slot, src_dur, w, h, fill_mode)
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        *input_args,
        "-i", str(src_path),
        "-t", f"{slot:.3f}",
        "-vf", vf,
        "-an",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-pix_fmt", "yuv420p",
        str(out),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        warn(f"Не удалось обработать {sc.video_file}: {e.stderr[:300]}")
        return None
    return out


def render_black(slot: float, w: int, h: int, out: Path) -> Optional[Path]:
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", f"color=c=black:s={w}x{h}:r={TARGET_FPS}",
        "-t", f"{max(MIN_SLOT_SECONDS, slot):.3f}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-pix_fmt", "yuv420p",
        str(out),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError:
        return None
    return out


def _copy_scene(sc: SceneTC) -> SceneTC:
    return SceneTC(
        global_scene_index=sc.global_scene_index, block_id=sc.block_id,
        scene_index_in_block=sc.scene_index_in_block,
        sentence_indexes=list(sc.sentence_indexes), block_text=sc.block_text,
        subject=sc.subject, video_file=sc.video_file, anchor=sc.anchor,
        start=sc.start, end=sc.end, exists=sc.exists,
        fill_mode=sc.fill_mode, stretched=sc.stretched,
    )


def redistribute_missing_slots(scenes: List[SceneTC]) -> List[SceneTC]:
    """
    Убирает недостающие ролики и делит их время между соседними готовыми:
    время сплошного «провала» из отсутствующих роликов делится ПОПОЛАМ между
    ближайшим готовым слева и ближайшим готовым справа (если сосед только с
    одной стороны — забирает всё). Растянутые соседи помечаются fill_mode="slow",
    чтобы при выходе за длину клипа замедляться, а не мигать чёрным.

    Тайминги остаются непрерывными, суммарная длина сохраняется -> синхрон с аудио.
    Возвращает новый список только из готовых сцен (в исходном порядке).
    """
    out = [_copy_scene(sc) for sc in scenes if sc.exists]
    if not out or all(sc.exists for sc in scenes):
        return out

    by_gsi = {c.global_scene_index: c for c in out}

    i = 0
    n = len(scenes)
    while i < n:
        if scenes[i].exists:
            i += 1
            continue
        # накопить сплошной провал недостающих [i..j)
        j = i
        while j < n and not scenes[j].exists:
            j += 1
        run = scenes[i:j]
        run_start = run[0].start
        run_end = run[-1].end
        gap = max(0.0, run_end - run_start)

        prev_exist = scenes[i - 1] if i - 1 >= 0 and scenes[i - 1].exists else None
        next_exist = scenes[j] if j < n and scenes[j].exists else None
        prev_copy = by_gsi.get(prev_exist.global_scene_index) if prev_exist else None
        next_copy = by_gsi.get(next_exist.global_scene_index) if next_exist else None

        if prev_copy and next_copy:
            mid = run_start + gap / 2.0
            prev_copy.end = mid
            next_copy.start = mid
            prev_copy.stretched = next_copy.stretched = True
            prev_copy.fill_mode = next_copy.fill_mode = "slow"
        elif prev_copy:
            prev_copy.end = run_end
            prev_copy.stretched = True
            prev_copy.fill_mode = "slow"
        elif next_copy:
            next_copy.start = run_start
            next_copy.stretched = True
            next_copy.fill_mode = "slow"
        # если соседей нет вообще (весь ролик пуст) — просто пропускаем провал
        i = j

    return out


def select_preview_scenes(scenes: List[SceneTC], preview_seconds: float) -> List[SceneTC]:
    """
    Оставляет только сцены, попадающие в окно [0, preview_seconds].
    Пограничную сцену обрезает ровно по границе. Возвращает копии (оригиналы не трогаем).
    """
    selected: List[SceneTC] = []
    for sc in scenes:
        if sc.start >= preview_seconds:
            break
        clip = _copy_scene(sc)
        clip.end = min(sc.end, preview_seconds)
        selected.append(clip)
    return selected


def render_montage(scenes: List[SceneTC], loc: Locale, preview_seconds: Optional[float] = None) -> None:
    if not which("ffmpeg"):
        warn("ffmpeg не найден — монтаж пропущен. Тайм-коды сохранены, "
             "монтаж можно собрать позже, установив ffmpeg. (ffprobe необязателен.)")
        return

    audio_path = loc.audio
    if not audio_path or not audio_path.exists():
        warn(f"[{loc.code}] нет аудио — монтаж пропускаю.")
        return

    # Недостающие ролики: либо растягиваем соседей (stretch), либо чёрные вставки.
    missing_total = sum(1 for sc in scenes if not sc.exists)
    if missing_total and MISSING_MODE == "stretch":
        scenes = redistribute_missing_slots(scenes)
        stretched = sum(1 for sc in scenes if sc.stretched)
        info(f"[{loc.code}] растягиваю соседей вместо {missing_total} недостающих роликов "
             f"(затронуто клипов: {stretched}).")

    out_video = loc.out_video
    limit_seconds: Optional[float] = None
    if preview_seconds and preview_seconds > 0:
        if not scenes or scenes[0].start >= preview_seconds:
            warn(f"[{loc.code}] в окне предпросмотра {preview_seconds:.0f}s нет ни одной сцены.")
            return
        full_count = len(scenes)
        scenes = select_preview_scenes(scenes, preview_seconds)
        limit_seconds = min(preview_seconds, scenes[-1].end)
        out_video = loc.preview_video
        info(f"[{loc.code}] ПРЕДПРОСМОТР: первые {limit_seconds:.1f}s — {len(scenes)} из {full_count} сцен.")

    w, h = parse_target_resolution(scenes)
    info(f"[{loc.code}] рендер {len(scenes)} сегментов в {w}x{h}@{TARGET_FPS}, заполнение={FILL_MODE}")

    tmp_dir = BASE_DIR / f"_montage_tmp_{loc.code}"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    total = len(scenes)
    total_dur = sum(max(MIN_SLOT_SECONDS, sc.end - sc.start) for sc in scenes) or 1.0
    done_dur = 0.0
    segments: List[Path] = []
    for i, sc in enumerate(scenes, start=1):
        slot = max(MIN_SLOT_SECONDS, sc.end - sc.start)
        if sc.exists:
            seg = render_clip(sc, w, h, tmp_dir)
        else:
            seg = render_black(slot, w, h, tmp_dir / f"seg_{sc.global_scene_index:04d}.mp4")
        if seg:
            segments.append(seg)
        done_dur += slot
        # живая шкала готовности монтажа (по доле собранного таймлайна)
        tag = "растянут" if sc.stretched else ("чёрный" if not sc.exists else "")
        sys.stdout.write(
            f"\r[{loc.code}] Монтаж {progress_bar(i, total)}  {format_tc(done_dur)}/{format_tc(total_dur)}  "
            f"[{sc.video_file}{(' ' + tag) if tag else ''}]        "
        )
        sys.stdout.flush()
    sys.stdout.write("\n")
    sys.stdout.flush()

    if not segments:
        warn("Не удалось подготовить ни одного сегмента — монтаж отменён.")
        return

    # concat demuxer
    concat_file = tmp_dir / "concat.txt"
    concat_file.write_text(
        "".join(f"file '{seg.as_posix()}'\n" for seg in segments), encoding="utf-8"
    )
    silent_video = tmp_dir / "video_no_audio.mp4"
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-f", "concat", "-safe", "0", "-i", str(concat_file),
                "-c", "copy", str(silent_video),
            ],
            check=True, capture_output=True, text=True,
        )
    except subprocess.CalledProcessError as e:
        warn(f"Ошибка склейки, пробую с перекодировкой: {e.stderr[:200]}")
        subprocess.run(
            [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-f", "concat", "-safe", "0", "-i", str(concat_file),
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                "-pix_fmt", "yuv420p", str(silent_video),
            ],
            check=True, capture_output=True, text=True,
        )

    # накладываем оригинальное аудио (в предпросмотре ограничиваем длину по -t)
    mux_cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(silent_video),
        "-i", str(audio_path),
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-shortest",
    ]
    if limit_seconds:
        mux_cmd += ["-t", f"{limit_seconds:.3f}"]
    mux_cmd.append(str(out_video))
    try:
        subprocess.run(mux_cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        fail(f"Не удалось наложить аудио: {e.stderr[:300]}")

    info(f"[{loc.code}] [DONE] Готовый монтаж: {out_video}")
    info(f"[{loc.code}] промежуточные файлы: {tmp_dir} (можно удалить).")


# =========================================================
# MAIN
# =========================================================

def _scenes_from_timecode_payload(payload: Dict[str, Any]) -> List[SceneTC]:
    return [
        SceneTC(
            global_scene_index=int(s["global_scene_index"]),
            block_id=int(s.get("block_id") or 0),
            scene_index_in_block=int(s.get("scene_index_in_block") or 1),
            sentence_indexes=list(s.get("sentence_indexes") or []),
            block_text=str(s.get("block_text") or ""),
            subject=str(s.get("subject") or ""),
            video_file=str(s["video_file"]),
            start=float(s["start"]),
            end=float(s["end"]),
            exists=(VIDEOS_DIR / str(s["video_file"])).exists(),
        )
        for s in payload["scenes"]
    ]


def process_locale(loc: Locale, scenario_text: str, scenes_master: List[SceneTC],
                   args, preview_seconds: Optional[float]) -> None:
    header = f"===== ЛОКАЛЬ [{loc.code}] : {loc.audio.name} ====="
    info("\n" + header)

    # Пропуск уже готового монтажа (например английский final_montage.mp4 уже есть).
    if (not args.force and not args.timecodes_only and preview_seconds is None
            and loc.out_video.exists()):
        info(f"[{loc.code}] {loc.out_video.name} уже существует — пропускаю "
             f"(перезапуск: --force).")
        return

    if args.render_only:
        payload = load_json(loc.timecodes_json, None)
        if not isinstance(payload, dict) or not payload.get("scenes"):
            warn(f"[{loc.code}] нет {loc.timecodes_json.name} — пропускаю (сначала без --render-only).")
            return
        scenes = _scenes_from_timecode_payload(payload)
        info(f"[{loc.code}] загружены тайм-коды: {len(scenes)} сцен.")
    else:
        scenes = build_timecodes(loc, scenario_text, scenes_master)
        if args.timecodes_only:
            return

    render_montage(scenes, loc, preview_seconds=preview_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Мультиязычный монтаж: тайм-коды по аудио каждого языка + сборка."
    )
    parser.add_argument("--timecodes-only", action="store_true",
                        help="Только этап 1: посчитать и сохранить тайм-коды.")
    parser.add_argument("--render-only", action="store_true",
                        help="Только этап 2: собрать монтаж по готовым video_timecodes[_xx].json.")
    parser.add_argument("--preview", nargs="?", type=float, const=PREVIEW_SECONDS_DEFAULT,
                        default=None, metavar="СЕК",
                        help="Смонтировать только первые N секунд (по умолчанию 60) в "
                             "final_montage[_xx]_preview.mp4 — быстрый тест без нагрузки.")
    parser.add_argument("--langs", type=str, default="", metavar="СПИСОК",
                        help="Какие языки делать, через запятую: en,ru,es,pt. "
                             "По умолчанию — все найденные по аудиофайлам.")
    parser.add_argument("--force", action="store_true",
                        help="Пересобрать даже если итоговый монтаж уже существует.")
    args = parser.parse_args()

    # Предпросмотр: приоритет у флага, иначе env PREVIEW_SECONDS.
    preview_seconds: Optional[float] = args.preview
    if preview_seconds is None and _PREVIEW_ENV:
        try:
            preview_seconds = float(_PREVIEW_ENV)
        except ValueError:
            warn(f"PREVIEW_SECONDS={_PREVIEW_ENV!r} — не число, игнорирую.")

    if not BASE_DIR.exists():
        fail(f"Нет базовой папки: {BASE_DIR}. Задай PROMPTS_BASE_DIR.")

    selected = [x.strip().lower() for x in args.langs.split(",") if x.strip()] or None
    locales = discover_locales(selected)
    if not locales:
        fail("Не найдено ни одной озвучки. Английская — аудио без префикса scenario; "
             "адаптации — scenario_ru/es/pt.<ext>.")
    info(f"Локали к обработке: {', '.join(l.code for l in locales)}")

    scenario_text = SCENARIO_FILE.read_text(encoding="utf-8").strip() if SCENARIO_FILE.exists() else ""
    if not scenario_text:
        warn("scenario.txt (английский) не найден — выравнивание по смыслу будет ограничено.")
    scenes_master = load_master_scenes()

    for loc in locales:
        try:
            process_locale(loc, scenario_text, scenes_master, args, preview_seconds)
        except SystemExit:
            raise
        except Exception as e:
            warn(f"[{loc.code}] локаль пропущена из-за ошибки: {e}")

    info("\n[ГОТОВО] Все локали обработаны.")


if __name__ == "__main__":
    main()
