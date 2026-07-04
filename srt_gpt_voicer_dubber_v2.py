#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SRT GPT + Voicer Dubber (self-launching edition)
================================================

Что делает скрипт:
1. Берёт английский .srt файл с таймкодами.
2. Объединяет слишком короткие субтитры в более естественные голосовые сегменты.
3. Переводит каждый сегмент через OpenAI API / ChatGPT API с учётом длительности таймкода.
4. Озвучивает каждый сегмент через Voicer API.
5. Автоматически проверяет длину аудио:
   - если аудио короче таймкода — добавляет тишину;
   - если аудио немного длиннее — ускоряет через ffmpeg/atempo;
   - если аудио сильно длиннее — просит GPT сократить фразу и озвучивает заново.
6. Собирает одну готовую аудиодорожку .wav под исходные таймкоды.
7. Сохраняет переведённый .srt и подробный log.json.
8. Опционально может сразу подставить новую аудиодорожку в видео.

========================================================================
ГЛАВНОЕ ОТЛИЧИЕ ЭТОЙ ВЕРСИИ: ничего не нужно настраивать вручную.
========================================================================
Просто дважды кликни по файлу (или запусти `python srt_gpt_voicer_dubber_v2.py`).
При первом запуске скрипт САМ:
  - создаст рядом с собой виртуальное окружение (папка `.venv_dubber`);
  - установит все нужные библиотеки (openai, requests, pysrt, pydub, tqdm);
  - скачает переносимый ffmpeg (через imageio-ffmpeg), если системный ffmpeg не найден;
  - перезапустит сам себя внутри этого окружения.
Больше не нужно каждый раз вручную активировать venv и что-то ставить.

Если запустить БЕЗ аргументов (например двойным кликом), скрипт спросит
путь к .srt и язык прямо в консоли. Если передать аргументы — работает как раньше:

    python srt_gpt_voicer_dubber_v2.py --srt input_en.srt --lang RU
    python srt_gpt_voicer_dubber_v2.py --srt input_en.srt --lang ALL
    python srt_gpt_voicer_dubber_v2.py --srt input_en.srt --lang RU --video video_en.mp4

ВАЖНО:
- Ключи можно вставить прямо ниже в настройки OPENAI_API_KEY и VOICER_API_KEY.
- Можно передавать через переменные окружения (OPENAI_API_KEY, VOICER_API_KEY)
  или в командной строке (--openai-key, --voicer-key).
"""

from __future__ import annotations

# ============================================================
# 0. АВТО-ЗАПУСК ВИРТУАЛЬНОГО ОКРУЖЕНИЯ (BOOTSTRAP)
# ------------------------------------------------------------
# Этот блок выполняется ПЕРВЫМ, до импорта сторонних библиотек.
# Он создаёт venv, ставит зависимости и перезапускает скрипт внутри venv.
# Пользователю больше ничего не нужно делать вручную.
# ============================================================

import os
import sys
import subprocess
from pathlib import Path

# Библиотеки, которые ставятся в автоматическое окружение.
BOOTSTRAP_REQUIREMENTS = [
    "openai",
    "requests",
    "pysrt",
    "pydub",
    "tqdm",
    "imageio-ffmpeg",  # переносимый ffmpeg на случай, если системного нет
]

# Имя папки с виртуальным окружением (создаётся рядом со скриптом).
BOOTSTRAP_VENV_DIR = ".venv_dubber"

# Переменная-маркер, чтобы не уйти в бесконечный перезапуск.
BOOTSTRAP_MARKER_ENV = "SRT_DUBBER_BOOTSTRAPPED"


def _bootstrap_venv_python(venv_dir: Path) -> Path:
    """Путь к python внутри venv для текущей ОС."""
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _bootstrap_pause_if_clicked(message: str) -> None:
    """
    Если окно, скорее всего, открыто двойным кликом — не даём ему мгновенно
    закрыться, чтобы пользователь успел прочитать сообщение об ошибке.
    """
    print(message)
    try:
        if sys.stdin and sys.stdin.isatty():
            input("\nНажми Enter, чтобы закрыть окно...")
    except Exception:
        pass


def _bootstrap_ensure_environment() -> None:
    """
    Создаёт (при необходимости) виртуальное окружение, ставит зависимости
    и перезапускает скрипт внутри него. Если мы уже внутри — просто выходит.
    """
    # Уже перезапущены внутри своего venv — ничего не делаем.
    if os.environ.get(BOOTSTRAP_MARKER_ENV) == "1":
        return

    script_dir = Path(__file__).resolve().parent
    venv_dir = script_dir / BOOTSTRAP_VENV_DIR
    venv_python = _bootstrap_venv_python(venv_dir)

    # 1. Создаём venv, если его ещё нет.
    if not venv_python.exists():
        print("[SETUP] Первый запуск: создаю виртуальное окружение...")
        print(f"[SETUP] Папка окружения: {venv_dir}")
        try:
            import venv as _venv_module

            _venv_module.EnvBuilder(with_pip=True).create(str(venv_dir))
        except Exception as exc:
            _bootstrap_pause_if_clicked(
                "[ОШИБКА] Не удалось создать виртуальное окружение.\n"
                f"Причина: {exc}\n"
                "На Linux, возможно, нужно поставить пакет python3-venv:\n"
                "    sudo apt install python3-venv"
            )
            sys.exit(1)

    # 2. Ставим зависимости, если это ещё не сделано (по «штампу»).
    stamp_path = venv_dir / ".deps_ok"
    wanted_stamp = "\n".join(sorted(BOOTSTRAP_REQUIREMENTS))
    deps_ready = stamp_path.exists() and stamp_path.read_text(encoding="utf-8") == wanted_stamp

    if not deps_ready:
        print("[SETUP] Устанавливаю необходимые библиотеки (это делается один раз)...")
        try:
            subprocess.run(
                [str(venv_python), "-m", "pip", "install", "--upgrade", "pip"],
                check=True,
            )
            subprocess.run(
                [str(venv_python), "-m", "pip", "install", *BOOTSTRAP_REQUIREMENTS],
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            _bootstrap_pause_if_clicked(
                "[ОШИБКА] Не удалось установить библиотеки.\n"
                f"Причина: {exc}\n"
                "Проверь подключение к интернету и попробуй запустить снова."
            )
            sys.exit(1)
        stamp_path.write_text(wanted_stamp, encoding="utf-8")
        print("[SETUP] Готово. Библиотеки установлены.")

    # 3. Перезапускаем скрипт внутри venv.
    env = dict(os.environ)
    env[BOOTSTRAP_MARKER_ENV] = "1"
    try:
        completed = subprocess.run(
            [str(venv_python), str(Path(__file__).resolve()), *sys.argv[1:]],
            env=env,
        )
    except KeyboardInterrupt:
        sys.exit(1)
    sys.exit(completed.returncode)


# Запускаем bootstrap сразу. Если мы не внутри venv — этот вызов не вернётся
# (скрипт перезапустится). Если внутри — просто продолжим импорты ниже.
_bootstrap_ensure_environment()


# ============================================================
# Теперь можно безопасно импортировать сторонние библиотеки:
# мы гарантированно внутри виртуального окружения со всем нужным.
# ============================================================

import argparse
import json
import re
import shutil
import tempfile
import time
import random
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from typing import Optional, Iterable

import warnings

import pysrt
import requests
from openai import OpenAI
# pydub при импорте предупреждает, что не нашёл ffmpeg в PATH. Мы задаём ffmpeg
# вручную (системный или переносимый) чуть ниже, поэтому это предупреждение не нужно.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)
    from pydub import AudioSegment
from tqdm import tqdm


# ============================================================
# 0b. НАСТРОЙКА FFMPEG (системный или переносимый)
# ============================================================

def resolve_ffmpeg_binary() -> str:
    """
    Возвращает путь к ffmpeg. Сначала ищет системный ffmpeg в PATH,
    иначе использует переносимый ffmpeg из пакета imageio-ffmpeg,
    который ставится автоматически. Так пользователю не нужно
    отдельно устанавливать ffmpeg вручную.
    """
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg

    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


# Единый путь к ffmpeg для всего скрипта.
FFMPEG_BINARY = resolve_ffmpeg_binary()

# Подсказываем pydub, каким ffmpeg пользоваться (важно для переносимого ffmpeg).
AudioSegment.converter = FFMPEG_BINARY
AudioSegment.ffmpeg = FFMPEG_BINARY
# ffprobe у imageio-ffmpeg нет, но для чтения/экспорта достаточно ffmpeg.
if shutil.which("ffprobe"):
    AudioSegment.ffprobe = shutil.which("ffprobe")


# ============================================================
# 1. КЛЮЧИ API
# ------------------------------------------------------------
# Ключи НЕ хранятся прямо в этом файле (иначе GitHub блокирует загрузку
# из-за защиты секретов, да и держать ключи в коде небезопасно).
#
# Ключи берутся автоматически, в таком порядке:
#   1) аргументы --openai-key / --voicer-key;
#   2) переменные окружения OPENAI_API_KEY / VOICER_API_KEY;
#   3) локальный файл рядом со скриптом: dubber_keys.txt (см. ниже).
#
# Файл dubber_keys.txt имеет простой формат (по одному ключу в строке):
#     OPENAI_API_KEY=sk-proj-...
#     VOICER_API_KEY=123456:abcdef...
# Он добавлен в .gitignore и НЕ попадает в репозиторий.
#
# Если ключей нигде нет, при первом запуске (двойным кликом) скрипт сам
# спросит их и сохранит в dubber_keys.txt — больше вводить не придётся.
# ============================================================

# Имя локального файла с ключами (рядом со скриптом).
KEYS_FILE_NAME = "dubber_keys.txt"

# Значения по умолчанию оставлены пустыми специально: реальные ключи
# подставляются из окружения / dubber_keys.txt (см. load_local_keys ниже).
OPENAI_API_KEY = ""
VOICER_API_KEY = ""


def _keys_file_path() -> Path:
    return Path(__file__).resolve().parent / KEYS_FILE_NAME


def load_local_keys() -> dict:
    """Читает ключи из локального файла dubber_keys.txt, если он есть."""
    keys: dict = {}
    path = _keys_file_path()
    if not path.exists():
        return keys
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, _, value = line.partition("=")
            keys[name.strip()] = value.strip().strip('"').strip("'")
    except Exception as exc:
        log(f"[WARN] Не удалось прочитать {path}: {exc}")
    return keys


def save_local_keys(openai_key: str, voicer_key: str) -> None:
    """Сохраняет ключи в локальный файл, чтобы вводить их только один раз."""
    path = _keys_file_path()
    try:
        path.write_text(
            "# Ключи для дубляжа. Этот файл НЕ загружается в GitHub.\n"
            f"OPENAI_API_KEY={openai_key}\n"
            f"VOICER_API_KEY={voicer_key}\n",
            encoding="utf-8",
        )
        log(f"[KEYS] Ключи сохранены в {path.name}. В следующий раз вводить не нужно.")
    except Exception as exc:
        log(f"[WARN] Не удалось сохранить ключи в {path}: {exc}")


# ============================================================
# 2. ОСНОВНЫЕ НАСТРОЙКИ
# ============================================================

# Модель OpenAI для перевода и адаптации текста под тайминг.
# Если модель недоступна в твоём аккаунте, замени на доступную, например:
# "gpt-4.1-mini", "gpt-4o-mini", "gpt-5-mini".
TRANSLATION_MODEL = "gpt-4.1-mini"

# Языки, которые поддерживает этот скрипт.
# RU = русский, ES = испанский для Латинской Америки, PT = бразильский португальский.
LANGUAGE_SETTINGS = {
    "RU": {
        "name": "Russian",
        "style": "живой, понятный русский язык для христианского YouTube-ролика; без канцелярита; уважительный библейский тон",
        "tts_instruction": "Clear calm Russian Christian narration. Warm, serious, cinematic tone.",
    },
    "ES": {
        "name": "Latin American Spanish",
        "style": "natural Latin American Spanish for a Christian YouTube video; reverent, clear, emotionally engaging",
        "tts_instruction": "Clear Latin American Spanish Christian narration. Warm, serious, cinematic tone.",
    },
    "PT": {
        "name": "Brazilian Portuguese",
        "style": "natural Brazilian Portuguese for a Christian YouTube video; reverent, clear, emotionally engaging",
        "tts_instruction": "Clear Brazilian Portuguese Christian narration. Warm, serious, cinematic tone.",
    },
}

# Папка для результатов по умолчанию.
OUTPUT_DIR = Path("dub_output")

# Если True, старые сегменты будут использоваться повторно.
# Это удобно, если скрипт прервался и ты запускаешь его заново.
USE_CACHE = True

# Если True, старые переводы будут использоваться повторно.
USE_TRANSLATION_CACHE = True

# Улучшенная логика перевода:
# 1) сначала создаётся краткий контекст всего ролика;
# 2) затем перевод выполняется не по одной строке, а пачками.
# Это сильно уменьшает дословность и ускоряет обработку.
BUILD_CONTEXT_BRIEF = True
TRANSLATE_IN_BATCHES = True
TRANSLATION_BATCH_SIZE = 14

# Сколько параллельных задач озвучки запускать через Voicer.
# 1 = максимально безопасно, но медленно. 2-3 обычно быстрее.
TTS_MAX_WORKERS = 2

# Создавать итоговый .mp4, если передан --video.
MAKE_VIDEO_WITH_NEW_AUDIO = True

# Насколько можно ускорять готовый сегмент без перегенерации текста.
# 1.12 = максимум +12% скорости. Лучше не ставить слишком высоко, чтобы голос не звучал странно.
MAX_SAFE_POST_SPEEDUP = 1.12

# Если после сокращения аудио всё ещё длиннее таймкода, допускается более сильное ускорение.
MAX_EMERGENCY_POST_SPEEDUP = 1.28

# Небольшой запас, чтобы фраза не упиралась прямо в конец сегмента.
# 80 мс обычно достаточно.
SEGMENT_END_MARGIN_MS = 80

# Финальный запас тишины в конце всей дорожки.
TAIL_SILENCE_MS = 1000


# ============================================================
# 3. НАСТРОЙКИ ОБЪЕДИНЕНИЯ SRT-СЕГМЕНТОВ
# ============================================================

# Если субтитры слишком короткие, озвучка будет рваной.
# Поэтому скрипт может объединять соседние строки SRT в более естественные блоки.
MERGE_SHORT_SUBTITLES = True

# Минимальная желательная длительность голосового блока.
MIN_GROUP_DURATION_MS = 6500

# Максимальная длительность голосового блока.
MAX_GROUP_DURATION_MS = 15000

# Максимальная пауза между соседними SRT-строками, при которой их можно объединять.
MAX_GAP_TO_MERGE_MS = 1200

# Максимум символов в одном голосовом блоке.
MAX_GROUP_CHARS = 950


# ============================================================
# 4. НАСТРОЙКИ VOICER API
# ============================================================

# Новый домен ставим первым, потому что старый voiceapi.csv666.ru
# может отвечать на /balance, но зависать на создании задач или скачивании результата.
PRIMARY_BASE_URL = "https://voiceapiru.csv666.ru"
BACKUP_BASE_URL = "https://voiceapi.csv666.ru"

POLL_SECONDS = 8
MAX_WAIT_SECONDS = 0  # 0 = ждать бесконечно

# requests допускает tuple: (connect_timeout, read_timeout).
# Для скачивания аудио даём больше времени на чтение.
REQUEST_TIMEOUT = (30, 300)

# Повторы для временных сетевых сбоев Voicer/API.
# Без этого один ConnectionResetError может оборвать весь длинный дубляж.
RETRY_ATTEMPTS = 6
RETRY_BASE_SECONDS = 3
RETRY_MAX_SECONDS = 30
RETRY_STATUS_CODES = {429, 500, 502, 503, 504}

# Настройки голоса из твоего Voicer-скрипта.
# При необходимости замени VOICE_ID на другой голос.
VOICE_ID = "wnKyx1zkUEUnfURKiuaP"
VOICE_VERSION = "v3"
MODEL_ID = "eleven_v3"
VOICE_SPEED = 0.90

# Если у тебя есть готовый шаблон голоса из Voicer-бота, можно вставить UUID сюда.
# Если None — будет использоваться INLINE_TEMPLATE.
TEMPLATE_UUID: Optional[str] = None

INLINE_TEMPLATE: Optional[dict] = {
    "voice_id": VOICE_ID,
    "version": VOICE_VERSION,
    "model_id": MODEL_ID,
    "voice_settings": {
        "stability": 0.85,
        "similarity_boost": 0.75,
        "use_speaker_boost": True,
        "style": 0.28,
        "speed": VOICE_SPEED,
    },
}

CHUNK_SIZE: Optional[int] = None
PAUSE_SETTINGS: Optional[dict] = None
STRESS_SETTINGS: Optional[dict] = None

# Теги пауз для Eleven v3 / Voicer.
# Для сегментного дубляжа лучше не ставить слишком много пауз, иначе аудио не влезет в тайминг.
ADD_V3_PAUSE_TAGS = False
PARAGRAPH_PAUSE_TAG = "[short pause]"
SECTION_PAUSE_TAG = "[pause]"


# ============================================================
# 5. ГЛОССАРИЙ ДЛЯ БИБЛЕЙСКИХ РОЛИКОВ
# ============================================================

BIBLE_GLOSSARY = {
    "Jesus Christ": {"RU": "Иисус Христос", "ES": "Jesucristo", "PT": "Jesus Cristo"},
    "Jesus": {"RU": "Иисус", "ES": "Jesús", "PT": "Jesus"},
    "Christ": {"RU": "Христос", "ES": "Cristo", "PT": "Cristo"},
    "God": {"RU": "Бог", "ES": "Dios", "PT": "Deus"},
    "Lord": {"RU": "Господь", "ES": "Señor", "PT": "Senhor"},
    "Holy Spirit": {"RU": "Святой Дух", "ES": "Espíritu Santo", "PT": "Espírito Santo"},
    "Kingdom of God": {"RU": "Царство Божье", "ES": "Reino de Dios", "PT": "Reino de Deus"},
    "Scripture": {"RU": "Писание", "ES": "Escritura", "PT": "Escritura"},
    "Bible": {"RU": "Библия", "ES": "Biblia", "PT": "Bíblia"},
    "Gospel": {"RU": "Евангелие", "ES": "Evangelio", "PT": "Evangelho"},
    "faith": {"RU": "вера", "ES": "fe", "PT": "fé"},
    "grace": {"RU": "благодать", "ES": "gracia", "PT": "graça"},
    "sin": {"RU": "грех", "ES": "pecado", "PT": "pecado"},
    "repentance": {"RU": "покаяние", "ES": "arrepentimiento", "PT": "arrependimento"},
    "salvation": {"RU": "спасение", "ES": "salvación", "PT": "salvação"},
    "disciple": {"RU": "ученик", "ES": "discípulo", "PT": "discípulo"},
    "disciples": {"RU": "ученики", "ES": "discípulos", "PT": "discípulos"},
    "apostle": {"RU": "апостол", "ES": "apóstol", "PT": "apóstolo"},
    "prophet": {"RU": "пророк", "ES": "profeta", "PT": "profeta"},
}


# ============================================================
# 6. DATA CLASSES
# ============================================================

@dataclass
class SubtitleUnit:
    index: int
    start_ms: int
    end_ms: int
    text: str


@dataclass
class VoiceGroup:
    group_id: int
    source_indices: list[int]
    start_ms: int
    end_ms: int
    english_text: str

    @property
    def duration_ms(self) -> int:
        return max(0, self.end_ms - self.start_ms)

    @property
    def duration_sec(self) -> float:
        return self.duration_ms / 1000.0


# ============================================================
# 7. УТИЛИТЫ
# ============================================================

def log(message: str) -> None:
    print(message, flush=True)


def require_ffmpeg() -> None:
    # После bootstrap ffmpeg почти всегда доступен (системный или переносимый).
    # Проверяем именно выбранный бинарь, а не только PATH.
    if FFMPEG_BINARY == "ffmpeg" and shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "Не найден ffmpeg. Обычно он ставится автоматически через imageio-ffmpeg. "
            "Если этого не произошло, установи ffmpeg вручную и добавь его в PATH. "
            "macOS: brew install ffmpeg"
        )
    log(f"[FFMPEG] {FFMPEG_BINARY}")


def normalize_api_key(value: Optional[str]) -> str:
    if not value:
        return ""
    value = value.strip()
    if value in {"PASTE_OPENAI_API_KEY_HERE", "PASTE_VOICER_API_KEY_HERE", "your-api-key-here"}:
        return ""
    return value


def srt_time_to_ms(t) -> int:
    return (
        t.hours * 3600 * 1000
        + t.minutes * 60 * 1000
        + t.seconds * 1000
        + t.milliseconds
    )


def ms_to_srt_time(ms: int) -> str:
    ms = max(0, int(ms))
    hours = ms // 3_600_000
    ms %= 3_600_000
    minutes = ms // 60_000
    ms %= 60_000
    seconds = ms // 1000
    milliseconds = ms % 1000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def clean_subtitle_text(text: str) -> str:
    text = text.replace("﻿", "")
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def add_v3_pause_tags(text: str) -> str:
    if not ADD_V3_PAUSE_TAGS:
        return text.strip()

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n\s*[-—_]{3,}\s*\n", f" {SECTION_PAUSE_TAG} ", text)
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    paragraphs = [re.sub(r"\s*\n\s*", " ", p) for p in paragraphs]
    return f" {PARAGRAPH_PAUSE_TAG} ".join(paragraphs).strip()


def safe_filename_part(text: str, limit: int = 60) -> str:
    text = re.sub(r"[^\w\-.]+", "_", text, flags=re.UNICODE)
    text = text.strip("_")
    return text[:limit] if text else "file"


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


# ============================================================
# 8. ЧТЕНИЕ И ГРУППИРОВКА SRT
# ============================================================

def read_srt_units(srt_path: Path) -> list[SubtitleUnit]:
    subs = pysrt.open(str(srt_path), encoding="utf-8")
    units: list[SubtitleUnit] = []

    for sub in subs:
        text = clean_subtitle_text(sub.text)
        if not text:
            continue
        units.append(
            SubtitleUnit(
                index=int(sub.index),
                start_ms=srt_time_to_ms(sub.start),
                end_ms=srt_time_to_ms(sub.end),
                text=text,
            )
        )

    if not units:
        raise RuntimeError(f"В SRT нет текста: {srt_path}")

    return units


def should_merge(current: VoiceGroup, next_unit: SubtitleUnit) -> bool:
    if not MERGE_SHORT_SUBTITLES:
        return False

    gap = next_unit.start_ms - current.end_ms
    projected_duration = next_unit.end_ms - current.start_ms
    projected_chars = len(current.english_text) + 1 + len(next_unit.text)

    if gap < 0:
        # Иногда SRT немного пересекаются. В таком случае объединять можно.
        gap = 0

    if gap > MAX_GAP_TO_MERGE_MS:
        return False

    if projected_duration > MAX_GROUP_DURATION_MS:
        return False

    if projected_chars > MAX_GROUP_CHARS:
        return False

    # Объединяем короткий блок до минимальной желательной длины.
    if current.duration_ms < MIN_GROUP_DURATION_MS:
        return True

    # Также объединяем, если текущий текст заканчивается явно незавершённо.
    if not re.search(r"[.!?…]['\"»”)]?$", current.english_text.strip()):
        return True

    return False


def build_voice_groups(units: list[SubtitleUnit]) -> list[VoiceGroup]:
    groups: list[VoiceGroup] = []
    current: Optional[VoiceGroup] = None

    for unit in units:
        if current is None:
            current = VoiceGroup(
                group_id=1,
                source_indices=[unit.index],
                start_ms=unit.start_ms,
                end_ms=unit.end_ms,
                english_text=unit.text,
            )
            continue

        if should_merge(current, unit):
            current.source_indices.append(unit.index)
            current.end_ms = max(current.end_ms, unit.end_ms)
            current.english_text = f"{current.english_text} {unit.text}".strip()
        else:
            groups.append(current)
            current = VoiceGroup(
                group_id=len(groups) + 1,
                source_indices=[unit.index],
                start_ms=unit.start_ms,
                end_ms=unit.end_ms,
                english_text=unit.text,
            )

    if current is not None:
        groups.append(current)

    # Перенумеруем на всякий случай.
    for i, group in enumerate(groups, start=1):
        group.group_id = i

    return groups


# ============================================================
# 9. OPENAI: ПЕРЕВОД И СОКРАЩЕНИЕ ПОД ТАЙМИНГ
# ============================================================

def build_glossary_text(lang: str) -> str:
    lines = []
    for source, mapping in BIBLE_GLOSSARY.items():
        if lang in mapping:
            lines.append(f"- {source} => {mapping[lang]}")
    return "\n".join(lines)


def make_openai_client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key)


def call_openai_text(client: OpenAI, prompt: str) -> str:
    response = client.responses.create(
        model=TRANSLATION_MODEL,
        input=prompt,
    )
    text = getattr(response, "output_text", "") or ""
    text = text.strip()
    if not text:
        raise RuntimeError("OpenAI вернул пустой текст.")
    return text


def extract_json_array(text: str) -> list:
    """Достаёт JSON-массив из ответа модели, даже если он случайно пришёл в ```json``` блоке."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)

    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"Не найден JSON-массив в ответе OpenAI:\n{text[:1000]}")

    return json.loads(text[start:end + 1])


def build_context_brief(client: OpenAI, full_english_text: str, lang: str) -> str:
    """
    Делает краткую карту ролика: тема, стиль, эмоциональная дуга, ключевые термины.
    Потом эта карта подставляется в каждый batch-перевод.
    """
    lang_info = LANGUAGE_SETTINGS[lang]
    source = full_english_text.strip()
    if len(source) > 18000:
        source = source[:9000] + "\n\n[...middle omitted...]\n\n" + source[-9000:]

    prompt = f"""
You are preparing a professional dubbing translation for a Christian/Bible YouTube video.

Target language: {lang_info['name']}.

Read the English subtitle text and create a compact translation brief.
The brief will be used by another translator to avoid literal translation and keep context.

Include:
1. Main topic and message of the video.
2. Emotional tone and narration style.
3. Key biblical terms/names that must be consistent.
4. Warnings about what NOT to translate literally.
5. How the narrator should sound in the target language.

Return the brief in Russian if target is Russian, otherwise in English.
Keep it concise but useful.

English subtitle text:
{source}
""".strip()

    return call_openai_text(client, prompt).strip()


def translate_batch_for_timing(
    client: OpenAI,
    groups: list[VoiceGroup],
    lang: str,
    context_brief: str,
    previous_translation: str = "",
    next_english_after_batch: str = "",
) -> dict[int, str]:
    """
    Переводит сразу пачку голосовых сегментов.
    Это лучше для качества: модель видит соседние мысли и меньше переводит дословно.
    """
    lang_info = LANGUAGE_SETTINGS[lang]
    glossary = build_glossary_text(lang)

    items = []
    for group in groups:
        items.append({
            "id": group.group_id,
            "duration_seconds": round(group.duration_sec, 2),
            "english": group.english_text,
        })

    # Отдельные инструкции для русского, потому что именно там чаще всего заметна дословность.
    russian_extra = """
For Russian specifically:
- Do NOT copy English sentence structure.
- Translate like a native Russian Christian narrator, not like subtitles.
- Prefer: clear, warm, spoken Russian.
- Avoid awkward calques such as "это есть", "в этот момент времени", "он будет иметь".
- If the English phrase is idiomatic, convey the meaning naturally, not literally.
""" if lang == "RU" else ""

    prompt = f"""
You are a senior dubbing translator and voiceover script adapter.

Your job is NOT literal subtitle translation.
Your job is to adapt English narration into natural {lang_info['name']} voiceover for a Christian/Bible YouTube video.

GLOBAL CONTEXT BRIEF:
{context_brief if context_brief else '[none]'}

TARGET STYLE:
{lang_info['style']}

BIBLICAL GLOSSARY. Use consistently when relevant:
{glossary}

TIMING RULES:
- Each translated segment must fit its duration.
- But do NOT make it too short: aim to fill about 85-95% of the available spoken time.
- Rebuild the sentence naturally for the target language.
- Use punctuation intentionally so TTS understands pauses and intonation.
- Keep the emotional flow between segments.
- Avoid filler words, but do not remove important biblical meaning.

INTONATION RULES:
- Use commas, dashes, and sentence breaks to guide the voice.
- If a phrase is a warning, make it firm.
- If a phrase is about grace, hope, salvation, or Christ, make it warm and reverent.
- If a phrase introduces a new thought, make it clean and clear.

{russian_extra}

Previous translated segment before this batch:
{previous_translation if previous_translation else '[none]'}

Next English segment after this batch:
{next_english_after_batch if next_english_after_batch else '[none]'}

Return ONLY valid JSON array.
Do not wrap it in markdown.
Each item must be:
{{"id": 1, "text": "translated voiceover text"}}

Segments to translate:
{json.dumps(items, ensure_ascii=False, indent=2)}
""".strip()

    raw = call_openai_text(client, prompt)
    data = extract_json_array(raw)

    result: dict[int, str] = {}
    for item in data:
        group_id = int(item["id"])
        text = clean_subtitle_text(str(item["text"]))
        if text:
            result[group_id] = text

    missing = [g.group_id for g in groups if g.group_id not in result]
    if missing:
        raise RuntimeError(f"OpenAI batch не вернул переводы для id: {missing}")

    return result


# Старый одиночный перевод оставлен как fallback, если batch-режим не сработает.
def translate_for_timing(
    client: OpenAI,
    english_text: str,
    lang: str,
    duration_sec: float,
    previous_translation: str = "",
    next_english_text: str = "",
    context_brief: str = "",
) -> str:
    lang_info = LANGUAGE_SETTINGS[lang]
    glossary = build_glossary_text(lang)

    prompt = f"""
You are a professional Christian video dubbing translator.

Task:
Translate and adapt the English subtitle segment into {lang_info['name']}.

Global context brief:
{context_brief if context_brief else '[none]'}

Target language style:
{lang_info['style']}

Timing requirement:
The translated voiceover must fit approximately {duration_sec:.2f} seconds.
Aim to fill about 85-95% of the available spoken time: not too long, but not too short.
Do NOT translate word-for-word. Rebuild the sentence naturally for the target language.
Preserve the exact meaning, biblical reverence, emotional force, and clarity.
Make it sound natural when spoken aloud.
Use punctuation to guide TTS intonation.
Avoid unnecessary filler words.
Avoid long complicated sentences.

Biblical glossary. Use these terms consistently when relevant:
{glossary}

Context from previous translated segment:
{previous_translation if previous_translation else '[none]'}

Context from next English segment:
{next_english_text if next_english_text else '[none]'}

Return ONLY the final translated voiceover line.
No explanations.
No quotation marks.

English segment:
{english_text}
""".strip()

    result = call_openai_text(client, prompt)
    result = clean_subtitle_text(result)
    return result


def shorten_for_timing(client: OpenAI, text: str, lang: str, duration_sec: float, reduction_hint: str = "20%") -> str:
    lang_info = LANGUAGE_SETTINGS[lang]
    prompt = f"""
You are editing a translated voiceover line for timing.

Language: {lang_info['name']}
Style: {lang_info['style']}

The current line is too long for the video segment.
Shorten it by about {reduction_hint} so it can fit approximately {duration_sec:.2f} seconds.
Keep the same core meaning, biblical tone, and emotional clarity.
Make it natural for spoken narration.
Do not add explanations.
Return ONLY the shortened line.

Text:
{text}
""".strip()

    result = call_openai_text(client, prompt)
    return clean_subtitle_text(result)


# ============================================================
# 10. VOICER API
# ============================================================

def voicer_headers(api_key: str) -> dict:
    return {
        "X-API-Key": api_key,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def voicer_download_headers(api_key: str) -> dict:
    return {"X-API-Key": api_key}


def pretty_json(data) -> str:
    try:
        return json.dumps(data, ensure_ascii=False, indent=2)
    except Exception:
        return str(data)


def retry_delay(attempt: int) -> float:
    """Небольшая экспоненциальная пауза между повторами."""
    return min(RETRY_MAX_SECONDS, RETRY_BASE_SECONDS * (2 ** max(0, attempt - 1))) + random.uniform(0, 1.5)


def request_json(method: str, base_url: str, path: str, api_key: str, *, json_body: Optional[dict] = None) -> dict:
    url = base_url.rstrip("/") + path
    last_error: Optional[BaseException] = None

    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            response = requests.request(
                method=method,
                url=url,
                headers=voicer_headers(api_key),
                json=json_body,
                timeout=REQUEST_TIMEOUT,
            )
        except requests.RequestException as exc:
            last_error = exc
            if attempt < RETRY_ATTEMPTS:
                delay = retry_delay(attempt)
                log(f"[WARN] Ошибка соединения с {url}: {exc}. Повтор {attempt}/{RETRY_ATTEMPTS} через {delay:.1f} сек.")
                time.sleep(delay)
                continue
            raise RuntimeError(f"Ошибка соединения с {url}: {exc}") from exc

        try:
            data = response.json()
        except Exception:
            data = {"raw_text": response.text}

        if response.status_code in RETRY_STATUS_CODES and attempt < RETRY_ATTEMPTS:
            delay = retry_delay(attempt)
            log(f"[WARN] HTTP {response.status_code} от {url}. Повтор {attempt}/{RETRY_ATTEMPTS} через {delay:.1f} сек.")
            time.sleep(delay)
            continue

        if response.status_code >= 400:
            raise RuntimeError(f"HTTP {response.status_code} от {url}\nОтвет:\n{pretty_json(data)}")

        return data

    raise RuntimeError(f"Не удалось получить JSON от {url}. Последняя ошибка: {last_error}")


def choose_working_base_url(api_key: str) -> str:
    for base_url in [PRIMARY_BASE_URL, BACKUP_BASE_URL]:
        try:
            data = request_json("GET", base_url, "/balance", api_key)
            balance = data.get("balance_text") or data.get("balance")
            log(f"[VOICER] Использую {base_url} | balance: {balance}")
            return base_url
        except Exception as exc:
            log(f"[WARN] Не удалось проверить {base_url}: {exc}")

    raise RuntimeError("Не удалось подключиться ни к основному, ни к backup Voicer API.")


def build_voicer_task_payload(text: str) -> dict:
    text = add_v3_pause_tags(text)

    payload = {
        "text": text,
        "voice_id": VOICE_ID,
        "version": VOICE_VERSION,
        "model_id": MODEL_ID,
    }

    if TEMPLATE_UUID and INLINE_TEMPLATE:
        raise RuntimeError("Нельзя одновременно использовать TEMPLATE_UUID и INLINE_TEMPLATE.")

    if TEMPLATE_UUID:
        payload["template_uuid"] = TEMPLATE_UUID

    if INLINE_TEMPLATE:
        payload["template"] = INLINE_TEMPLATE

    if CHUNK_SIZE is not None:
        payload["chunk_size"] = CHUNK_SIZE

    if PAUSE_SETTINGS is not None:
        payload["pause_settings"] = PAUSE_SETTINGS

    if STRESS_SETTINGS is not None:
        payload["stress_settings"] = STRESS_SETTINGS

    return payload


def create_voicer_tts_task(base_url: str, api_key: str, text: str) -> int:
    payload = build_voicer_task_payload(text)
    data = request_json("POST", base_url, "/tasks", api_key, json_body=payload)

    task_id = data.get("task_id")
    if task_id is None:
        raise RuntimeError(f"Voicer API не вернул task_id. Ответ:\n{pretty_json(data)}")

    return int(task_id)


def wait_for_voicer_task(base_url: str, api_key: str, task_id: int) -> str:
    started = time.time()

    while True:
        data = request_json("GET", base_url, f"/tasks/{task_id}/status", api_key)
        status = data.get("status")
        label = data.get("status_label", "")

        log(f"[VOICER STATUS] task_id={task_id} | {status} {label}")

        if status in {"ending", "ending_processed"}:
            return str(status)

        if status in {"error", "error_handled"}:
            raise RuntimeError(f"Voicer задача #{task_id} завершилась ошибкой:\n{pretty_json(data)}")

        if MAX_WAIT_SECONDS > 0 and time.time() - started > MAX_WAIT_SECONDS:
            raise TimeoutError(f"Превышено время ожидания Voicer задачи #{task_id}: {MAX_WAIT_SECONDS} секунд")

        time.sleep(POLL_SECONDS)


def filename_from_content_disposition(value: str) -> Optional[str]:
    if not value:
        return None

    m = re.search(r"filename\*=UTF-8''([^;]+)", value, flags=re.IGNORECASE)
    if m:
        from urllib.parse import unquote
        return unquote(m.group(1).strip().strip('"'))

    m = re.search(r'filename="?([^";]+)"?', value, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()

    return None


def guess_ext_from_response(response: requests.Response) -> str:
    cd_name = filename_from_content_disposition(response.headers.get("Content-Disposition", ""))
    if cd_name:
        suffix = Path(cd_name).suffix
        if suffix:
            return suffix

    content_type = response.headers.get("Content-Type", "").lower()
    if "zip" in content_type:
        return ".zip"
    if "wav" in content_type:
        return ".wav"
    if "mpeg" in content_type or "mp3" in content_type or "audio" in content_type:
        return ".mp3"

    return ".mp3"


def download_voicer_result(base_url: str, api_key: str, task_id: int, out_base: Path) -> Path:
    url = base_url.rstrip("/") + f"/tasks/{task_id}/result"
    last_error: Optional[BaseException] = None

    for attempt in range(1, RETRY_ATTEMPTS + 1):
        response = None
        try:
            response = requests.get(
                url,
                headers=voicer_download_headers(api_key),
                timeout=REQUEST_TIMEOUT,
                stream=True,
            )
        except requests.RequestException as exc:
            last_error = exc
            if attempt < RETRY_ATTEMPTS:
                delay = retry_delay(attempt)
                log(f"[WARN] Ошибка скачивания результата {url}: {exc}. Повтор {attempt}/{RETRY_ATTEMPTS} через {delay:.1f} сек.")
                time.sleep(delay)
                continue
            raise RuntimeError(f"Ошибка скачивания результата {url}: {exc}") from exc

        try:
            if response.status_code == 202 and attempt < RETRY_ATTEMPTS:
                delay = retry_delay(attempt)
                log(f"[WARN] Файл задачи #{task_id} ещё не готов. Повтор {attempt}/{RETRY_ATTEMPTS} через {delay:.1f} сек.")
                response.close()
                time.sleep(delay)
                continue

            if response.status_code in RETRY_STATUS_CODES and attempt < RETRY_ATTEMPTS:
                delay = retry_delay(attempt)
                log(f"[WARN] HTTP {response.status_code} при скачивании результата {url}. Повтор {attempt}/{RETRY_ATTEMPTS} через {delay:.1f} сек.")
                response.close()
                time.sleep(delay)
                continue

            if response.status_code == 202:
                raise RuntimeError(f"Файл ещё не готов, хотя статус уже проверен. Ответ: {response.text}")

            if response.status_code >= 400:
                raise RuntimeError(f"HTTP {response.status_code} при скачивании результата.\nОтвет:\n{response.text}")

            ext = guess_ext_from_response(response)
            out_path = out_base.with_suffix(ext)
            out_path.parent.mkdir(parents=True, exist_ok=True)

            tmp_path = out_path.with_suffix(out_path.suffix + ".part")
            bytes_written = 0
            with tmp_path.open("wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    f.write(chunk)
                    bytes_written += len(chunk)

            if bytes_written <= 0:
                raise RuntimeError(f"Скачанный файл пустой: {url}")

            tmp_path.replace(out_path)

            if not out_path.exists() or out_path.stat().st_size == 0:
                raise RuntimeError(f"Файл не сохранился или пустой: {out_path}")

            return out_path

        except requests.RequestException as exc:
            last_error = exc
            if attempt < RETRY_ATTEMPTS:
                delay = retry_delay(attempt)
                log(f"[WARN] Сбой во время потокового скачивания {url}: {exc}. Повтор {attempt}/{RETRY_ATTEMPTS} через {delay:.1f} сек.")
                time.sleep(delay)
                continue
            raise RuntimeError(f"Ошибка потокового скачивания результата {url}: {exc}") from exc

        finally:
            try:
                response.close()
            except Exception:
                pass

    raise RuntimeError(f"Не удалось скачать результат {url}. Последняя ошибка: {last_error}")


def extract_or_normalize_audio(input_path: Path, output_wav_path: Path) -> Path:
    """
    Voicer иногда может вернуть mp3, wav или zip.
    На выходе всегда получаем wav.
    """
    output_wav_path.parent.mkdir(parents=True, exist_ok=True)

    if input_path.suffix.lower() == ".zip":
        extract_dir = input_path.with_suffix("")
        extract_dir.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(input_path, "r") as zf:
            zf.extractall(extract_dir)

        audio_files = []
        for ext in ["*.mp3", "*.wav", "*.m4a", "*.flac", "*.ogg"]:
            audio_files.extend(sorted(extract_dir.rglob(ext)))

        if not audio_files:
            raise RuntimeError(f"ZIP не содержит аудиофайлов: {input_path}")

        combined = AudioSegment.empty()
        for audio_file in audio_files:
            combined += AudioSegment.from_file(audio_file)
        combined.export(output_wav_path, format="wav")
        return output_wav_path

    audio = AudioSegment.from_file(input_path)
    audio.export(output_wav_path, format="wav")
    return output_wav_path


def voicer_tts_to_wav(
    base_url: str,
    api_key: str,
    text: str,
    out_base: Path,
    final_wav_path: Path,
) -> Path:
    task_id = create_voicer_tts_task(base_url, api_key, text)
    log(f"[VOICER TASK] создана задача #{task_id}")
    wait_for_voicer_task(base_url, api_key, task_id)
    raw_result = download_voicer_result(base_url, api_key, task_id, out_base)
    return extract_or_normalize_audio(raw_result, final_wav_path)


# ============================================================
# 11. АУДИО: УСКОРЕНИЕ, ПОДГОНКА, СБОРКА
# ============================================================

def ffmpeg_atempo_chain(speed: float) -> str:
    """
    ffmpeg atempo традиционно работает в диапазоне 0.5..2.0 за один фильтр.
    Для наших скоростей достаточно одного фильтра, но оставляем универсально.
    """
    speed = float(speed)
    if speed <= 0:
        raise ValueError("speed must be > 0")

    filters = []
    while speed > 2.0:
        filters.append("atempo=2.0")
        speed /= 2.0
    while speed < 0.5:
        filters.append("atempo=0.5")
        speed /= 0.5
    filters.append(f"atempo={speed:.6f}")
    return ",".join(filters)


def change_audio_speed(input_path: Path, output_path: Path, speed: float) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    filter_arg = ffmpeg_atempo_chain(speed)
    cmd = [
        FFMPEG_BINARY,
        "-y",
        "-i",
        str(input_path),
        "-filter:a",
        filter_arg,
        "-vn",
        str(output_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return output_path


def load_audio(path: Path) -> AudioSegment:
    return AudioSegment.from_file(path)


def pad_or_trim(audio: AudioSegment, target_ms: int) -> AudioSegment:
    if len(audio) < target_ms:
        return audio + AudioSegment.silent(duration=target_ms - len(audio))
    if len(audio) > target_ms:
        # Лёгкий fade-out, чтобы аварийная обрезка не звучала щелчком.
        return audio[:target_ms].fade_out(min(80, max(0, target_ms // 8)))
    return audio


def fit_segment_audio(
    audio_wav_path: Path,
    target_ms: int,
    work_dir: Path,
    group_id: int,
    emergency: bool = False,
) -> tuple[Path, dict]:
    """
    Подгоняет аудио под длительность сегмента.
    Возвращает путь к итоговому wav и статистику.
    """
    target_ms = max(100, target_ms - SEGMENT_END_MARGIN_MS)
    audio = load_audio(audio_wav_path)
    original_ms = len(audio)

    info = {
        "original_audio_ms": original_ms,
        "target_fit_ms": target_ms,
        "speedup_used": 1.0,
        "trimmed": False,
        "padded_ms": 0,
    }

    if original_ms <= target_ms:
        final_audio = pad_or_trim(audio, target_ms)
        info["padded_ms"] = max(0, target_ms - original_ms)
        fitted_path = work_dir / f"segment_{group_id:04d}_fitted.wav"
        final_audio.export(fitted_path, format="wav")
        return fitted_path, info

    required_speed = original_ms / target_ms
    max_speed = MAX_EMERGENCY_POST_SPEEDUP if emergency else MAX_SAFE_POST_SPEEDUP

    if required_speed <= max_speed:
        sped_path = work_dir / f"segment_{group_id:04d}_speed_{required_speed:.3f}.wav"
        change_audio_speed(audio_wav_path, sped_path, required_speed)
        sped_audio = load_audio(sped_path)
        final_audio = pad_or_trim(sped_audio, target_ms)
        fitted_path = work_dir / f"segment_{group_id:04d}_fitted.wav"
        final_audio.export(fitted_path, format="wav")
        info["speedup_used"] = round(required_speed, 4)
        info["trimmed"] = len(sped_audio) > target_ms
        info["padded_ms"] = max(0, target_ms - len(sped_audio))
        return fitted_path, info

    # Слишком длинно, без перегенерации нормально не влезает.
    return audio_wav_path, info


def mux_video_with_audio(video_path: Path, audio_path: Path, output_video_path: Path) -> None:
    output_video_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        FFMPEG_BINARY,
        "-y",
        "-i",
        str(video_path),
        "-i",
        str(audio_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-shortest",
        str(output_video_path),
    ]
    subprocess.run(cmd, check=True)


# ============================================================
# 12. SRT OUTPUT
# ============================================================

def save_translated_srt(groups: list[VoiceGroup], translations: dict[int, str], out_path: Path) -> None:
    lines = []
    for i, group in enumerate(groups, start=1):
        text = translations.get(group.group_id, "").strip()
        if not text:
            text = group.english_text
        lines.append(str(i))
        lines.append(f"{ms_to_srt_time(group.start_ms)} --> {ms_to_srt_time(group.end_ms)}")
        lines.append(text)
        lines.append("")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


# ============================================================
# 13. ОСНОВНАЯ ОБРАБОТКА ЯЗЫКА
# ============================================================

def process_tts_group_worker(
    *,
    group: VoiceGroup,
    lang: str,
    translated: str,
    lang_outdir: Path,
    voicer_base_url: str,
    voicer_api_key: str,
    openai_api_key: str,
) -> dict:
    """Озвучивает один голосовой сегмент. Можно запускать параллельно."""
    raw_segments_dir = lang_outdir / "raw_segments"
    wav_segments_dir = lang_outdir / "wav_segments"
    fitted_segments_dir = lang_outdir / "fitted_segments"

    target_ms = group.duration_ms
    final_segment_path = fitted_segments_dir / f"segment_{group.group_id:04d}_final.wav"
    raw_wav_path = wav_segments_dir / f"segment_{group.group_id:04d}.wav"
    raw_out_base = raw_segments_dir / f"segment_{group.group_id:04d}"

    tts_text_used = translated
    shortened_used = False
    fit_info = {}

    if USE_CACHE and final_segment_path.exists() and final_segment_path.stat().st_size > 0:
        segment_audio = load_audio(final_segment_path)
        return {
            "group_id": group.group_id,
            "audio_path": str(final_segment_path),
            "final_text": tts_text_used,
            "log_item": {
                "group_id": group.group_id,
                "cached": True,
                "source_indices": group.source_indices,
                "start_ms": group.start_ms,
                "end_ms": group.end_ms,
                "target_ms": target_ms,
                "english_text": group.english_text,
                "final_text": tts_text_used,
                "final_audio_ms": len(segment_audio),
            },
        }

    voicer_tts_to_wav(voicer_base_url, voicer_api_key, tts_text_used, raw_out_base, raw_wav_path)
    fitted_path, fit_info = fit_segment_audio(raw_wav_path, target_ms, fitted_segments_dir, group.group_id, emergency=False)
    fitted_audio = load_audio(fitted_path)

    if len(fitted_audio) > max(100, target_ms - SEGMENT_END_MARGIN_MS):
        log(f"[SHORTEN] segment {group.group_id}: аудио слишком длинное ({len(fitted_audio)} ms / {target_ms} ms)")
        local_client = make_openai_client(openai_api_key)
        shortened = shorten_for_timing(local_client, translated, lang, group.duration_sec, reduction_hint="25-35%")
        tts_text_used = shortened
        shortened_used = True

        raw_wav_path = wav_segments_dir / f"segment_{group.group_id:04d}_short.wav"
        raw_out_base = raw_segments_dir / f"segment_{group.group_id:04d}_short"
        voicer_tts_to_wav(voicer_base_url, voicer_api_key, tts_text_used, raw_out_base, raw_wav_path)
        fitted_path, fit_info = fit_segment_audio(raw_wav_path, target_ms, fitted_segments_dir, group.group_id, emergency=True)
        fitted_audio = load_audio(fitted_path)

    fitted_audio = pad_or_trim(fitted_audio, target_ms)
    fitted_audio.export(final_segment_path, format="wav")

    return {
        "group_id": group.group_id,
        "audio_path": str(final_segment_path),
        "final_text": tts_text_used,
        "log_item": {
            "group_id": group.group_id,
            "cached": False,
            "source_indices": group.source_indices,
            "start_ms": group.start_ms,
            "end_ms": group.end_ms,
            "target_ms": target_ms,
            "duration_sec": round(group.duration_sec, 3),
            "english_text": group.english_text,
            "translated_initial": translated,
            "final_text": tts_text_used,
            "shortened_used": shortened_used,
            "final_audio_ms": len(fitted_audio),
            "fit_info": fit_info,
        },
    }


def process_language(
    *,
    srt_path: Path,
    lang: str,
    outdir: Path,
    openai_client: OpenAI,
    openai_api_key: str,
    voicer_base_url: str,
    voicer_api_key: str,
    video_path: Optional[Path] = None,
) -> None:
    if lang not in LANGUAGE_SETTINGS:
        raise RuntimeError(f"Неподдерживаемый язык: {lang}. Доступно: {', '.join(LANGUAGE_SETTINGS)}")

    lang_outdir = outdir / lang
    raw_segments_dir = lang_outdir / "raw_segments"
    wav_segments_dir = lang_outdir / "wav_segments"
    fitted_segments_dir = lang_outdir / "fitted_segments"
    cache_dir = lang_outdir / "cache"

    raw_segments_dir.mkdir(parents=True, exist_ok=True)
    wav_segments_dir.mkdir(parents=True, exist_ok=True)
    fitted_segments_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    units = read_srt_units(srt_path)
    groups = build_voice_groups(units)

    total_duration_ms = max(group.end_ms for group in groups) + TAIL_SILENCE_MS
    final_audio = AudioSegment.silent(duration=total_duration_ms)

    translations: dict[int, str] = {}
    log_items = []

    translations_cache_path = cache_dir / "translations.json"
    if USE_TRANSLATION_CACHE and translations_cache_path.exists():
        try:
            translations = {int(k): v for k, v in read_json(translations_cache_path).items()}
            log(f"[CACHE] Загружены переводы: {translations_cache_path}")
        except Exception as exc:
            log(f"[WARN] Не удалось прочитать translation cache: {exc}")
            translations = {}

    log("")
    log("=" * 70)
    log(f"[LANG] {lang} | {LANGUAGE_SETTINGS[lang]['name']}")
    log(f"[SRT] {srt_path}")
    log(f"[GROUPS] {len(groups)} голосовых сегментов")
    log(f"[OUTPUT] {lang_outdir}")
    log("=" * 70)

    # -------------------------
    # Общий контекст ролика
    # -------------------------
    context_cache_path = cache_dir / "context_brief.txt"
    context_brief = ""
    if BUILD_CONTEXT_BRIEF:
        if USE_TRANSLATION_CACHE and context_cache_path.exists():
            try:
                context_brief = context_cache_path.read_text(encoding="utf-8").strip()
                log(f"[CACHE] Загружен контекст ролика: {context_cache_path}")
            except Exception as exc:
                log(f"[WARN] Не удалось прочитать context cache: {exc}")

        if not context_brief:
            full_english_text = "\n".join(group.english_text for group in groups)
            log("[OPENAI] Создаю краткий контекст всего ролика для более естественного перевода...")
            context_brief = build_context_brief(openai_client, full_english_text, lang)
            context_cache_path.write_text(context_brief, encoding="utf-8")

    # -------------------------
    # Перевод заранее, batch-режим
    # -------------------------
    missing_groups = [group for group in groups if not (group.group_id in translations and USE_TRANSLATION_CACHE)]

    if missing_groups:
        log(f"[OPENAI] Нужно перевести сегментов: {len(missing_groups)}")

    if TRANSLATE_IN_BATCHES:
        previous_translation_for_batch = ""
        for batch_start in tqdm(range(0, len(groups), TRANSLATION_BATCH_SIZE), desc=f"Translate {lang}"):
            batch = groups[batch_start:batch_start + TRANSLATION_BATCH_SIZE]
            batch_missing = [g for g in batch if not (g.group_id in translations and USE_TRANSLATION_CACHE)]
            if not batch_missing:
                # Для связности сохраняем последнюю уже готовую фразу.
                previous_translation_for_batch = translations.get(batch[-1].group_id, previous_translation_for_batch)
                continue

            next_after = ""
            next_index = batch_start + TRANSLATION_BATCH_SIZE
            if next_index < len(groups):
                next_after = groups[next_index].english_text

            try:
                translated_batch = translate_batch_for_timing(
                    openai_client,
                    batch,
                    lang,
                    context_brief,
                    previous_translation=previous_translation_for_batch,
                    next_english_after_batch=next_after,
                )
                translations.update(translated_batch)
            except Exception as exc:
                log(f"[WARN] Batch-перевод не сработал, перехожу на одиночный перевод: {exc}")
                for i, group in enumerate(batch):
                    if group.group_id in translations and USE_TRANSLATION_CACHE:
                        continue
                    next_text = batch[i + 1].english_text if i + 1 < len(batch) else next_after
                    translations[group.group_id] = translate_for_timing(
                        openai_client,
                        group.english_text,
                        lang,
                        group.duration_sec,
                        previous_translation=previous_translation_for_batch,
                        next_english_text=next_text,
                        context_brief=context_brief,
                    )
                    previous_translation_for_batch = translations[group.group_id]

            previous_translation_for_batch = translations.get(batch[-1].group_id, previous_translation_for_batch)
            write_json(translations_cache_path, {str(k): v for k, v in translations.items()})
    else:
        previous_translation = ""
        for idx, group in enumerate(tqdm(groups, desc=f"Translate {lang}"), start=1):
            if group.group_id in translations and USE_TRANSLATION_CACHE:
                previous_translation = translations[group.group_id]
                continue
            next_text = groups[idx].english_text if idx < len(groups) else ""
            translations[group.group_id] = translate_for_timing(
                openai_client,
                group.english_text,
                lang,
                group.duration_sec,
                previous_translation=previous_translation,
                next_english_text=next_text,
                context_brief=context_brief,
            )
            previous_translation = translations[group.group_id]
            write_json(translations_cache_path, {str(k): v for k, v in translations.items()})

    # -------------------------
    # Озвучка: теперь можно параллелить, потому что переводы уже готовы
    # -------------------------
    worker_args = []
    for group in groups:
        translated = translations.get(group.group_id, "").strip()
        if not translated:
            raise RuntimeError(f"Нет перевода для сегмента {group.group_id}")
        worker_args.append({
            "group": group,
            "lang": lang,
            "translated": translated,
            "lang_outdir": lang_outdir,
            "voicer_base_url": voicer_base_url,
            "voicer_api_key": voicer_api_key,
            "openai_api_key": openai_api_key,
        })

    results = []
    tts_errors = []
    if TTS_MAX_WORKERS <= 1:
        for kwargs in tqdm(worker_args, desc=f"TTS {lang}"):
            group_id = kwargs["group"].group_id
            try:
                results.append(process_tts_group_worker(**kwargs))
            except Exception as exc:
                log(f"[ERROR] TTS segment {group_id}: {exc}")
                tts_errors.append((group_id, str(exc)))
    else:
        with ThreadPoolExecutor(max_workers=TTS_MAX_WORKERS) as executor:
            future_to_group = {executor.submit(process_tts_group_worker, **kwargs): kwargs["group"].group_id for kwargs in worker_args}
            for future in tqdm(as_completed(future_to_group), total=len(future_to_group), desc=f"TTS {lang}"):
                group_id = future_to_group[future]
                try:
                    results.append(future.result())
                except Exception as exc:
                    log(f"[ERROR] TTS segment {group_id}: {exc}")
                    tts_errors.append((group_id, str(exc)))

    if tts_errors:
        failed = ", ".join(str(group_id) for group_id, _ in tts_errors[:20])
        more = "..." if len(tts_errors) > 20 else ""
        write_json(lang_outdir / f"tts_errors_{lang}.json", [{"group_id": g, "error": e} for g, e in tts_errors])
        raise RuntimeError(f"Не удалось озвучить {len(tts_errors)} сегментов: {failed}{more}. Подробности в tts_errors_{lang}.json")

    results.sort(key=lambda item: int(item["group_id"]))

    for item in results:
        group = groups[int(item["group_id"]) - 1]
        final_text = str(item["final_text"])
        translations[group.group_id] = final_text
        segment_audio = load_audio(Path(str(item["audio_path"])))
        final_audio = final_audio.overlay(pad_or_trim(segment_audio, group.duration_ms), position=group.start_ms)
        log_items.append(item["log_item"])

    write_json(translations_cache_path, {str(k): v for k, v in translations.items()})

    # -------------------------
    # Финальные файлы
    # -------------------------
    final_audio_path = lang_outdir / f"voice_{lang}.wav"
    translated_srt_path = lang_outdir / f"subtitles_{lang}.srt"
    log_path = lang_outdir / f"log_{lang}.json"
    groups_path = lang_outdir / f"groups_{lang}.json"

    final_audio.export(final_audio_path, format="wav")
    save_translated_srt(groups, translations, translated_srt_path)
    write_json(log_path, log_items)
    write_json(groups_path, [asdict(group) for group in groups])

    log("")
    log(f"[DONE] {lang}")
    log(f"[AUDIO] {final_audio_path}")
    log(f"[SRT]   {translated_srt_path}")
    log(f"[LOG]   {log_path}")

    if video_path and MAKE_VIDEO_WITH_NEW_AUDIO:
        output_video_path = lang_outdir / f"video_{lang}.mp4"
        mux_video_with_audio(video_path, final_audio_path, output_video_path)
        log(f"[VIDEO] {output_video_path}")


# ============================================================
# 14. ИНТЕРАКТИВНЫЙ РЕЖИМ (для двойного клика без аргументов)
# ============================================================

def _prompt(text: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    try:
        answer = input(f"{text}{suffix}: ").strip()
    except EOFError:
        answer = ""
    return answer or default


def interactive_collect_args() -> argparse.Namespace:
    """
    Спрашивает у пользователя параметры прямо в консоли.
    Используется, когда скрипт запущен двойным кликом без аргументов.
    """
    print("=" * 70)
    print("SRT GPT + Voicer Dubber — интерактивный запуск")
    print("Нажимай Enter, чтобы оставить значение по умолчанию.")
    print("=" * 70)

    srt = ""
    while not srt:
        srt = _prompt("Путь к .srt файлу (можно перетащить файл в окно)")
        srt = srt.strip().strip('"').strip("'")
        if srt and not Path(srt).expanduser().exists():
            print(f"  [!] Файл не найден: {srt}. Попробуй ещё раз.")
            srt = ""

    lang = ""
    while lang not in {"RU", "ES", "PT", "ALL"}:
        lang = _prompt("Язык (RU / ES / PT / ALL)", "RU").upper()
        if lang not in {"RU", "ES", "PT", "ALL"}:
            print("  [!] Допустимо: RU, ES, PT или ALL.")

    video = _prompt("Путь к видео для озвучки (необязательно, Enter — пропустить)")
    video = video.strip().strip('"').strip("'") or None
    if video and not Path(video).expanduser().exists():
        print(f"  [!] Видео не найдено, продолжаю без него: {video}")
        video = None

    outdir = _prompt("Папка для результатов", str(OUTPUT_DIR))

    # Собираем namespace с теми же полями, что и parse_args().
    return argparse.Namespace(
        srt=srt,
        lang=lang,
        outdir=outdir,
        video=video,
        openai_key=None,
        voicer_key=None,
        voice_id=None,
        voice_speed=None,
        no_cache=False,
        no_merge=False,
        tts_workers=None,
        no_batch_translate=False,
        interactive=True,
    )


# ============================================================
# 15. CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Translate English SRT with OpenAI and dub it with Voicer API.")
    parser.add_argument("--srt", required=True, help="Path to input English .srt file")
    parser.add_argument("--lang", required=True, choices=["RU", "ES", "PT", "ALL"], help="Target language")
    parser.add_argument("--outdir", default=str(OUTPUT_DIR), help="Output directory")
    parser.add_argument("--video", default=None, help="Optional source video path. If provided, script creates localized MP4.")
    parser.add_argument("--openai-key", default=None, help="OpenAI API key. Overrides value inside script/env.")
    parser.add_argument("--voicer-key", default=None, help="Voicer API key. Overrides value inside script/env.")
    parser.add_argument("--voice-id", default=None, help="Override VOICE_ID for Voicer")
    parser.add_argument("--voice-speed", type=float, default=None, help="Override VOICE_SPEED for Voicer template")
    parser.add_argument("--no-cache", action="store_true", help="Do not reuse cached audio/translation files")
    parser.add_argument("--no-merge", action="store_true", help="Do not merge short SRT subtitles")
    parser.add_argument("--tts-workers", type=int, default=None, help="How many Voicer TTS tasks to run in parallel. Default is script setting.")
    parser.add_argument("--no-batch-translate", action="store_true", help="Disable batch translation and translate segment by segment.")
    args = parser.parse_args()
    args.interactive = False
    return args


def get_args() -> argparse.Namespace:
    """
    Если аргументы командной строки переданы — используем их (старое поведение).
    Если нет (например, двойной клик) — спрашиваем интерактивно.
    """
    if len(sys.argv) > 1:
        return parse_args()
    return interactive_collect_args()


def main() -> None:
    global VOICE_ID, VOICE_SPEED, INLINE_TEMPLATE, USE_CACHE, USE_TRANSLATION_CACHE, MERGE_SHORT_SUBTITLES, TTS_MAX_WORKERS, TRANSLATE_IN_BATCHES

    args = get_args()
    require_ffmpeg()

    if args.no_cache:
        USE_CACHE = False
        USE_TRANSLATION_CACHE = False

    if args.no_merge:
        MERGE_SHORT_SUBTITLES = False

    if args.tts_workers is not None:
        TTS_MAX_WORKERS = max(1, int(args.tts_workers))

    if args.no_batch_translate:
        TRANSLATE_IN_BATCHES = False

    if args.voice_id:
        VOICE_ID = args.voice_id.strip()

    if args.voice_speed is not None:
        VOICE_SPEED = float(args.voice_speed)

    # Обновляем template, если voice/speed поменяли через аргументы.
    INLINE_TEMPLATE = {
        "voice_id": VOICE_ID,
        "version": VOICE_VERSION,
        "model_id": MODEL_ID,
        "voice_settings": {
            "stability": 0.85,
            "similarity_boost": 0.75,
            "use_speaker_boost": True,
            "style": 0.28,
            "speed": VOICE_SPEED,
        },
    }

    local_keys = load_local_keys()
    openai_key = (
        normalize_api_key(args.openai_key)
        or normalize_api_key(os.getenv("OPENAI_API_KEY"))
        or normalize_api_key(local_keys.get("OPENAI_API_KEY"))
        or normalize_api_key(OPENAI_API_KEY)
    )
    voicer_key = (
        normalize_api_key(args.voicer_key)
        or normalize_api_key(os.getenv("VOICER_API_KEY"))
        or normalize_api_key(local_keys.get("VOICER_API_KEY"))
        or normalize_api_key(VOICER_API_KEY)
    )

    # Если ключей нет и запуск интерактивный (двойной клик) — спросим один раз
    # и сохраним в dubber_keys.txt, чтобы больше не вводить.
    interactive = getattr(args, "interactive", False)
    if interactive and (not openai_key or not voicer_key):
        print("")
        print("Похоже, ключи API ещё не заданы. Введи их один раз — я их запомню.")
        if not openai_key:
            openai_key = normalize_api_key(_prompt("OpenAI API key (sk-...)"))
        if not voicer_key:
            voicer_key = normalize_api_key(_prompt("Voicer API key"))
        if openai_key and voicer_key:
            save_local_keys(openai_key, voicer_key)

    if not openai_key:
        print(f"[ERROR] Не найден OpenAI API key. Укажи его через --openai-key, "
              f"переменную окружения OPENAI_API_KEY или в файле {KEYS_FILE_NAME}.")
        _hold_console_if_interactive(interactive)
        sys.exit(1)

    if not voicer_key:
        print(f"[ERROR] Не найден Voicer API key. Укажи его через --voicer-key, "
              f"переменную окружения VOICER_API_KEY или в файле {KEYS_FILE_NAME}.")
        _hold_console_if_interactive(interactive)
        sys.exit(1)

    srt_path = Path(args.srt).expanduser().resolve()
    if not srt_path.exists():
        raise FileNotFoundError(f"Не найден SRT файл: {srt_path}")

    video_path = Path(args.video).expanduser().resolve() if args.video else None
    if video_path and not video_path.exists():
        raise FileNotFoundError(f"Не найден видеофайл: {video_path}")

    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    log(f"[OPENAI MODEL] {TRANSLATION_MODEL}")
    log(f"[VOICE] {VOICE_ID} | {MODEL_ID} | speed={VOICE_SPEED}")
    log(f"[MERGE] {'ON' if MERGE_SHORT_SUBTITLES else 'OFF'} | min={MIN_GROUP_DURATION_MS}ms max={MAX_GROUP_DURATION_MS}ms")
    log(f"[BATCH TRANSLATE] {'ON' if TRANSLATE_IN_BATCHES else 'OFF'} | batch={TRANSLATION_BATCH_SIZE}")
    log(f"[TTS WORKERS] {TTS_MAX_WORKERS}")
    log(f"[CACHE] {'ON' if USE_CACHE else 'OFF'}")

    openai_client = make_openai_client(openai_key)
    voicer_base_url = choose_working_base_url(voicer_key)

    languages = ["RU", "ES", "PT"] if args.lang == "ALL" else [args.lang]

    errors = []
    for lang in languages:
        try:
            process_language(
                srt_path=srt_path,
                lang=lang,
                outdir=outdir,
                openai_client=openai_client,
                openai_api_key=openai_key,
                voicer_base_url=voicer_base_url,
                voicer_api_key=voicer_key,
                video_path=video_path,
            )
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            log(f"[ERROR] {lang}: {exc}")
            errors.append((lang, str(exc)))

    log("")
    log("=" * 70)
    if errors:
        log("[FINISH] Завершено с ошибками:")
        for lang, error in errors:
            log(f"- {lang}: {error}")
        _hold_console_if_interactive(getattr(args, "interactive", False))
        sys.exit(1)

    log("[FINISH] Готово.")
    _hold_console_if_interactive(getattr(args, "interactive", False))


def _hold_console_if_interactive(interactive: bool) -> None:
    """
    Не даём окну закрыться сразу после завершения, если запуск был
    двойным кликом (интерактивный режим), чтобы можно было прочитать итог.
    """
    if not interactive:
        return
    try:
        input("\nГотово. Нажми Enter, чтобы закрыть окно...")
    except Exception:
        pass


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nОстановлено пользователем.")
        sys.exit(1)
