#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
smart_sync_montage.py — умная синхронизация "аудио + промпты + ролики" одним запуском.

ЗАЧЕМ
=====
Старый конвейер выравнивал аудио на scenario.txt через нарезку на предложения,
которая ломается на русском тексте (кавычки-«ёлочки», многоточия, куски без
точек). Из-за этого блоки получали неверные границы, слоты по 40-100 секунд и
монтаж замирал. Плюс правильный результат требовал запускать два скрипта с
нужными флагами и переменными окружения — одна ошибка, и фризы возвращались.

Этот скрипт самодостаточен и НЕ зависит от scenario.txt вообще:

  1) АУДИО:   транскрибирует озвучку Whisper'ом с таймкодами слов
              (кэш audio_transcript.json переиспользуется — повторные запуски
              бесплатны и работают без интернета).
  2) ПРОМПТЫ: берёт block_text КАЖДОГО блока прямо из generated_prompts.json —
              тот самый текст, к которому привязаны сгенерированные сцены.
  3) РОЛИКИ:  измеряет реальную длительность каждого 000N.mp4 (ffprobe).

Дальше выравнивает тексты блоков на транскрипт (пословно, difflib) — каждый
блок получает РЕАЛЬНЫЙ интервал звучания, а его сцены делят этот интервал
пропорционально длине своих роликов. Встроенный анти-фриз: ни один слот не
может быть длиннее ролика более чем в MAX_STRETCH раз (лишнее время уходит
ближайшим соседям с запасом), поэтому застывших кадров не бывает в принципе —
длинные слоты рендерятся лёгким замедлением.

В конце пишет video_timecodes.json/.txt (клип -> текстовый промежуток -> время)
и сразу собирает final_montage.mp4.

ЗАПУСК
======
    python3 smart_sync_montage.py              # всё сразу: тайм-коды + монтаж
    python3 smart_sync_montage.py --preview 90 # быстрый тест: первые 90 секунд
    python3 smart_sync_montage.py --timecodes-only   # только посчитать тайм-коды
    python3 smart_sync_montage.py --audio путь/к/озвучке.wav   # явное аудио

Можно запускать хоть кнопкой Run из редактора — флаги не обязательны,
переменные окружения не нужны, повторный запуск всегда безопасен.

OpenAI ключ (нужен только если нет кэша транскрипта): переменная окружения
OPENAI_API_KEY или файл openai_key.txt в базовой папке.
"""

from __future__ import annotations

import argparse
import difflib
import json
import math
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# =========================================================
# CONFIG
# =========================================================

BASE_DIR = Path(os.getenv("PROMPTS_BASE_DIR", "/Users/aleksandrtomilov/Desktop/ПРОМПТЫ"))
VIDEOS_DIR = BASE_DIR / "ВИДЕО"
PROMPTS_JSON = BASE_DIR / "generated_prompts.json"
TRANSCRIPT_JSON = BASE_DIR / "audio_transcript.json"      # кэш Whisper (совместим со старым)
TIMECODES_JSON = BASE_DIR / "video_timecodes.json"
TIMECODES_TXT = BASE_DIR / "video_timecodes.txt"
FINAL_VIDEO = BASE_DIR / "final_montage.mp4"
PREVIEW_VIDEO = BASE_DIR / "final_montage_preview.mp4"

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "whisper-1")
WHISPER_MAX_BYTES = 24 * 1024 * 1024
WHISPER_CHUNK_SECONDS = int(os.getenv("WHISPER_CHUNK_SECONDS", "1200"))

CLIP_SECONDS_FALLBACK = float(os.getenv("CLIP_SECONDS", "8"))
MAX_STRETCH = float(os.getenv("MAX_STRETCH", "1.6"))   # потолок slot/clip (анти-фриз)
MAX_SLOW_FACTOR = 3.0                                  # предел замедления в рендере
MIN_SLOT_SECONDS = 0.7                                 # короче — клип сливается с соседом
TARGET_FPS = int(os.getenv("TARGET_FPS", "30"))
TARGET_RESOLUTION = os.getenv("TARGET_RESOLUTION", "").strip()  # напр. "1920x1080"

# Порог качества выравнивания: доля слов транскрипта, нашедших место в текстах блоков.
ALIGN_MIN_COVERAGE = float(os.getenv("ALIGN_MIN_COVERAGE", "0.45"))

AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg"}
EPS = 1e-4


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


def which(program: str) -> Optional[str]:
    return shutil.which(program)


def format_tc(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    m = int(seconds // 60)
    s = seconds - m * 60
    return f"{m:02d}:{s:06.3f}"


def progress_bar(done: int, total: int, width: int = 32) -> str:
    total = max(1, total)
    frac = max(0.0, min(1.0, done / total))
    filled = int(round(frac * width))
    return f"[{'█' * filled}{'░' * (width - filled)}] {frac * 100:5.1f}%  ({done}/{total})"


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def save_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def backup(path: Path) -> None:
    if path.exists():
        bak = path.with_suffix(path.suffix + ".bak")
        if not bak.exists():
            shutil.copy2(path, bak)


def resolve_openai_key() -> str:
    key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if key:
        return key
    key_file = BASE_DIR / "openai_key.txt"
    if key_file.exists():
        return key_file.read_text(encoding="utf-8").strip()
    return ""


# =========================================================
# ffprobe / ffmpeg
# =========================================================

def ffprobe_duration(path: Path) -> Optional[float]:
    if not which("ffprobe") or not path.exists():
        return None
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, check=True,
        )
        return float(result.stdout.strip())
    except Exception:
        return None


def ffmpeg_duration(path: Path) -> Optional[float]:
    if not which("ffmpeg") or not path.exists():
        return None
    try:
        result = subprocess.run(["ffmpeg", "-hide_banner", "-i", str(path)],
                                capture_output=True, text=True)
        m = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.?\d*)", result.stderr)
        if m:
            h, mi, s = m.groups()
            return int(h) * 3600 + int(mi) * 60 + float(s)
    except Exception:
        pass
    return None


def wave_duration(path: Path) -> Optional[float]:
    if path.suffix.lower() != ".wav":
        return None
    try:
        import wave
        with wave.open(str(path), "rb") as wf:
            return wf.getnframes() / float(wf.getframerate())
    except Exception:
        return None


def probe_duration(path: Path) -> Optional[float]:
    return ffprobe_duration(path) or wave_duration(path) or ffmpeg_duration(path)


def ffprobe_resolution(path: Path) -> Optional[Tuple[int, int]]:
    if which("ffprobe"):
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0",
                 "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x", str(path)],
                capture_output=True, text=True, check=True,
            )
            w, h = result.stdout.strip().split("x")
            return int(w), int(h)
        except Exception:
            pass
    return None


# =========================================================
# АУДИО
# =========================================================

def find_audio_file(explicit: Optional[Path]) -> Path:
    if explicit:
        if not explicit.exists():
            fail(f"Указанное аудио не найдено: {explicit}")
        return explicit
    preferred = ["voice", "audio"]
    files = sorted(
        [p for p in BASE_DIR.iterdir() if p.is_file() and p.suffix.lower() in AUDIO_EXTS],
        key=lambda x: x.name.lower(),
    ) if BASE_DIR.exists() else []
    for stem in preferred:
        for p in files:
            if p.stem.lower() == stem:
                return p
    for p in files:  # любой, кроме scenario_xx (адаптации других языков)
        if not p.stem.lower().startswith("scenario"):
            return p
    if files:
        return files[0]
    fail(f"В {BASE_DIR} нет ни одного аудиофайла ({', '.join(sorted(AUDIO_EXTS))}).")
    raise SystemExit


# =========================================================
# ТРАНСКРИБАЦИЯ (WHISPER, с кэшем)
# =========================================================

@dataclass
class Word:
    text: str
    start: float
    end: float


@dataclass
class Transcript:
    words: List[Word] = field(default_factory=list)
    segments: List[Word] = field(default_factory=list)
    language: str = ""


def _transcript_from_dict(data: Dict[str, Any]) -> Transcript:
    words, segments = [], []
    for w in (data.get("words") or []):
        try:
            words.append(Word(str(w.get("word", w.get("text", ""))),
                              float(w["start"]), float(w["end"])))
        except Exception:
            continue
    for s in (data.get("segments") or []):
        try:
            segments.append(Word(str(s.get("text", "")), float(s["start"]), float(s["end"])))
        except Exception:
            continue
    return Transcript(words=words, segments=segments, language=str(data.get("language", "")))


def _run_whisper_file(client: Any, path: Path) -> Dict[str, Any]:
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


def _compress_audio(src: Path, out: Path) -> Optional[Path]:
    if not which("ffmpeg"):
        return None
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
             "-i", str(src), "-ac", "1", "-ar", "16000",
             "-c:a", "libmp3lame", "-b:a", "32k", str(out)],
            check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        warn(f"Сжатие аудио не удалось: {e.stderr[:200]}")
        return None
    return out if out.exists() and out.stat().st_size > 0 else None


def _split_audio(src: Path, chunk_seconds: int, tmp_dir: Path) -> List[Tuple[Path, float]]:
    if not which("ffmpeg"):
        return []
    pattern = str(tmp_dir / "chunk_%03d.mp3")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
             "-i", str(src), "-f", "segment", "-segment_time", str(chunk_seconds),
             "-c", "copy", pattern],
            check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        warn(f"Нарезка аудио не удалась: {e.stderr[:200]}")
        return []
    result, offset = [], 0.0
    for c in sorted(tmp_dir.glob("chunk_*.mp3")):
        result.append((c, offset))
        offset += probe_duration(c) or chunk_seconds
    return result


def transcribe_audio(audio_path: Path) -> Optional[Transcript]:
    cached = load_json(TRANSCRIPT_JSON, None)
    if isinstance(cached, dict) and cached.get("audio_name") == audio_path.name \
            and not cached.get("translated"):
        info(f"Использую кэш транскрипта: {TRANSCRIPT_JSON.name}")
        return _transcript_from_dict(cached)

    api_key = resolve_openai_key()
    if not api_key:
        warn("Нет OpenAI ключа и нет кэша транскрипта — тайминги будут пропорциональными "
             "(менее точными). Ключ: переменная OPENAI_API_KEY или файл openai_key.txt.")
        return None
    try:
        from openai import OpenAI
    except Exception as e:
        warn(f"Не установлен пакет openai ({e}) — транскрибация пропущена.")
        return None

    client = OpenAI(api_key=api_key)
    tmp_dir = BASE_DIR / "_whisper_tmp_smart"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    src = audio_path
    if src.stat().st_size > WHISPER_MAX_BYTES:
        info(f"Аудио {src.stat().st_size / 1048576:.1f} МБ > лимита Whisper — сжимаю...")
        compressed = _compress_audio(src, tmp_dir / "whisper_input.mp3")
        if compressed:
            src = compressed

    if src.stat().st_size <= WHISPER_MAX_BYTES:
        parts: List[Tuple[Path, float]] = [(src, 0.0)]
    else:
        parts = _split_audio(src, WHISPER_CHUNK_SECONDS, tmp_dir)
        if not parts:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            return None

    info(f"Whisper: {audio_path.name}, частей: {len(parts)}...")
    all_words: List[Word] = []
    all_segments: List[Word] = []
    language, ok = "", False
    for idx, (fpath, offset) in enumerate(parts, start=1):
        try:
            data = _run_whisper_file(client, fpath)
        except Exception as e:
            warn(f"Часть {idx}/{len(parts)} не обработана: {str(e)[:160]}")
            continue
        part = _transcript_from_dict(data)
        language = language or part.language
        all_words += [Word(w.text, w.start + offset, w.end + offset) for w in part.words]
        all_segments += [Word(s.text, s.start + offset, s.end + offset) for s in part.segments]
        ok = True
        if len(parts) > 1:
            info(f"  часть {idx}/{len(parts)} готова (+{len(part.words)} слов)")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    if not ok:
        return None

    tr = Transcript(words=all_words, segments=all_segments, language=language)
    save_json(TRANSCRIPT_JSON, {
        "audio_name": audio_path.name, "translated": False, "language": tr.language,
        "words": [w.__dict__ for w in tr.words],
        "segments": [s.__dict__ for s in tr.segments],
    })
    info(f"Транскрипт: слов={len(tr.words)}, сегментов={len(tr.segments)}, язык={tr.language or '?'}")
    return tr


# =========================================================
# ПРОМПТЫ -> БЛОКИ И СЦЕНЫ
# =========================================================

@dataclass
class Scene:
    global_scene_index: int
    block_id: int
    scene_index_in_block: int
    block_text: str
    subject: str
    video_file: str
    exists: bool = False
    clip_dur: float = CLIP_SECONDS_FALLBACK
    start: float = 0.0
    end: float = 0.0


@dataclass
class Block:
    block_id: int
    text: str
    scenes: List[Scene] = field(default_factory=list)
    start: float = 0.0
    end: float = 0.0
    aligned: bool = False


def load_blocks() -> List[Block]:
    data = load_json(PROMPTS_JSON, None)
    if not isinstance(data, list) or not data:
        fail(f"Не найден или пуст {PROMPTS_JSON} — сначала мастер-скрипт промптов.")
    blocks: Dict[int, Block] = {}
    order: List[int] = []
    for i, rec in enumerate(data, start=1):
        if not isinstance(rec, dict):
            continue
        gsi = int(rec.get("global_scene_index") or i)
        bid = int(rec.get("block_id") or 0)
        sc = Scene(
            global_scene_index=gsi,
            block_id=bid,
            scene_index_in_block=int(rec.get("scene_index_in_block") or 1),
            block_text=str(rec.get("block_text") or ""),
            subject=str(rec.get("subject") or ""),
            video_file=f"{gsi:04d}.mp4",
        )
        sc.exists = (VIDEOS_DIR / sc.video_file).exists()
        if sc.exists:
            sc.clip_dur = probe_duration(VIDEOS_DIR / sc.video_file) or CLIP_SECONDS_FALLBACK
        if bid not in blocks:
            blocks[bid] = Block(block_id=bid, text=sc.block_text)
            order.append(bid)
        blocks[bid].scenes.append(sc)
        if not blocks[bid].text and sc.block_text:
            blocks[bid].text = sc.block_text
    result = [blocks[b] for b in order]
    for b in result:
        b.scenes.sort(key=lambda s: (s.scene_index_in_block, s.global_scene_index))
    total = sum(len(b.scenes) for b in result)
    missing = sum(1 for b in result for s in b.scenes if not s.exists)
    info(f"Промпты: блоков={len(result)}, сцен={total}"
         + (f", НЕТ роликов у {missing} сцен (их время уйдёт соседям)" if missing else ""))
    return result


# =========================================================
# ВЫРАВНИВАНИЕ ТЕКСТОВ БЛОКОВ НА ТРАНСКРИПТ
# =========================================================

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _interpolate(times: List[Optional[float]]) -> None:
    n = len(times)
    first = next((i for i in range(n) if times[i] is not None), None)
    if first is None:
        return
    last = next((i for i in range(n - 1, -1, -1) if times[i] is not None), None)
    for i in range(first):
        times[i] = times[first]
    for i in range(last + 1, n):
        times[i] = times[last]
    i = first
    while i <= last:
        if times[i] is not None:
            i += 1
            continue
        prev = i - 1
        nxt = i
        while times[nxt] is None:
            nxt += 1
        t0, t1, gap = times[prev], times[nxt], nxt - prev
        for k in range(prev + 1, nxt):
            times[k] = t0 + (t1 - t0) * (k - prev) / gap
        i = nxt


def align_blocks_to_transcript(blocks: List[Block], tr: Transcript,
                               audio_seconds: float) -> float:
    """
    Выравнивает склеенные тексты блоков на пословный транскрипт (difflib).
    Каждому блоку проставляет реальные start/end. Возвращает coverage —
    долю слов транскрипта, нашедших своё место в текстах блоков.
    """
    t_words: List[Word] = list(tr.words)
    if not t_words and tr.segments:
        for seg in tr.segments:
            toks = _TOKEN_RE.findall(seg.text)
            if not toks:
                continue
            step = max(0.01, (seg.end - seg.start)) / len(toks)
            for k, tok in enumerate(toks):
                ws = seg.start + k * step
                t_words.append(Word(tok, ws, ws + step))
    if not t_words:
        return 0.0

    t_tokens = [w.text.lower().strip() for w in t_words]

    b_tokens: List[str] = []
    b_owner: List[int] = []
    for bi, block in enumerate(blocks):
        for tok in _TOKEN_RE.findall(block.text.lower()):
            b_tokens.append(tok)
            b_owner.append(bi)
    if not b_tokens:
        return 0.0

    matcher = difflib.SequenceMatcher(a=b_tokens, b=t_tokens, autojunk=False)
    b_time: List[Optional[float]] = [None] * len(b_tokens)
    matched = 0
    for blk in matcher.get_matching_blocks():
        for off in range(blk.size):
            b_time[blk.a + off] = t_words[blk.b + off].start
            matched += 1
    coverage = matched / max(1, len(t_tokens))

    _interpolate(b_time)
    if all(v is None for v in b_time):
        return coverage

    for bi, block in enumerate(blocks):
        times = [b_time[i] for i in range(len(b_tokens)) if b_owner[i] == bi]
        times = [t for t in times if t is not None]
        if times:
            block.start = min(times)
            block.end = max(times)
            block.aligned = True

    # монотонность и покрытие всего аудио без дыр:
    # границы блоков = их старты; конец блока = старт следующего
    prev = 0.0
    for block in blocks:
        if block.start < prev:
            block.start = prev
        prev = max(prev, block.start) + 0.01
    blocks[0].start = 0.0
    for i, block in enumerate(blocks):
        block.end = blocks[i + 1].start if i + 1 < len(blocks) else audio_seconds
        if block.end <= block.start:
            block.end = block.start + 0.01
    return coverage


def blocks_proportional(blocks: List[Block], audio_seconds: float) -> None:
    """Fallback без транскрипта: время блоков пропорционально длине текста."""
    total_chars = sum(max(len(b.text), 1) for b in blocks) or 1
    t = 0.0
    for b in blocks:
        dur = audio_seconds * max(len(b.text), 1) / total_chars
        b.start, b.end, b.aligned = t, t + dur, False
        t += dur
    blocks[-1].end = audio_seconds


# =========================================================
# СЦЕНЫ: СЛОТЫ ВНУТРИ БЛОКА + АНТИ-ФРИЗ
# =========================================================

def scenes_with_slots(blocks: List[Block]) -> List[Scene]:
    """Интервал блока делится между его сценами пропорционально длине роликов."""
    all_scenes: List[Scene] = []
    for block in blocks:
        span = max(0.0, block.end - block.start)
        weights = [max(sc.clip_dur, 0.5) for sc in block.scenes]
        wsum = sum(weights) or 1.0
        t = block.start
        for sc, w in zip(block.scenes, weights):
            dur = span * w / wsum
            sc.start, sc.end = t, t + dur
            t += dur
            all_scenes.append(sc)
        if block.scenes:
            block.scenes[-1].end = block.end
    all_scenes.sort(key=lambda s: s.global_scene_index)
    return all_scenes


def rebalance(scenes: List[Scene], audio_seconds: float) -> Dict[str, Any]:
    """
    Анти-фриз: слот <= clip_dur * MAX_STRETCH; недостающие ролики получают
    потолок 0 (их время уходит соседям). Излишки разливаются на ближайшие
    слоты со свободным запасом. Сумма длительностей == длине аудио.
    """
    n = len(scenes)
    orig = [max(0.0, sc.end - sc.start) for sc in scenes]
    caps = [(sc.clip_dur * MAX_STRETCH) if sc.exists else 0.0 for sc in scenes]

    total = sum(orig)
    capacity = sum(caps)
    if capacity < total - EPS:
        scale = total / max(capacity, EPS)
        caps = [c * scale for c in caps]
        warn(f"Суммарной длины роликов не хватает — все потолки подняты в {scale:.2f} раза "
             f"(замедление будет заметнее).")

    d = list(orig)
    overflows: List[Tuple[int, float]] = []
    for i in range(n):
        if d[i] > caps[i] + EPS:
            overflows.append((i, d[i] - caps[i]))
            d[i] = caps[i]

    for i, ov in overflows:
        left, right, rest = i - 1, i + 1, ov
        while rest > EPS:
            while left >= 0 and caps[left] - d[left] <= EPS:
                left -= 1
            while right < n and caps[right] - d[right] <= EPS:
                right += 1
            if left < 0 and right >= n:
                for k in range(n):
                    d[k] += rest / n
                break
            dist_l = i - left if left >= 0 else None
            dist_r = right - i if right < n else None
            targets = []
            if dist_l is not None and (dist_r is None or dist_l <= dist_r):
                targets.append(left)
            if dist_r is not None and (dist_l is None or dist_r <= dist_l):
                targets.append(right)
            share = rest / len(targets)
            for t in targets:
                take = min(share, caps[t] - d[t])
                d[t] += take
                rest -= take

    # восстановить границы, подогнать хвост
    t = 0.0
    starts = []
    for dur in d:
        starts.append(t)
        t += dur
    if t > 0 and abs(t - audio_seconds) > 0.25:
        k = audio_seconds / t
        starts = [s * k for s in starts]
        d = [dur * k for dur in d]

    shifts = []
    for sc, s, dur in zip(scenes, starts, d):
        shifts.append(abs(s - sc.start))
        sc.start, sc.end = s, s + dur
    scenes[-1].end = audio_seconds

    max_factor = max((max(0.0, sc.end - sc.start) / sc.clip_dur)
                     for sc in scenes if sc.exists)
    stats = {
        "capped_slots": len(overflows),
        "worst_slot_before": round(max(orig), 1),
        "worst_slot_after": round(max(d), 1),
        "max_slow_factor": round(max_factor, 2),
        "mean_shift": round(sum(shifts) / max(len(shifts), 1), 1),
        "max_shift": round(max(shifts) if shifts else 0.0, 1),
    }
    info(f"Анти-фриз: обрезано слотов {stats['capped_slots']}, "
         f"худший слот {stats['worst_slot_before']}s -> {stats['worst_slot_after']}s, "
         f"макс. замедление x{stats['max_slow_factor']}, "
         f"сдвиг синхронизации средн./макс. {stats['mean_shift']}/{stats['max_shift']}s")
    return stats


# =========================================================
# СОХРАНЕНИЕ ТАЙМ-КОДОВ
# =========================================================

def save_timecodes(scenes: List[Scene], audio: Path, audio_seconds: float,
                   strategy: str, stats: Dict[str, Any]) -> None:
    payload = {
        "locale": "smart",
        "audio_file": audio.name,
        "audio_seconds": round(audio_seconds, 3),
        "strategy": strategy,
        "rebalance": stats,
        "scenes": [
            {
                "global_scene_index": sc.global_scene_index,
                "video_file": sc.video_file,
                "exists": sc.exists,
                "block_id": sc.block_id,
                "scene_index_in_block": sc.scene_index_in_block,
                "sentence_indexes": [],
                "start": round(sc.start, 3),
                "end": round(sc.end, 3),
                "duration": round(sc.end - sc.start, 3),
                "subject": sc.subject,
                "block_text": sc.block_text,
            }
            for sc in scenes
        ],
    }
    backup(TIMECODES_JSON)
    backup(TIMECODES_TXT)
    save_json(TIMECODES_JSON, payload)

    lines = [
        f"# Тайм-коды [smart] | аудио={audio.name} | {audio_seconds:.2f}s | стратегия={strategy}",
        "# файл    старт --> конец  (длина)  | блок/сцена | текст",
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
    TIMECODES_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    info(f"Тайм-коды сохранены: {TIMECODES_JSON.name}, {TIMECODES_TXT.name}")


# =========================================================
# РЕНДЕР (замедление вместо фриза, всегда)
# =========================================================

def build_clip_filter(slot: float, src_dur: float, w: int, h: int) -> str:
    base = (f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
            f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={TARGET_FPS}")
    if slot <= src_dur + 0.05:
        return base
    factor = slot / src_dur
    if factor <= MAX_SLOW_FACTOR:
        return f"setpts={factor:.5f}*PTS," + base
    extra = slot - src_dur * MAX_SLOW_FACTOR
    return (f"setpts={MAX_SLOW_FACTOR:.5f}*PTS," + base +
            f",tpad=stop_mode=clone:stop_duration={extra:.3f}")


def render_montage(scenes: List[Scene], audio: Path,
                   preview_seconds: Optional[float]) -> None:
    if not which("ffmpeg"):
        warn("ffmpeg не найден — монтаж пропущен, тайм-коды сохранены.")
        return

    render_list = [sc for sc in scenes if sc.exists and (sc.end - sc.start) >= MIN_SLOT_SECONDS]
    dropped = sum(1 for sc in scenes if sc.exists) - len(render_list)
    if dropped:
        info(f"Слоты короче {MIN_SLOT_SECONDS}s ({dropped} шт.) слиты с соседними.")
    # слитые/пропущенные интервалы отдаём предыдущему клипу, чтобы не было дыр
    merged: List[Scene] = []
    for sc in render_list:
        if merged and sc.start > merged[-1].end + EPS:
            merged[-1].end = sc.start
        merged.append(sc)
    if merged:
        merged[0].start = 0.0
    render_list = merged

    out_video = FINAL_VIDEO
    if preview_seconds:
        render_list = [sc for sc in render_list if sc.start < preview_seconds]
        for sc in render_list:
            sc.end = min(sc.end, preview_seconds)
        out_video = PREVIEW_VIDEO
        info(f"ПРЕДПРОСМОТР: первые {preview_seconds:.0f}s — {len(render_list)} сцен.")
    if not render_list:
        warn("Нечего рендерить.")
        return

    w, h = (1920, 1080)
    if TARGET_RESOLUTION:
        try:
            ws, hs = TARGET_RESOLUTION.lower().split("x")
            w, h = int(ws), int(hs)
        except Exception:
            warn(f"TARGET_RESOLUTION={TARGET_RESOLUTION!r} некорректно.")
    else:
        for sc in render_list:
            res = ffprobe_resolution(VIDEOS_DIR / sc.video_file)
            if res:
                w, h = res
                break

    info(f"Рендер {len(render_list)} сегментов в {w}x{h}@{TARGET_FPS} "
         f"(длинные слоты — замедление, без фризов)...")
    tmp_dir = BASE_DIR / "_montage_tmp_smart"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    segments: List[Path] = []
    total = len(render_list)
    for i, sc in enumerate(render_list, start=1):
        slot = sc.end - sc.start
        out = tmp_dir / f"seg_{sc.global_scene_index:04d}.mp4"
        vf = build_clip_filter(slot, max(sc.clip_dur, 0.2), w, h)
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
               "-i", str(VIDEOS_DIR / sc.video_file),
               "-t", f"{slot:.3f}", "-vf", vf, "-an",
               "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
               "-pix_fmt", "yuv420p", str(out)]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            segments.append(out)
        except subprocess.CalledProcessError as e:
            warn(f"Не удалось обработать {sc.video_file}: {e.stderr[:200]}")
        sys.stdout.write(f"\rМонтаж {progress_bar(i, total)}  [{sc.video_file}]      ")
        sys.stdout.flush()
    sys.stdout.write("\n")

    if not segments:
        warn("Ни одного сегмента — монтаж отменён.")
        return

    concat_file = tmp_dir / "concat.txt"
    concat_file.write_text("".join(f"file '{s.as_posix()}'\n" for s in segments),
                           encoding="utf-8")
    silent = tmp_dir / "video_no_audio.mp4"
    try:
        subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                        "-f", "concat", "-safe", "0", "-i", str(concat_file),
                        "-c", "copy", str(silent)],
                       check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError:
        subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                        "-f", "concat", "-safe", "0", "-i", str(concat_file),
                        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                        "-pix_fmt", "yuv420p", str(silent)],
                       check=True, capture_output=True, text=True)

    mux = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
           "-i", str(silent), "-i", str(audio),
           "-map", "0:v:0", "-map", "1:a:0",
           "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest"]
    if preview_seconds:
        mux += ["-t", f"{preview_seconds:.3f}"]
    mux.append(str(out_video))
    try:
        subprocess.run(mux, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        fail(f"Не удалось наложить аудио: {e.stderr[:300]}")

    shutil.rmtree(tmp_dir, ignore_errors=True)
    info(f"[DONE] Готовый монтаж: {out_video}")


# =========================================================
# MAIN
# =========================================================

def main() -> None:
    global MAX_STRETCH
    parser = argparse.ArgumentParser(
        description="Аудио + промпты + ролики -> тайм-коды по смыслу + монтаж без фризов.")
    parser.add_argument("--audio", type=Path, default=None,
                        help="Аудиофайл озвучки (по умолчанию ищется в базовой папке).")
    parser.add_argument("--preview", nargs="?", type=float, const=60.0, default=None,
                        metavar="СЕК", help="Смонтировать только первые N секунд (60).")
    parser.add_argument("--timecodes-only", action="store_true",
                        help="Только посчитать и сохранить тайм-коды, без монтажа.")
    parser.add_argument("--max-stretch", type=float, default=None,
                        help=f"Потолок slot/clip для анти-фриза (по умолчанию {MAX_STRETCH}).")
    args = parser.parse_args()

    if args.max_stretch:
        if args.max_stretch < 1.0:
            fail("--max-stretch должен быть >= 1.0")
        MAX_STRETCH = args.max_stretch

    if not BASE_DIR.exists():
        fail(f"Нет базовой папки {BASE_DIR}. Задай переменную PROMPTS_BASE_DIR.")

    audio = find_audio_file(args.audio)
    audio_seconds = probe_duration(audio)
    if not audio_seconds:
        fail(f"Не удалось прочитать длительность аудио {audio.name} (нужен ffmpeg/ffprobe).")
    info(f"Аудио: {audio.name} | {audio_seconds:.1f}s")

    blocks = load_blocks()
    tr = transcribe_audio(audio)

    strategy = "proportional"
    if tr:
        coverage = align_blocks_to_transcript(blocks, tr, audio_seconds)
        info(f"Выравнивание блоков на аудио: coverage={coverage:.0%}")
        if coverage >= ALIGN_MIN_COVERAGE:
            strategy = "whisper-block-align"
        else:
            warn(f"coverage {coverage:.0%} < {ALIGN_MIN_COVERAGE:.0%} — тексты промптов не "
                 f"совпадают с озвучкой, откатываюсь на пропорциональные тайминги.")
            blocks_proportional(blocks, audio_seconds)
    else:
        blocks_proportional(blocks, audio_seconds)

    info(f"Стратегия: {strategy}")
    scenes = scenes_with_slots(blocks)
    stats = rebalance(scenes, audio_seconds)
    save_timecodes(scenes, audio, audio_seconds, strategy + "+antifreeze", stats)

    if args.timecodes_only:
        info("Режим --timecodes-only: монтаж пропущен.")
        return
    render_montage(scenes, audio, args.preview)


if __name__ == "__main__":
    main()
