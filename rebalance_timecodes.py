#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rebalance_timecodes.py — перераспределение слотов в video_timecodes[_xx].json/.txt

ПРОБЛЕМА
========
Мастер-скрипт иногда склеивает огромный кусок текста в ОДИН блок (на русском —
из-за кавычек-«ёлочек», многоточий и длинных кусков без точек), блок получает
максимум 3 сцены, и при монтаже клип ~8с вынужден закрывать слот в 40–100+
секунд. FILL_MODE=freeze превращает это в долгий застывший кадр.

ЧТО ДЕЛАЕТ СКРИПТ
=================
НЕ зацикливает и НЕ добавляет клипы. Работает только с таймингами:

  1) Каждому слоту ставится потолок: длина клипа * MAX_STRETCH (по умолчанию 1.6,
     т.е. клип 8с может закрыть максимум ~12.8с лёгким замедлением).
  2) Излишек переполненных слотов "разливается" на соседние клипы: ближайшие
     соседи со свободным запасом забирают время первыми, дальние — только когда
     ближним уже некуда расти. Порядок клипов сохраняется, суммарная длина
     остаётся равной длине аудио.
  3) Пишется новый video_timecodes.json (+ читаемый .txt), совместимый с
     audio_video_sync_montage.py --render-only. Старые файлы сохраняются в .bak.

Синхронизация с речью страдает только в окрестности проблемных мест (это
неизбежно: дыру в 5 минут можно закрыть только клипами из соседних участков),
зато ни один кадр больше не стоит на месте.

ЗАПУСК
======
    python rebalance_timecodes.py
        # без аргументов: сам берёт video_timecodes.json из базовой папки
        # конвейера (PROMPTS_BASE_DIR, по умолчанию ~/Desktop/ПРОМПТЫ) и
        # папку ВИДЕО оттуда же.

    python rebalance_timecodes.py /path/to/video_timecodes.json   # явный путь
    python rebalance_timecodes.py /path/to/video_timecodes.txt    # если json нет

    Опции:
      --max-stretch 1.6    потолок slot/clip (насколько можно замедлить клип)
      --clip-seconds 8     длительность клипа, если ffprobe/файл недоступны
      --videos-dir DIR     папка с 0001.mp4... для точных длительностей (ffprobe)
      --output PATH        куда писать json (по умолчанию — поверх входного json
                           с бэкапом .bak)
      --dry-run            только показать отчёт, ничего не записывать

Затем монтаж (slow вместо freeze, чтобы растянутые слоты замедлялись, а не
замирали; --force потому что final_montage.mp4 уже существует):

    FILL_MODE=slow python audio_video_sync_montage.py --render-only --force
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

EPS = 1e-4

# Та же базовая папка, что и в остальных скриптах конвейера.
BASE_DIR = Path(os.getenv("PROMPTS_BASE_DIR", "/Users/aleksandrtomilov/Desktop/ПРОМПТЫ"))
DEFAULT_VIDEOS_DIR = BASE_DIR / "ВИДЕО"


def info(msg: str) -> None:
    print(f"[INFO] {msg}", flush=True)


def warn(msg: str) -> None:
    print(f"[WARN] {msg}", flush=True)


def fail(msg: str) -> None:
    print(f"[ERROR] {msg}", flush=True)
    sys.exit(1)


def format_tc(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    m = int(seconds // 60)
    s = seconds - m * 60
    return f"{m:02d}:{s:06.3f}"


def which(program: str) -> Optional[str]:
    return shutil.which(program)


def ffprobe_duration(path: Path) -> Optional[float]:
    if not which("ffprobe") or not path.exists():
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


# =========================================================
# ЧТЕНИЕ ВХОДНЫХ ДАННЫХ (json или txt)
# =========================================================

TXT_HEADER_RE = re.compile(
    r"#\s*Тайм-коды\s*\[(?P<loc>\w+)\]\s*\|\s*аудио=(?P<audio>.+?)\s*\|\s*(?P<dur>[\d.]+)s\s*\|\s*стратегия=(?P<strat>\S+)"
)
TXT_LINE_RE = re.compile(
    r"^(?P<file>\S+\.mp4)\s+"
    r"(?P<sm>\d+):(?P<ss>\d+(?:\.\d+)?)\s*-->\s*(?P<em>\d+):(?P<es>\d+(?:\.\d+)?)\s+"
    r"\(\s*[\d.]+s\)\s+\|\s*b(?P<block>\d+)\.s(?P<sib>\d+)\s*\|\s*(?P<text>.*)$"
)
TXT_MISSING_MARK = "[!] нет файла"


def load_payload_from_txt(path: Path) -> Dict[str, Any]:
    lines = path.read_text(encoding="utf-8").splitlines()
    payload: Dict[str, Any] = {
        "locale": "en", "audio_file": "", "audio_seconds": 0.0,
        "strategy": "unknown", "scenes": [],
    }
    for line in lines:
        m = TXT_HEADER_RE.search(line)
        if m:
            payload["locale"] = m.group("loc")
            payload["audio_file"] = m.group("audio")
            payload["audio_seconds"] = float(m.group("dur"))
            payload["strategy"] = m.group("strat")
            continue
        m = TXT_LINE_RE.match(line.strip())
        if not m:
            continue
        text = m.group("text")
        missing = TXT_MISSING_MARK in text
        if missing:
            text = text.replace(TXT_MISSING_MARK, "").strip()
        start = int(m.group("sm")) * 60 + float(m.group("ss"))
        end = int(m.group("em")) * 60 + float(m.group("es"))
        stem = Path(m.group("file")).stem
        payload["scenes"].append({
            "global_scene_index": int(stem) if stem.isdigit() else len(payload["scenes"]) + 1,
            "video_file": m.group("file"),
            "exists": not missing,
            "block_id": int(m.group("block")),
            "scene_index_in_block": int(m.group("sib")),
            "sentence_indexes": [],
            "start": start,
            "end": end,
            "duration": round(end - start, 3),
            "subject": "",
            "block_text": text,
        })
    if not payload["scenes"]:
        fail(f"В {path} не удалось распарсить ни одной строки тайм-кодов.")
    if not payload["audio_seconds"]:
        payload["audio_seconds"] = payload["scenes"][-1]["end"]
    return payload


def load_payload(input_path: Path) -> Dict[str, Any]:
    if input_path.suffix.lower() == ".json":
        data = json.loads(input_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not data.get("scenes"):
            fail(f"{input_path}: нет поля scenes — это не файл тайм-кодов.")
        return data
    # txt: если рядом лежит одноимённый json — он богаче, берём его
    sibling = input_path.with_suffix(".json")
    if sibling.exists():
        info(f"Рядом найден {sibling.name} — использую его вместо txt (там больше данных).")
        return load_payload(sibling)
    return load_payload_from_txt(input_path)


# =========================================================
# РЕБАЛАНСИРОВКА
# =========================================================

def clip_durations(scenes: List[Dict[str, Any]], videos_dir: Optional[Path],
                   fallback: float) -> List[float]:
    """Реальная длительность каждого клипа (ffprobe), иначе fallback."""
    durs: List[float] = []
    probed = 0
    for sc in scenes:
        dur = None
        if videos_dir is not None:
            dur = ffprobe_duration(videos_dir / str(sc.get("video_file", "")))
            if dur:
                probed += 1
        durs.append(dur if dur and dur > 0.2 else fallback)
    if videos_dir is not None and probed:
        info(f"Длительности клипов: {probed}/{len(scenes)} измерены ffprobe, "
             f"остальные = {fallback:.1f}s.")
    else:
        info(f"Длительности клипов приняты равными {fallback:.1f}s "
             f"(для точности укажи --videos-dir).")
    return durs


def rebalance_durations(orig: List[float], caps: List[float]) -> List[float]:
    """
    Ограничивает каждый слот его потолком cap и разливает излишек на ближайшие
    слоты со свободным запасом (сначала самые близкие, дальше — по мере
    заполнения). Сумма длительностей сохраняется точно.
    """
    n = len(orig)
    total = sum(orig)
    capacity = sum(caps)
    if capacity < total - EPS:
        # Общий дефицит материала: равномерно поднимаем потолки (лёгкое общее
        # замедление лучше, чем фризы).
        scale = total / capacity
        caps = [c * scale for c in caps]
        warn(f"Суммарной вместимости клипов не хватает — поднимаю потолок "
             f"всех слотов в {scale:.3f} раза.")

    d = list(orig)
    overflows: List[tuple] = []
    for i in range(n):
        if d[i] > caps[i] + EPS:
            overflows.append((i, d[i] - caps[i]))
            d[i] = caps[i]

    for i, ov in overflows:
        left, right = i - 1, i + 1
        rest = ov
        while rest > EPS:
            while left >= 0 and caps[left] - d[left] <= EPS:
                left -= 1
            while right < n and caps[right] - d[right] <= EPS:
                right += 1
            if left < 0 and right >= n:
                # некуда деть (не должно случаться после масштабирования caps) —
                # раздаём остаток равномерно всем, жертвуя потолком
                for k in range(n):
                    d[k] += rest / n
                rest = 0.0
                break
            # выбираем ближайший свободный слот; при равном расстоянии — оба
            dist_l = i - left if left >= 0 else None
            dist_r = right - i if right < n else None
            targets: List[int] = []
            if dist_l is not None and (dist_r is None or dist_l <= dist_r):
                targets.append(left)
            if dist_r is not None and (dist_l is None or dist_r <= dist_l):
                targets.append(right)
            share = rest / len(targets)
            for t in targets:
                take = min(share, caps[t] - d[t])
                d[t] += take
                rest -= take
    return d


def apply_rebalance(payload: Dict[str, Any], videos_dir: Optional[Path],
                    clip_seconds: float, max_stretch: float) -> Dict[str, Any]:
    scenes = payload["scenes"]
    audio_seconds = float(payload.get("audio_seconds") or scenes[-1]["end"])

    orig = [max(0.0, float(sc["end"]) - float(sc["start"])) for sc in scenes]
    clips = clip_durations(scenes, videos_dir, clip_seconds)
    caps = [c * max_stretch for c in clips]

    new = rebalance_durations(orig, caps)

    # Восстанавливаем границы и подгоняем хвост точно под длину аудио.
    starts: List[float] = []
    t = 0.0
    for dur in new:
        starts.append(t)
        t += dur
    if abs(t - audio_seconds) > 0.5:
        scale = audio_seconds / t if t > 0 else 1.0
        starts = [s * scale for s in starts]
        new = [dur * scale for dur in new]
        t = audio_seconds

    old_starts = [float(sc["start"]) for sc in scenes]
    shifts = [abs(a - b) for a, b in zip(starts, old_starts)]
    capped = sum(1 for o, c in zip(orig, caps) if o > c + EPS)
    grown = sum(1 for o, nn in zip(orig, new) if nn > o + EPS)
    max_factor = max((nn / c) for nn, c in zip(new, clips))

    info("--- ОТЧЁТ ---")
    info(f"Слотов всего: {len(scenes)} | аудио: {audio_seconds:.1f}s")
    info(f"Худший слот было: {max(orig):.1f}s -> стало: {max(new):.1f}s")
    info(f"Обрезано переполненных слотов: {capped} | получили добавку: {grown}")
    info(f"Максимальное замедление при рендере: x{max_factor:.2f} "
         f"(рендерить с FILL_MODE=slow)")
    info(f"Сдвиг от исходной синхронизации: средний {sum(shifts)/len(shifts):.1f}s, "
         f"максимальный {max(shifts):.1f}s")

    for sc, s, dur in zip(scenes, starts, new):
        sc["start"] = round(s, 3)
        sc["end"] = round(s + dur, 3)
        sc["duration"] = round(dur, 3)
    scenes[-1]["end"] = round(audio_seconds, 3)
    scenes[-1]["duration"] = round(scenes[-1]["end"] - scenes[-1]["start"], 3)

    payload["strategy"] = str(payload.get("strategy", "unknown")) + "+rebalanced"
    payload["rebalance"] = {
        "max_stretch": max_stretch,
        "clip_seconds_fallback": clip_seconds,
        "capped_slots": capped,
        "max_slow_factor": round(max_factor, 3),
    }
    return payload


# =========================================================
# ЗАПИСЬ РЕЗУЛЬТАТА
# =========================================================

def backup(path: Path) -> None:
    if path.exists():
        bak = path.with_suffix(path.suffix + ".bak")
        if not bak.exists():
            shutil.copy2(path, bak)
            info(f"Бэкап: {bak.name}")


def write_txt(payload: Dict[str, Any], path: Path) -> None:
    lines = [
        f"# Тайм-коды [{payload.get('locale', '?')}] | аудио={payload.get('audio_file', '?')} | "
        f"{float(payload.get('audio_seconds', 0)):.2f}s | стратегия={payload.get('strategy', '?')}",
        "# файл    старт --> конец  (длина)  | блок/сцена | текст",
        "",
    ]
    for sc in payload["scenes"]:
        flag = "" if sc.get("exists", True) else f"  {TXT_MISSING_MARK}"
        preview = str(sc.get("block_text") or sc.get("subject") or "").strip().replace("\n", " ")
        if len(preview) > 70:
            preview = preview[:67] + "..."
        dur = float(sc["end"]) - float(sc["start"])
        lines.append(
            f"{sc['video_file']}  {format_tc(float(sc['start']))} --> {format_tc(float(sc['end']))}  "
            f"({dur:4.1f}s)  | b{sc.get('block_id', 0)}.s{sc.get('scene_index_in_block', 1)} | {preview}{flag}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def find_default_input() -> Path:
    """
    Ищет файл тайм-кодов, если он не указан аргументом:
    сначала в текущей папке, потом в BASE_DIR; json предпочтительнее txt.
    """
    for folder in (Path.cwd(), BASE_DIR):
        for name in ("video_timecodes.json", "video_timecodes.txt"):
            cand = folder / name
            if cand.exists():
                return cand
    fail(f"Не нашёл video_timecodes.json/.txt ни в {Path.cwd()}, ни в {BASE_DIR}. "
         f"Укажи путь аргументом или задай PROMPTS_BASE_DIR.")
    raise SystemExit  # для type-checker'а; fail() уже завершил процесс


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Перераспределяет время переполненных слотов тайм-кодов на "
                    "соседние клипы (без лупов и фризов)."
    )
    parser.add_argument("input", type=Path, nargs="?", default=None,
                        help="video_timecodes[_xx].json или .txt. Если не указан — "
                             "ищется автоматически в текущей папке и в "
                             f"{BASE_DIR}.")
    parser.add_argument("--max-stretch", type=float, default=1.6,
                        help="Потолок slot/clip (по умолчанию 1.6 — лёгкое замедление).")
    parser.add_argument("--clip-seconds", type=float, default=8.0,
                        help="Длительность клипа, если её нельзя измерить (8).")
    parser.add_argument("--videos-dir", type=Path, default=None,
                        help="Папка с роликами для точных длительностей (ffprobe).")
    parser.add_argument("--output", type=Path, default=None,
                        help="Куда писать json (по умолчанию поверх входного json, с .bak).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Только отчёт, без записи файлов.")
    args = parser.parse_args()

    if args.input is None:
        args.input = find_default_input()
        info(f"Файл тайм-кодов не указан — беру {args.input}")
    if not args.input.exists():
        fail(f"Нет входного файла: {args.input}")
    if args.max_stretch < 1.0:
        fail("--max-stretch должен быть >= 1.0")
    if args.videos_dir is None:
        for cand in (Path.cwd() / "ВИДЕО", DEFAULT_VIDEOS_DIR,
                     args.input.resolve().parent / "ВИДЕО"):
            if cand.is_dir():
                args.videos_dir = cand
                info(f"Папка с роликами: {cand}")
                break

    payload = load_payload(args.input)
    payload = apply_rebalance(payload, args.videos_dir, args.clip_seconds, args.max_stretch)

    if args.dry_run:
        info("Режим --dry-run: файлы не записаны.")
        return

    out_json = args.output
    if out_json is None:
        out_json = args.input if args.input.suffix.lower() == ".json" \
            else args.input.with_suffix(".json")
    out_txt = out_json.with_suffix(".txt")

    backup(out_json)
    backup(out_txt)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_txt(payload, out_txt)
    info(f"Записано: {out_json} и {out_txt}")
    info("Дальше: FILL_MODE=slow python audio_video_sync_montage.py --render-only --force")


if __name__ == "__main__":
    main()
