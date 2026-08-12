#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
СКРИПТ 3 — ОЖИВЛЕНИЕ ФОТО.

Берёт идеи со статусом ФОТО и превращает картинку в короткое видео.
Оживление НАМЕРЕННО без конкретики: никакого сюжета и никаких действий —
только естественное лёгкое движение, чтобы кадр «дышал» и не разъезжался.
Это то, что нужно для первых двух секунд ролика.

Механика та же, что в вашем рабочем скрипте:
  картинка -> inline base64 в inputs[] -> flow_video_from_ingredients -> mp4
Постоянные ошибки (safety) не повторяются, временные — ограниченно.

Запуск:
    python3 step3_animate.py
    python3 step3_animate.py --limit 3
    python3 step3_animate.py --idea 7
    python3 step3_animate.py --workers 3      # несколько генераций параллельно

Ключ берётся из окружения, как в старом скрипте:
    export FAST_GEN_API_KEY='...'
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

import config
import fastgen_client
from ideas_store import (
    STATUS_PHOTO,
    STATUS_VIDEO,
    Idea,
    load_ideas,
    save_ideas,
)

# Универсальный промпт оживления — одинаковый для всех идей.
# Задача: только микродвижение, без новых действий и без смены композиции.
ANIMATION_PROMPT = (
    "Animate this photo with very slow, subtle, natural motion only. "
    "Keep the exact same person, the same face, the same clothing, the same composition, "
    "the same colors and the same lighting as in the source image. "
    "Allowed motion: a barely noticeable camera push-in, gentle breathing, "
    "a small natural shift of the head or shoulders, slight movement of hair and fabric in the air, "
    "soft ambient movement in the background. "
    "No new actions, no gestures, no walking, no talking, no scene changes, no new objects, "
    "no morphing of the face, no distortion of hands or body. "
    "Keep it photorealistic and documentary-like, as if it were a real short clip filmed on a phone. "
    "Do not add any text, subtitles, captions, labels, logos or watermarks."
)


def _safe_name(text: str) -> str:
    keep = [c if (c.isalnum() or c in " -_") else "_" for c in text]
    return "".join(keep).strip().replace(" ", "_")[:40] or "idea"


def animate_one(idea: Idea) -> tuple[Idea, bool, str]:
    """Оживляет фото одной идеи. Возвращает (идея, успех, путь_или_ошибка)."""
    photo_path = Path(idea.photo_file).expanduser()
    if not photo_path.exists():
        return idea, False, f"нет файла фото: {photo_path}"

    out_path = config.VIDEOS_DIR / f"{idea.number:03d}_{_safe_name(idea.title)}.mp4"
    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"[{idea.number:03d}] уже оживлено, пропускаю: {out_path.name}")
        return idea, True, str(out_path)

    print(f"[{idea.number:03d}] оживляю: {photo_path.name}")
    try:
        fastgen_client.animate_image(photo_path, ANIMATION_PROMPT, out_path)
    except fastgen_client.PermanentError as exc:
        print(f"[{idea.number:03d}] ЗАБЛОКИРОВАНО провайдером: {exc}")
        return idea, False, str(exc)
    except Exception as exc:  # noqa: BLE001
        print(f"[{idea.number:03d}] ОШИБКА: {exc}")
        return idea, False, str(exc)

    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"[{idea.number:03d}] готово: {out_path.name} ({size_mb:.2f} MB)")
    return idea, True, str(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Оживление сгенерированных фото в короткие видео.")
    parser.add_argument("--limit", type=int, default=0, help="Максимум идей за запуск (0 = все).")
    parser.add_argument("--idea", type=int, default=0, help="Обработать только идею с этим номером.")
    parser.add_argument("--workers", type=int, default=1, help="Сколько генераций параллельно.")
    parser.add_argument("--ideas-file", default=str(config.IDEAS_FILE))
    args = parser.parse_args()

    config.require_fastgen_key()
    config.ensure_dirs()

    ideas_path = Path(args.ideas_file).expanduser()
    ideas = load_ideas(ideas_path)
    if not ideas:
        raise SystemExit(f"В файле нет идей: {ideas_path}")

    if args.idea:
        queue = [i for i in ideas if i.number == args.idea]
        if not queue:
            raise SystemExit(f"Идея №{args.idea} не найдена.")
    else:
        queue = [i for i in ideas if i.status == STATUS_PHOTO and i.photo_file]
        if args.limit:
            queue = queue[: args.limit]

    if not queue:
        print("Нет идей с готовым фото. Сначала запустите: python3 step2_photo.py")
        return

    print(f"Операция: {config.OP_VIDEO_FROM_IMAGE}")
    print(f"К обработке: {len(queue)} шт., потоков: {args.workers}")

    save_lock = Lock()
    done = 0

    def commit(idea: Idea, ok: bool, result: str) -> None:
        nonlocal done
        if not ok:
            return
        with save_lock:
            idea.video_file = result
            idea.status = STATUS_VIDEO
            save_ideas(ideas_path, ideas, make_backup=False)
            done += 1

    if args.workers > 1:
        workers = max(1, min(args.workers, len(queue)))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(animate_one, idea): idea for idea in queue}
            for future in as_completed(futures):
                commit(*future.result())
    else:
        for idea in queue:
            commit(*animate_one(idea))

    print(f"\nГотово. Оживлено: {done} из {len(queue)}")
    print(f"Папка: {config.VIDEOS_DIR}")
    if done:
        print("Дальше: python3 step4_montage.py")


if __name__ == "__main__":
    main()
