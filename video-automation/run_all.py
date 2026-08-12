#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ЗАПУСК ВСЕГО КОНВЕЙЕРА ОДНОЙ КОМАНДОЙ.

Прогоняет по порядку:
    step1_ideas.py    -> придумать новые идеи (с учётом просмотров)
    step2_photo.py    -> сгенерировать фото по вашему лицу
    step3_animate.py  -> оживить фото
    step4_montage.py  -> смонтировать ролики

Запуск:
    python3 run_all.py                  # 3 новые идеи -> 3 готовых ролика
    python3 run_all.py --count 5
    python3 run_all.py --skip-ideas     # работать по уже придуманным идеям

Если нужно отсмотреть идеи перед съёмкой (рекомендуется) — запускайте
шаги по отдельности: сначала step1, глазами проверьте ИДЕИ.txt,
удалите или поправьте что не нравится, потом step2/3/4.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run_step(script: str, args: list[str]) -> bool:
    print(f"\n{'=' * 70}\n>>> {script} {' '.join(args)}\n{'=' * 70}")
    result = subprocess.run([sys.executable, str(HERE / script), *args])
    if result.returncode != 0:
        print(f"\n[STOP] {script} завершился с ошибкой (код {result.returncode}).")
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Полный прогон конвейера.")
    parser.add_argument("--count", type=int, default=3, help="Сколько роликов сделать за прогон.")
    parser.add_argument("--skip-ideas", action="store_true", help="Не придумывать новые идеи.")
    parser.add_argument("--backend", choices=["openai", "fastgen"], default=None)
    parser.add_argument("--workers", type=int, default=1, help="Параллельных генераций видео.")
    args = parser.parse_args()

    count = str(args.count)

    if not args.skip_ideas:
        if not run_step("step1_ideas.py", ["--count", count]):
            return

    photo_args = ["--limit", count]
    if args.backend:
        photo_args += ["--backend", args.backend]
    if not run_step("step2_photo.py", photo_args):
        return

    if not run_step("step3_animate.py", ["--limit", count, "--workers", str(args.workers)]):
        return

    if not run_step("step4_montage.py", ["--limit", count]):
        return

    print("\n" + "=" * 70)
    print("ВСЁ ГОТОВО. Ролики лежат в папке ГОТОВОЕ.")
    print("После публикации впишите просмотры в ИДЕИ.txt — следующий прогон")
    print("будет придумывать идеи по тому, что реально зашло.")
    print("=" * 70)


if __name__ == "__main__":
    main()
