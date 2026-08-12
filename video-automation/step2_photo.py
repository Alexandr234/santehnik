#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
СКРИПТ 2 — ГЕНЕРАЦИЯ ФОТО ПО ВАШЕМУ ЛИЦУ.

Что делает:
  1. Берёт из ИДЕИ.txt все идеи со статусом НОВАЯ.
  2. Один раз отправляет ваше фото лица в GPT-4o (vision) и получает подробное
     описание внешности. Результат кэшируется в _кэш/профиль_лица.json,
     поэтому повторно фото не анализируется.
  3. Для каждой идеи собирает финальный промпт: внешность + сцена идеи +
     вертикальная документальная подача.
  4. Генерирует вертикальное фото:
       - IMAGE_BACKEND="openai"  -> gpt-image-1 с вашим фото как референсом
                                    (лицо сохраняется лучше всего);
       - IMAGE_BACKEND="fastgen" -> api.fast-gen.ai, как в вашем старом скрипте.
  5. Сохраняет файл в папку ФОТО и переводит идею в статус ФОТО.

Запуск:
    python3 step2_photo.py                 # обработать все новые идеи
    python3 step2_photo.py --limit 3       # только первые 3
    python3 step2_photo.py --idea 7        # только идею №7
    python3 step2_photo.py --backend fastgen

Ключи берутся из окружения, как в старых скриптах.
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

from openai import OpenAI

import config
import fastgen_client
from ideas_store import (
    STATUS_NEW,
    STATUS_PHOTO,
    Idea,
    load_ideas,
    save_ideas,
    stamp_today,
)

FACE_PROFILE_CACHE = "профиль_лица.json"

FACE_SYSTEM = """
Ты — ассистент фотографа. Тебе дают фотографию мужчины.
Опиши его внешность так, чтобы по описанию генератор изображений мог
воспроизвести ЭТОГО ЖЕ человека в другой обстановке.

Опиши: примерный возраст, форму лица, цвет и длину волос, причёску,
растительность на лице, брови, глаза, нос, губы, телосложение,
особые приметы (родинки, шрамы, очки), оттенок кожи.

Не оценивай внешность, не делай выводов о личности, не упоминай национальность.
Пиши по-русски, сплошным текстом, 60-90 слов, только визуальные признаки.

Верни строго JSON: {"внешность": "..."}
""".strip()


PROMPT_SYSTEM = """
Ты собираешь финальный промпт для генератора изображений.

Тебе дают:
  1. Описание внешности конкретного мужчины — он ДОЛЖЕН быть главным героем кадра.
  2. Сцену будущего фото.

Собери один цельный промпт на русском языке, который:
  - начинается с героя: его внешность (вплети описание естественно, не списком);
  - затем что он делает, поза, выражение лица;
  - затем окружение, другие люди, предметы;
  - затем место, свет, ракурс камеры;
  - кадр строго ВЕРТИКАЛЬНЫЙ 9:16, герой хорошо читается, лицо не перекрыто;
  - заканчивается дословно: «Документальная фотография, снято на смартфон,
    реалистичное освещение, естественные пропорции, высокая детализация кожи
    и одежды, без художественной стилизации, без размытия, максимально натурально.»

Никакого текста, надписей, логотипов и водяных знаков внутри картинки.
Объём 90-140 слов. Верни строго JSON: {"промпт": "..."}
""".strip()


def analyze_face(client: OpenAI, face_path: Path, cache_path: Path, force: bool = False) -> str:
    """Описание внешности по фото. Кэшируется, чтобы не гонять vision каждый раз."""
    if cache_path.exists() and not force:
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if cached.get("внешность") and cached.get("файл") == face_path.name:
                print(f"Профиль лица взят из кэша: {cache_path}")
                return cached["внешность"]
        except Exception:
            pass

    if not face_path.exists():
        raise SystemExit(f"Не найдено фото лица: {face_path}")

    print(f"Анализирую фото лица: {face_path.name}")
    encoded = base64.b64encode(face_path.read_bytes()).decode("ascii")
    suffix = face_path.suffix.lower().lstrip(".") or "jpeg"
    mime = "image/jpeg" if suffix in ("jpg", "jpeg") else f"image/{suffix}"

    response = client.chat.completions.create(
        model=config.VISION_MODEL,
        temperature=0.2,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": FACE_SYSTEM},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Опиши внешность этого человека."},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}},
                ],
            },
        ],
    )
    appearance = (json.loads(response.choices[0].message.content or "{}").get("внешность") or "").strip()
    if not appearance:
        raise SystemExit("Не удалось получить описание внешности. Попробуйте запустить ещё раз.")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps({"файл": face_path.name, "внешность": appearance}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Профиль лица сохранён: {cache_path}")
    return appearance


def build_photo_prompt(client: OpenAI, appearance: str, idea: Idea) -> str:
    """Собирает финальный промпт: внешность + сцена идеи."""
    scene = idea.photo_prompt or idea.description or idea.title
    user_prompt = (
        f"ВНЕШНОСТЬ ГЕРОЯ:\n{appearance}\n\n"
        f"СЦЕНА (идея «{idea.title}»):\n{scene}"
    )
    response = client.chat.completions.create(
        model=config.TEXT_MODEL,
        temperature=0.6,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": PROMPT_SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
    )
    prompt = (json.loads(response.choices[0].message.content or "{}").get("промпт") or "").strip()
    return prompt or scene


def generate_with_openai(client: OpenAI, prompt: str, face_path: Path, out_path: Path) -> None:
    """gpt-image-1: генерация с вашим фото как референсом — лучше держит лицо."""
    with face_path.open("rb") as face_file:
        response = client.images.edit(
            model=config.OPENAI_IMAGE_MODEL,
            image=[face_file],
            prompt=(
                "Сохрани лицо человека с приложенной фотографии — это должен быть "
                "тот же самый человек, узнаваемый по чертам лица. "
                f"Помести его в новую сцену:\n\n{prompt}"
            ),
            size=config.OPENAI_IMAGE_SIZE,
        )
    payload = response.data[0]
    raw = base64.b64decode(payload.b64_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(raw)


def process_idea(
    client: OpenAI,
    idea: Idea,
    appearance: str,
    backend: str,
) -> tuple[bool, str]:
    """Возвращает (успех, путь к файлу или текст ошибки)."""
    print(f"\n[{idea.number:03d}] {idea.title}")
    print(f"      надпись: «{idea.caption}»")

    prompt = build_photo_prompt(client, appearance, idea)
    print(f"      промпт: {prompt[:110]}...")

    out_path = config.PHOTOS_DIR / f"{idea.number:03d}_{_safe_name(idea.title)}.png"

    try:
        if backend == "openai":
            generate_with_openai(client, prompt, config.FACE_PHOTO, out_path)
        else:
            fastgen_client.generate_image(prompt, out_path, reference_image=config.FACE_PHOTO)
    except fastgen_client.PermanentError as exc:
        print(f"      ЗАБЛОКИРОВАНО: {exc}")
        return False, str(exc)
    except Exception as exc:  # noqa: BLE001
        print(f"      ОШИБКА: {exc}")
        return False, str(exc)

    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"      сохранено: {out_path.name} ({size_mb:.2f} MB)")
    return True, str(out_path)


def _safe_name(text: str) -> str:
    keep = [c if (c.isalnum() or c in " -_") else "_" for c in text]
    return "".join(keep).strip().replace(" ", "_")[:40] or "idea"


def main() -> None:
    parser = argparse.ArgumentParser(description="Генерация фото по идеям и вашему лицу.")
    parser.add_argument("--limit", type=int, default=0, help="Максимум идей за запуск (0 = все новые).")
    parser.add_argument("--idea", type=int, default=0, help="Обработать только идею с этим номером.")
    parser.add_argument("--backend", choices=["openai", "fastgen"], default=config.IMAGE_BACKEND)
    parser.add_argument("--refresh-face", action="store_true", help="Заново проанализировать фото лица.")
    parser.add_argument("--ideas-file", default=str(config.IDEAS_FILE))
    args = parser.parse_args()

    config.require_openai_key()
    if args.backend == "fastgen":
        config.require_fastgen_key()
    config.ensure_dirs()

    ideas_path = Path(args.ideas_file).expanduser()
    ideas = load_ideas(ideas_path)
    if not ideas:
        raise SystemExit(f"В файле нет идей: {ideas_path}\nСначала запустите: python3 step1_ideas.py")

    if args.idea:
        queue = [i for i in ideas if i.number == args.idea]
        if not queue:
            raise SystemExit(f"Идея №{args.idea} не найдена.")
    else:
        queue = [i for i in ideas if i.status == STATUS_NEW]
        if args.limit:
            queue = queue[: args.limit]

    if not queue:
        print("Новых идей нет — все уже обработаны.")
        print("Добавить новые: python3 step1_ideas.py")
        return

    print(f"Бэкенд генерации: {args.backend}")
    print(f"К обработке идей: {len(queue)}")

    client = OpenAI(api_key=config.OPENAI_API_KEY)
    appearance = analyze_face(
        client, config.FACE_PHOTO, config.CACHE_DIR / FACE_PROFILE_CACHE, force=args.refresh_face
    )

    done = 0
    for idea in queue:
        ok, result = process_idea(client, idea, appearance, args.backend)
        if ok:
            idea.photo_file = result
            idea.status = STATUS_PHOTO
            idea.date = idea.date or stamp_today()
            done += 1
            save_ideas(ideas_path, ideas, make_backup=False)

    print(f"\nГотово. Фото сгенерировано: {done} из {len(queue)}")
    print(f"Папка: {config.PHOTOS_DIR}")
    if done:
        print("Дальше: python3 step3_animate.py")


if __name__ == "__main__":
    main()
