#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
СКРИПТ 5 — ОПИСАНИЯ ПОД РОЛИКИ.

Для каждой снятой идеи собирает готовый текст описания к посту:

    Как сделать такое фото 👇
    1. Переходим в бота по ссылке в описании
    2. Заходим в свободный режим
    3. Отправляем своё фото 📸
    4. Прикладываем промпт 👇

    Промпт:

    Фотореалистичный вертикальный кадр в коридоре российского суда...

Промпт в описании — ПУБЛИЧНЫЙ: он описывает сцену так, чтобы любой человек
подставил своё лицо и получил такой же кадр. Внутренние инструкции по
сохранению вашей внешности туда не попадают.

Каждое описание сохраняется отдельным файлом в папку ОПИСАНИЯ — открыл,
выделил всё, скопировал. Путь к файлу записывается в ИДЕИ.txt.

Запуск:
    python3 step5_description.py              # для всех снятых роликов
    python3 step5_description.py --idea 7
    python3 step5_description.py --all        # включая ещё не снятые
    python3 step5_description.py --print      # ещё и вывести в консоль
    python3 step5_description.py --force      # перегенерировать существующие
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from openai import OpenAI

import config
from ideas_store import STATUS_DONE, Idea, load_ideas, save_ideas

# =============================================================================
# ШАБЛОН ОПИСАНИЯ — правьте здесь, если захотите поменять формулировки
# =============================================================================

HEADER = "Как сделать такое фото 👇"

STEPS = [
    "Переходим в бота по ссылке в описании",
    "Заходим в свободный режим",
    "Отправляем своё фото 📸",
    "Прикладываем промпт 👇",
]

PROMPT_LABEL = "Промпт:"


def build_post(prompt: str) -> str:
    """Собирает финальный текст описания по шаблону."""
    lines = [HEADER]
    lines += [f"{i}. {step}" for i, step in enumerate(STEPS, 1)]
    lines += ["", PROMPT_LABEL, "", prompt.strip()]
    return "\n".join(lines) + "\n"


# =============================================================================
# ПУБЛИЧНЫЙ ПРОМПТ
# =============================================================================

PUBLIC_PROMPT_SYSTEM = """
Ты переписываешь описание сцены в ПУБЛИЧНЫЙ промпт для генератора изображений.
Этот промпт человек скопирует в бота вместе со СВОИМ фото, чтобы получить
такой же кадр, но со своим лицом.

СТРУКТУРА — ровно четыре абзаца, разделённых пустой строкой:

Абзац 1. Начни словами «Фотореалистичный вертикальный кадр» + где происходит.
  Затем главный герой: «По центру молодой мужчина» + обобщённая внешность
  (короткие тёмные волосы, аккуратная борода) + во что одет.
  Затем что он делает, поза, выражение лица.

Абзац 2. Кто и что вокруг него: другие люди, их одежда и действия, что попадает
  в кадр на переднем плане, какой эффект это создаёт.

Абзац 3. Место целиком: помещение или улица, стены, пол, потолок, мебель,
  таблички, техника, машины — конкретные предметы, по которым место узнаётся.

Абзац 4. Композиция и техническая часть. Заканчивай ДОСЛОВНО так:
  «Реалистичное освещение, естественные пропорции людей, высокая детализация
  формы, кожи и одежды, документальная фотография, снято на смартфон, широкий
  угол, без художественной стилизации, без размытия, максимально натурально.»

ЖЁСТКИЕ ПРАВИЛА:
  - пиши по-русски, живым описательным языком;
  - внешность героя описывай ОБОБЩЁННО, чтобы подошла любому мужчине;
    никаких имён, никаких ссылок на приложенные файлы и референсы;
  - не пиши служебных инструкций вроде «сохрани лицо с фото», «не меняй черты» —
    это внутренние указания, в публичный промпт они не идут;
  - не упоминай ИИ, нейросети, бота и генерацию;
  - никаких заголовков, нумерации и маркеров списка — только четыре абзаца;
  - объём 130-200 слов.

Верни строго JSON: {"промпт": "текст из четырёх абзацев"}
""".strip()


def build_public_prompt(client: OpenAI, idea: Idea) -> str:
    scene = idea.photo_prompt or idea.description or idea.title
    response = client.chat.completions.create(
        model=config.TEXT_MODEL,
        temperature=0.4,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": PUBLIC_PROMPT_SYSTEM},
            {
                "role": "user",
                "content": f"СЦЕНА (идея «{idea.title}»):\n{scene}",
            },
        ],
    )
    prompt = (json.loads(response.choices[0].message.content or "{}").get("промпт") or "").strip()
    return prompt or scene


# =============================================================================

def _safe_name(text: str) -> str:
    keep = [c if (c.isalnum() or c in " -_") else "_" for c in text]
    return "".join(keep).strip().replace(" ", "_")[:40] or "idea"


def main() -> None:
    parser = argparse.ArgumentParser(description="Генерация описаний под снятые ролики.")
    parser.add_argument("--idea", type=int, default=0, help="Только идея с этим номером.")
    parser.add_argument("--all", action="store_true", help="Все идеи, а не только снятые.")
    parser.add_argument("--force", action="store_true", help="Перегенерировать уже готовые описания.")
    parser.add_argument("--print", dest="show", action="store_true", help="Вывести текст в консоль.")
    parser.add_argument("--ideas-file", default=str(config.IDEAS_FILE))
    args = parser.parse_args()

    config.require_openai_key()
    config.ensure_dirs()
    config.DESCRIPTIONS_DIR.mkdir(parents=True, exist_ok=True)

    ideas_path = Path(args.ideas_file).expanduser()
    ideas = load_ideas(ideas_path)
    if not ideas:
        raise SystemExit(f"В файле нет идей: {ideas_path}")

    if args.idea:
        queue = [i for i in ideas if i.number == args.idea]
        if not queue:
            raise SystemExit(f"Идея №{args.idea} не найдена.")
    elif args.all:
        queue = [i for i in ideas if i.photo_prompt or i.description]
    else:
        queue = [i for i in ideas if i.status == STATUS_DONE]

    if not queue:
        print("Нет снятых роликов. Сначала смонтируйте: python3 step4_montage.py")
        print("Либо запустите с --all, чтобы сделать описания для всех идей.")
        return

    if not args.force:
        queue = [
            i for i in queue
            if not (i.post_file and Path(i.post_file).expanduser().exists())
        ]
        if not queue:
            print("Описания уже готовы для всех роликов. Перегенерировать: --force")
            return

    print(f"Готовлю описаний: {len(queue)}")
    client = OpenAI(api_key=config.OPENAI_API_KEY)

    done = 0
    for idea in queue:
        print(f"\n[{idea.number:03d}] {idea.title}")
        try:
            prompt = build_public_prompt(client, idea)
        except Exception as exc:  # noqa: BLE001
            from step1_ideas import explain_openai_error  # noqa: PLC0415

            print(f"      ОШИБКА: {explain_openai_error(exc)}")
            continue

        text = build_post(prompt)
        out_path = config.DESCRIPTIONS_DIR / f"{idea.number:03d}_{_safe_name(idea.title)}.txt"
        out_path.write_text(text, encoding="utf-8")

        idea.post_file = str(out_path)
        save_ideas(ideas_path, ideas, make_backup=False)
        done += 1

        print(f"      сохранено: {out_path.name} ({len(text)} символов)")
        if args.show:
            print("\n" + "-" * 60)
            print(text)
            print("-" * 60)

    print(f"\nГотово: {done} из {len(queue)}")
    print(f"Папка: {config.DESCRIPTIONS_DIR}")
    print("Откройте нужный файл, выделите всё и скопируйте в описание к ролику.")


if __name__ == "__main__":
    main()
