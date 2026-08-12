#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
СКРИПТ 1 — АНАЛИЗ ИДЕЙ И ГЕНЕРАЦИЯ НОВЫХ.

Что делает:
  1. Читает ИДЕИ.txt.
  2. Смотрит на поле ПРОСМОТРЫ у уже снятых роликов.
     Если просмотры проставлены — раскладывает идеи на «зашедшие» и «слабые»
     и просит GPT вытащить конкретные сработавшие паттерны:
     какой типаж сцены, какая механика надписи, какая эмоция.
  3. Придумывает новые идеи ПО СРАБОТАВШИМ ПАТТЕРНАМ (если статистики нет —
     работает от базового набора приёмов) и дописывает их в ИДЕИ.txt.
  4. Для каждой идеи сразу придумывает НАДПИСЬ — текст-крючок на первые 2 секунды.
  5. Пишет отчёт по аналитике в _кэш/аналитика.txt

Запуск:
    python3 step1_ideas.py                # добавить 5 новых идей
    python3 step1_ideas.py --count 10     # добавить 10
    python3 step1_ideas.py --only-analyze # только показать аналитику, ничего не добавлять

Ключ берётся из окружения — так же, как в старых скриптах:
    export OPENAI_API_KEY='sk-...'
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from openai import OpenAI

import config
from ideas_store import (
    STATUS_DONE,
    STATUS_NEW,
    Idea,
    load_ideas,
    next_number,
    save_ideas,
)

# =============================================================================
# ПРОМПТЫ
# =============================================================================

ANALYST_SYSTEM = """
Ты — аналитик коротких вертикальных видео (Reels/TikTok/Shorts) для Telegram-бота,
который превращает обычное фото человека в реалистичное фото в необычной ситуации.

Тебе дают список уже снятых роликов с количеством просмотров.
Твоя задача — понять, ЧТО ИМЕННО сработало, а что нет.

Анализируй по осям:
  - тип сцены (криминал/суд, роскошь, слава, спорт, опасность, ностальгия, абсурд);
  - механика надписи (интрига, соцдоказательство, вызов, признание, недосказанность);
  - эмоция зрителя (зависть, недоверие, шок, узнавание, смех);
  - «правдоподобность» — насколько кадр похож на реальный документальный снимок.

Верни строго JSON:
{
  "работает": ["3-6 конкретных паттернов, которые дали высокие просмотры"],
  "не_работает": ["2-4 паттерна, которые дали низкие просмотры"],
  "вывод": "1-2 предложения: в какую сторону двигаться дальше"
}

Если данных мало — честно скажи об этом в поле "вывод", но паттерны всё равно
предположи, опираясь на то, что видно.
""".strip()


IDEAS_SYSTEM = """
Ты — креативный продюсер вирусных вертикальных роликов для Telegram-бота,
который по фото человека генерирует реалистичное фото в необычной ситуации.

Формат ролика, для которого ты придумываешь идеи:
  [1] 2 секунды — оживлённое ИИ-фото человека в необычной ситуации + надпись-крючок
  [2] 3 секунды — постоянный скринкаст: как пользоваться ботом
  [3] 2 секунды — итоговое фото

Значит идея должна «читаться» за 2 секунды без звука и вызывать желание
досмотреть и повторить это со своим фото.

ТРЕБОВАНИЯ К ИДЕЕ:
  - Сцена должна быть СНИМАЕМОЙ на смартфон и выглядеть как реальный
    документальный кадр, а не как рисунок или фантастика.
  - Человек в кадре — обычный мужчина, лицо всегда одно и то же (лицо владельца).
  - Ситуация должна быть неожиданной, статусной или щекочущей самолюбие:
    такой, чтобы зритель захотел показать её друзьям.
  - Никакого криминала против реальных людей, никаких реальных брендов
    в уничижительном контексте, никаких политических лозунгов, никакой эротики.
  - Идеи в партии должны быть РАЗНЫМИ по типу сцены: не пять вариантов роскоши,
    а разброс по разным эмоциям и мирам.

НАДПИСЬ — самое важное. Правила:
  - на русском языке, 3-8 слов, максимум 2 строки;
  - разговорная, как будто пишет живой человек, а не реклама;
  - не описывает картинку буквально, а создаёт интригу или социальное доказательство;
  - без хэштегов, без эмодзи, без кавычек, без точки в конце.

ПРОМПТ_ФОТО пиши по-русски, подробно (60-110 слов), по структуре:
  кто в кадре и во что одет -> что делает и какая поза -> кто и что вокруг ->
  где происходит -> свет и ракурс -> и обязательно в конце дословно:
  «Вертикальный кадр 9:16. Документальная фотография, снято на смартфон,
  реалистичное освещение, естественные пропорции, высокая детализация кожи и одежды,
  без художественной стилизации, без размытия, максимально натурально.»

Верни строго JSON:
{
  "идеи": [
    {
      "название": "2-4 слова",
      "надпись": "текст-крючок",
      "описание": "одно предложение, что происходит в кадре",
      "промпт_фото": "подробный промпт по структуре выше"
    }
  ]
}
""".strip()


def build_analysis_payload(ideas: list[Idea]) -> list[dict]:
    """Собирает данные по роликам, у которых проставлены просмотры."""
    rows = []
    for idea in ideas:
        views = idea.views_int
        if views is None:
            continue
        rows.append({
            "название": idea.title,
            "надпись": idea.caption,
            "описание": idea.description,
            "просмотры": views,
        })
    return sorted(rows, key=lambda r: r["просмотры"], reverse=True)


def analyze_performance(client: OpenAI, rows: list[dict]) -> dict:
    """Просит GPT вытащить сработавшие паттерны из статистики просмотров."""
    if not rows:
        return {}

    user_prompt = (
        "Вот снятые ролики с просмотрами, отсортированы по убыванию.\n"
        "Найди, что общего у верхних и чего не хватает нижним.\n\n"
        + json.dumps(rows, ensure_ascii=False, indent=2)
    )
    try:
        response = client.chat.completions.create(
            model=config.TEXT_MODEL,
            temperature=0.3,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": ANALYST_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
        )
        return json.loads(response.choices[0].message.content or "{}")
    except Exception as exc:  # noqa: BLE001 — без аналитики генерация всё равно идёт
        print(f"[WARN] Анализ просмотров не удался ({exc}). Работаю без статистики.")
        return {}


def format_analysis(analysis: dict) -> str:
    if not analysis:
        return (
            "Статистики пока нет — ни у одной идеи не заполнено поле ПРОСМОТРЫ.\n"
            "Придумываю разноплановые идеи, чтобы набрать первые данные."
        )
    works = analysis.get("работает") or []
    fails = analysis.get("не_работает") or []
    verdict = analysis.get("вывод", "")

    lines = []
    if works:
        lines.append("СРАБОТАЛО (усиливать):")
        lines += [f"  + {item}" for item in works]
    if fails:
        lines.append("НЕ СРАБОТАЛО (избегать):")
        lines += [f"  - {item}" for item in fails]
    if verdict:
        lines.append(f"ВЫВОД: {verdict}")
    return "\n".join(lines)


def generate_ideas(
    client: OpenAI,
    count: int,
    analysis: dict,
    existing: list[Idea],
) -> list[dict]:
    """Генерирует новые идеи с учётом аналитики и уже существующих идей."""
    used = [
        {"название": i.title, "надпись": i.caption}
        for i in existing
        if i.title or i.caption
    ]

    parts = [f"Придумай ровно {count} новых идей."]

    if analysis:
        parts.append(
            "АНАЛИТИКА ПО УЖЕ СНЯТЫМ РОЛИКАМ — обязательно опирайся на неё:\n"
            + json.dumps(analysis, ensure_ascii=False, indent=2)
            + "\n\nБольшинство новых идей должны развивать то, что в списке «работает», "
            "но 1-2 идеи сделай экспериментальными в новом направлении, "
            "чтобы находить новые рабочие приёмы."
        )
    else:
        parts.append(
            "Статистики просмотров пока нет. Сделай партию максимально РАЗНОПЛАНОВОЙ: "
            "разные миры и разные эмоции, чтобы по итогам замеров стало видно, "
            "какое направление выстреливает."
        )

    if used:
        parts.append(
            "УЖЕ ИСПОЛЬЗОВАННЫЕ идеи и надписи — не повторяй их и не делай их пересказ:\n"
            + json.dumps(used, ensure_ascii=False, indent=2)
        )

    response = client.chat.completions.create(
        model=config.TEXT_MODEL,
        temperature=0.95,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": IDEAS_SYSTEM},
            {"role": "user", "content": "\n\n".join(parts)},
        ],
    )
    data = json.loads(response.choices[0].message.content or "{}")
    return data.get("идеи", []) or []


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Анализ ИДЕИ.txt по просмотрам и генерация новых идей."
    )
    parser.add_argument("--count", type=int, default=5, help="Сколько новых идей добавить (по умолчанию 5).")
    parser.add_argument("--only-analyze", action="store_true", help="Только аналитика, без добавления идей.")
    parser.add_argument("--ideas-file", default=str(config.IDEAS_FILE), help="Путь к ИДЕИ.txt")
    args = parser.parse_args()

    config.require_openai_key()
    config.ensure_dirs()

    ideas_path = Path(args.ideas_file).expanduser()
    ideas = load_ideas(ideas_path)

    print(f"Файл идей: {ideas_path}")
    print(f"Всего идей в файле: {len(ideas)}")
    by_status: dict[str, int] = {}
    for idea in ideas:
        by_status[idea.status] = by_status.get(idea.status, 0) + 1
    if by_status:
        print("По статусам: " + ", ".join(f"{k}={v}" for k, v in by_status.items()))

    client = OpenAI(api_key=config.OPENAI_API_KEY)

    # --- Аналитика по просмотрам ---
    rows = build_analysis_payload(ideas)
    print(f"\nРоликов с заполненными просмотрами: {len(rows)}")
    if rows:
        print("Топ по просмотрам:")
        for row in rows[:5]:
            print(f"  {row['просмотры']:>9,} — {row['название']} | «{row['надпись']}»".replace(",", " "))

    analysis = analyze_performance(client, rows) if rows else {}
    report = format_analysis(analysis)
    print("\n" + report)

    analytics_path = config.CACHE_DIR / "аналитика.txt"
    analytics_path.parent.mkdir(parents=True, exist_ok=True)
    analytics_path.write_text(
        f"Роликов со статистикой: {len(rows)}\n\n{report}\n",
        encoding="utf-8",
    )

    if args.only_analyze:
        print(f"\nОтчёт сохранён: {analytics_path}")
        return

    # --- Генерация новых идей ---
    print(f"\nГенерирую {args.count} новых идей...")
    raw_ideas = generate_ideas(client, args.count, analysis, ideas)
    if not raw_ideas:
        print("GPT не вернул ни одной идеи. Попробуйте запустить ещё раз.")
        return

    number = next_number(ideas)
    added: list[Idea] = []
    for raw in raw_ideas[: args.count]:
        title = (raw.get("название") or "").strip()
        caption = (raw.get("надпись") or "").strip().strip('"«»')
        if not title or not caption:
            continue
        added.append(Idea(
            number=number,
            status=STATUS_NEW,
            title=title,
            caption=caption,
            description=(raw.get("описание") or "").strip(),
            photo_prompt=(raw.get("промпт_фото") or "").strip(),
            views="",
        ))
        number += 1

    save_ideas(ideas_path, ideas + added)

    print(f"\nДобавлено идей: {len(added)}")
    for idea in added:
        print(f"  [{idea.number:03d}] {idea.title} — «{idea.caption}»")
    print(f"\nФайл обновлён: {ideas_path}")
    print("Дальше: python3 step2_photo.py")


if __name__ == "__main__":
    main()
