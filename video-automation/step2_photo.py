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
    python3 step2_photo.py --backend openai   # запасной вариант через gpt-image-1

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
import image_utils
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
  - начинается с героя: если внешность описана — вплети её естественно, не списком;
    если сказано брать внешность с фото — пиши просто «мужчина с фото»
    и НЕ придумывай никаких черт лица, возраста, причёски и бороды;
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


def build_photo_prompt(client: OpenAI, appearance: str, idea: Idea, with_appearance: bool = True) -> str:
    """Собирает финальный промпт по сцене идеи.

    with_appearance=False используется, когда лицо передаётся картинкой-референсом:
    словесное описание внешности в этом случае ВРЕДИТ — модель начинает рисовать
    «мужчину, подходящего под описание», вместо конкретного человека с фото.
    """
    scene = idea.photo_prompt or idea.description or idea.title
    if with_appearance:
        user_prompt = (
            f"ВНЕШНОСТЬ ГЕРОЯ:\n{appearance}\n\n"
            f"СЦЕНА (идея «{idea.title}»):\n{scene}"
        )
    else:
        user_prompt = (
            "ВНЕШНОСТЬ ГЕРОЯ: берётся с приложенного фото, описывать её словами НЕ НУЖНО. "
            "Называй его просто «мужчина с фото».\n\n"
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


def identity_instruction(reference_names: list[str] | None = None) -> str:
    """Инструкция сохранения личности.

    Если референсы переданы именованными, промпт ссылается на них по именам
    файлов — так модель понимает, что на всех снимках ОДИН И ТОТ ЖЕ человек,
    и держит лицо заметно точнее, чем от общей фразы «на фото».
    """
    if reference_names:
        names = ", ".join(reference_names)
        head = (
            f"На изображениях {names} — ОДИН И ТОТ ЖЕ конкретный реальный мужчина, "
            f"снятый с разных ракурсов. Сгенерируй в новой сцене ИМЕННО ЕГО, "
            f"а не похожего на него человека. Его лицо с {names} перенеси "
            f"без изменений, черта в черту.\n\n"
        )
    else:
        head = (
            "На приложенных фотографиях — конкретный реальный мужчина. "
            "В результате должен быть ИМЕННО ОН, а не похожий на него человек.\n\n"
        )
    return head + IDENTITY_RULES


IDENTITY_RULES = (
    "СТРОГО СОХРАНИ БЕЗ ИЗМЕНЕНИЙ: форму и пропорции лица, форму и посадку глаз, "
    "форму бровей, форму носа, форму губ, линию челюсти и подбородка, форму ушей, "
    "линию роста волос, длину и форму бороды и усов, оттенок кожи, родинки и "
    "любые особые приметы. Возраст оставь тот же.\n\n"
    "НЕЛЬЗЯ: делать лицо моложе, стройнее, симметричнее или «красивее», "
    "менять форму носа и губ, убирать морщины и родинки, менять причёску и бороду, "
    "делать глянцевую ретушь кожи. Кожа должна остаться с реальной текстурой — "
    "порами, неровностями и естественным блеском.\n\n"
    "Меняй только одежду, позу, окружение и освещение — по описанию сцены ниже.\n\n"
    "СЦЕНА:\n"
)


def collect_face_photos() -> list[Path]:
    """Все доступные фото лица.

    Если рядом есть папка ЛИЦО — берём оттуда до 4 снимков: чем больше ракурсов,
    тем точнее модель держит внешность. Иначе используем одиночное фото из конфига.
    """
    photos: list[Path] = []
    if config.FACE_DIR.exists():
        photos = sorted(
            p for p in config.FACE_DIR.iterdir()
            if p.is_file()
            and p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")
            and not p.name.startswith(".")
        )[:4]
    if not photos and config.FACE_PHOTO.exists():
        photos = [config.FACE_PHOTO]
    return photos


def generate_with_openai(client: OpenAI, prompt: str, face_photos: list[Path], out_path: Path) -> None:
    """gpt-image-1 с вашими фото как референсом.

    input_fidelity="high" — ключевой параметр: именно он заставляет модель
    держать черты лица с исходника, а не рисовать «похожего человека».
    Если версия библиотеки или модель его не понимает — повторяем без него.
    """
    # От самого качественного набора параметров к самому совместимому
    variants = [
        {"input_fidelity": "high", "quality": "high"},
        {"quality": "high"},
        {},
    ]

    handles = [p.open("rb") for p in face_photos]
    try:
        response = None
        last_error: Exception | None = None

        for index, extra in enumerate(variants):
            for handle in handles:
                handle.seek(0)
            try:
                response = client.images.edit(
                    model=config.OPENAI_IMAGE_MODEL,
                    image=handles,
                    prompt=identity_instruction() + prompt,
                    size=config.OPENAI_IMAGE_SIZE,
                    **extra,
                )
                break
            except TypeError as exc:
                # Старая версия библиотеки не знает такой параметр
                last_error = exc
            except Exception as exc:  # noqa: BLE001
                message = str(exc).lower()
                unsupported = any(
                    marker in message
                    for marker in ("unknown", "unsupported", "unexpected", "not permitted")
                )
                # Настоящую ошибку (нет доступа, кончились деньги) не маскируем
                if not (unsupported and index < len(variants) - 1):
                    raise
                last_error = exc

        if response is None:
            raise RuntimeError(f"gpt-image-1 не принял запрос: {last_error}")
    finally:
        for handle in handles:
            handle.close()

    raw = base64.b64decode(response.data[0].b64_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(raw)


def generate_with_flow(prompt: str, face_photos: list[Path], out_path: Path) -> None:
    """Nano Banana Pro через api.fast-gen.ai.

    Референсы уходят ИМЕНОВАННЫМИ, и промпт ссылается на них по именам файлов —
    так модель понимает, что это один и тот же человек. Формат сразу 9:16,
    поэтому кадр не придётся растягивать при оживлении.
    """
    names = [
        fastgen_client.reference_filename(i, p)
        for i, p in enumerate(face_photos, 1)
    ]
    fastgen_client.generate_image(
        identity_instruction(names) + prompt,
        out_path,
        reference_images=face_photos,
        aspect_ratio=config.IMAGE_ASPECT_RATIO,
    )


def process_idea(
    client: OpenAI,
    idea: Idea,
    appearance: str,
    backend: str,
    face_photos: list[Path],
) -> tuple[bool, str]:
    """Возвращает (успех, путь к файлу или текст ошибки)."""
    print(f"\n[{idea.number:03d}] {idea.title}")
    print(f"      надпись: «{idea.caption}»")

    # При работе с картинкой-референсом словесное описание лица только мешает
    prompt = build_photo_prompt(client, appearance, idea, with_appearance=False)
    print(f"      промпт: {prompt[:110]}...")

    out_path = config.PHOTOS_DIR / f"{idea.number:03d}_{_safe_name(idea.title)}.png"

    try:
        if backend == "openai":
            generate_with_openai(client, prompt, face_photos, out_path)
        else:
            generate_with_flow(prompt, face_photos, out_path)
    except fastgen_client.PermanentError as exc:
        print(f"      ЗАБЛОКИРОВАНО: {exc}")
        return False, str(exc)
    except Exception as exc:  # noqa: BLE001
        print(f"      ОШИБКА: {exc}")
        return False, str(exc)

    # Приводим к 9:16: иначе провайдер растянет кадр при оживлении и лицо поплывёт
    if image_utils.ensure_vertical(out_path):
        print("      кадр обрезан до 9:16, чтобы лицо не растянулось при оживлении")

    size_mb = out_path.stat().st_size / 1024 / 1024
    aspect = image_utils.aspect_of(out_path)
    ratio = f", {aspect:.3f} (нужно 0.563)" if aspect else ""
    print(f"      сохранено: {out_path.name} ({size_mb:.2f} MB{ratio})")
    return True, str(out_path)


def _safe_name(text: str) -> str:
    keep = [c if (c.isalnum() or c in " -_") else "_" for c in text]
    return "".join(keep).strip().replace(" ", "_")[:40] or "idea"


def main() -> None:
    parser = argparse.ArgumentParser(description="Генерация фото по идеям и вашему лицу.")
    parser.add_argument("--limit", type=int, default=0, help="Максимум идей за запуск (0 = все новые).")
    parser.add_argument("--idea", type=int, default=0, help="Обработать только идею с этим номером.")
    parser.add_argument("--backend", choices=["flow", "openai"], default=config.IMAGE_BACKEND)
    parser.add_argument("--refresh-face", action="store_true", help="Заново проанализировать фото лица.")
    parser.add_argument("--ideas-file", default=str(config.IDEAS_FILE))
    args = parser.parse_args()

    config.require_openai_key()
    if args.backend != "openai":
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

    face_photos = collect_face_photos()
    if not face_photos:
        raise SystemExit(
            f"Не найдено ни одного фото лица.\n"
            f"Положите фото сюда: {config.FACE_PHOTO}\n"
            f"или несколько снимков в папку: {config.FACE_DIR}"
        )

    print(f"Бэкенд генерации: {args.backend}")
    print(f"Фото лица: {len(face_photos)} шт. ({', '.join(p.name for p in face_photos)})")
    if len(face_photos) == 1:
        print(
            "  Подсказка: чтобы лицо получалось точнее, положите 3-4 своих фото\n"
            f"  (анфас, полуоборот, разный свет) в папку {config.FACE_DIR.name}"
        )
    print(f"К обработке идей: {len(queue)}")

    client = OpenAI(api_key=config.OPENAI_API_KEY)
    try:
        appearance = analyze_face(
            client, face_photos[0], config.CACHE_DIR / FACE_PROFILE_CACHE, force=args.refresh_face
        )
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        from step1_ideas import explain_openai_error  # noqa: PLC0415

        raise SystemExit(explain_openai_error(exc)) from None

    done = 0
    for idea in queue:
        ok, result = process_idea(client, idea, appearance, args.backend, face_photos)
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
