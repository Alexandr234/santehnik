#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Чтение и запись файла ИДЕИ.txt.

ФОРМАТ ФАЙЛА (читаемый глазами, можно править руками в любом текстовом редакторе):

    ═══ ИДЕЯ 001 ═══
    СТАТУС: НОВАЯ
    НАЗВАНИЕ: Задержание в суде
    НАДПИСЬ: Как сделать фото, которое разорвет директ
    ОПИСАНИЕ: Мужчину в наручниках ведут по коридору суда двое полицейских.
    ПРОМПТ_ФОТО: Вертикальный кадр. Мужчина с короткими тёмными волосами...
    ПРОСМОТРЫ:
    ФОТО_ФАЙЛ:
    ВИДЕО_ФАЙЛ:
    РОЛИК_ФАЙЛ:
    ДАТА:

Правила:
  * Порядок полей не важен, регистр названий полей не важен.
  * ПРОСМОТРЫ вы заполняете руками после публикации: `ПРОСМОТРЫ: 45000`.
    Скрипт 1 читает это число и учится на том, что зашло.
  * СТАТУС меняют сами скрипты, руками трогать не нужно:
        НОВАЯ   — идея придумана, ничего ещё не сгенерировано
        ФОТО    — фото сгенерировано (скрипт 2)
        ВИДЕО   — фото оживлено (скрипт 3)
        СНЯТА   — ролик смонтирован, идея отработана (скрипт 4)
  * Многострочные значения поддерживаются: продолжение строки пишется
    с отступом (пробел или таб в начале).
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# Статусы жизненного цикла идеи
STATUS_NEW = "НОВАЯ"
STATUS_PHOTO = "ФОТО"
STATUS_VIDEO = "ВИДЕО"
STATUS_DONE = "СНЯТА"
STATUS_REJECTED = "ОТКЛОНЕНА"   # идея не годится, в работу не берётся

STATUS_ORDER = [STATUS_NEW, STATUS_PHOTO, STATUS_VIDEO, STATUS_DONE, STATUS_REJECTED]

# Поля в том порядке, в котором они пишутся в файл
FIELD_ORDER = [
    "СТАТУС",
    "НАЗВАНИЕ",
    "НАДПИСЬ",
    "ОПИСАНИЕ",
    "ПРОМПТ_ФОТО",
    "ПРОСМОТРЫ",
    "ФОТО_ФАЙЛ",
    "ВИДЕО_ФАЙЛ",
    "РОЛИК_ФАЙЛ",
    "ДАТА",
]

HEADER_RE = re.compile(r"^\s*[═=\-]{3}\s*ИДЕЯ\s+(\d+)\s*[═=\-]{3}\s*$", re.IGNORECASE)
FIELD_RE = re.compile(r"^([A-ZА-ЯЁ_]+)\s*:\s*(.*)$", re.IGNORECASE)


def _header(number: int) -> str:
    return f"═══ ИДЕЯ {number:03d} ═══"


@dataclass
class Idea:
    number: int
    status: str = STATUS_NEW
    title: str = ""
    caption: str = ""          # НАДПИСЬ — текст поверх видео
    description: str = ""
    photo_prompt: str = ""
    views: str = ""            # ПРОСМОТРЫ — заполняется вручную
    photo_file: str = ""
    video_file: str = ""
    reel_file: str = ""
    date: str = ""
    extra: dict[str, str] = field(default_factory=dict)

    @property
    def views_int(self) -> int | None:
        """ПРОСМОТРЫ как число. Понимает '45000', '45 000', '45k', '1.2M', '45тыс'."""
        raw = (self.views or "").strip().lower().replace(" ", "").replace(" ", "")
        if not raw:
            return None
        raw = raw.replace(",", ".")
        multiplier = 1
        for suffix, mult in (
            ("млн", 1_000_000), ("m", 1_000_000), ("м", 1_000_000),
            ("тыс", 1_000), ("k", 1_000), ("к", 1_000),
        ):
            if raw.endswith(suffix):
                raw = raw[: -len(suffix)]
                multiplier = mult
                break
        raw = re.sub(r"[^\d.]", "", raw)
        if not raw:
            return None
        try:
            return int(float(raw) * multiplier)
        except ValueError:
            return None

    def to_text(self) -> str:
        values = {
            "СТАТУС": self.status,
            "НАЗВАНИЕ": self.title,
            "НАДПИСЬ": self.caption,
            "ОПИСАНИЕ": self.description,
            "ПРОМПТ_ФОТО": self.photo_prompt,
            "ПРОСМОТРЫ": self.views,
            "ФОТО_ФАЙЛ": self.photo_file,
            "ВИДЕО_ФАЙЛ": self.video_file,
            "РОЛИК_ФАЙЛ": self.reel_file,
            "ДАТА": self.date,
        }
        values.update(self.extra)

        lines = [_header(self.number)]
        for key in FIELD_ORDER + [k for k in values if k not in FIELD_ORDER]:
            value = (values.get(key) or "").strip()
            # Многострочные значения пишем с отступом со второй строки
            parts = value.split("\n")
            lines.append(f"{key}: {parts[0]}")
            for part in parts[1:]:
                lines.append(f"    {part}")
        return "\n".join(lines)


def _flush_field(idea_fields: dict[str, list[str]], key: str | None, buffer: list[str]) -> None:
    if key is not None:
        idea_fields[key] = buffer[:]


def _build_idea(number: int, fields: dict[str, list[str]]) -> Idea:
    def get(name: str) -> str:
        return "\n".join(fields.get(name, [])).strip()

    known = {
        "СТАТУС", "НАЗВАНИЕ", "НАДПИСЬ", "ОПИСАНИЕ", "ПРОМПТ_ФОТО",
        "ПРОСМОТРЫ", "ФОТО_ФАЙЛ", "ВИДЕО_ФАЙЛ", "РОЛИК_ФАЙЛ", "ДАТА",
    }
    extra = {k: "\n".join(v).strip() for k, v in fields.items() if k not in known}

    status = get("СТАТУС").upper() or STATUS_NEW
    if status not in STATUS_ORDER:
        status = STATUS_NEW

    return Idea(
        number=number,
        status=status,
        title=get("НАЗВАНИЕ"),
        caption=get("НАДПИСЬ"),
        description=get("ОПИСАНИЕ"),
        photo_prompt=get("ПРОМПТ_ФОТО"),
        views=get("ПРОСМОТРЫ"),
        photo_file=get("ФОТО_ФАЙЛ"),
        video_file=get("ВИДЕО_ФАЙЛ"),
        reel_file=get("РОЛИК_ФАЙЛ"),
        date=get("ДАТА"),
        extra=extra,
    )


def load_ideas(path: Path) -> list[Idea]:
    """Читает ИДЕИ.txt. Если файла нет — возвращает пустой список."""
    if not path.exists():
        return []

    ideas: list[Idea] = []
    current_number: int | None = None
    current_fields: dict[str, list[str]] = {}
    current_key: str | None = None
    buffer: list[str] = []

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        header_match = HEADER_RE.match(raw_line)
        if header_match:
            _flush_field(current_fields, current_key, buffer)
            if current_number is not None:
                ideas.append(_build_idea(current_number, current_fields))
            current_number = int(header_match.group(1))
            current_fields = {}
            current_key = None
            buffer = []
            continue

        if current_number is None:
            continue  # шапка файла до первой идеи — пропускаем

        # Продолжение многострочного значения: строка с отступом
        if raw_line[:1] in (" ", "\t") and current_key is not None:
            buffer.append(raw_line.strip())
            continue

        field_match = FIELD_RE.match(raw_line)
        if field_match:
            _flush_field(current_fields, current_key, buffer)
            current_key = field_match.group(1).upper()
            buffer = [field_match.group(2).strip()]
            continue

        if not raw_line.strip():
            continue
        if current_key is not None:
            buffer.append(raw_line.strip())

    _flush_field(current_fields, current_key, buffer)
    if current_number is not None:
        ideas.append(_build_idea(current_number, current_fields))

    return ideas


def save_ideas(path: Path, ideas: list[Idea], make_backup: bool = True) -> None:
    """Перезаписывает ИДЕИ.txt. Перед записью делает резервную копию."""
    path.parent.mkdir(parents=True, exist_ok=True)

    if make_backup and path.exists():
        backup_dir = path.parent / "_кэш" / "бэкапы_идей"
        backup_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        shutil.copyfile(path, backup_dir / f"ИДЕИ_{stamp}.txt")

    head = [
        "# ИДЕИ ДЛЯ РОЛИКОВ",
        "#",
        "# После публикации ролика впишите число в поле ПРОСМОТРЫ — например: ПРОСМОТРЫ: 45000",
        "# Скрипт 1 читает эти числа, находит сработавшие приёмы и придумывает новые идеи по ним.",
        "#",
        "# СТАТУС меняют сами скрипты, руками трогать не нужно:",
        f"#   {STATUS_NEW} -> {STATUS_PHOTO} -> {STATUS_VIDEO} -> {STATUS_DONE} (идея отработана)",
        f"#   {STATUS_REJECTED} — идея забракована и в работу не берётся",
        "",
    ]

    body = "\n\n".join(idea.to_text() for idea in sorted(ideas, key=lambda i: i.number))
    path.write_text("\n".join(head) + body + "\n", encoding="utf-8")


def next_number(ideas: list[Idea]) -> int:
    return (max((i.number for i in ideas), default=0)) + 1


def update_idea(path: Path, number: int, **changes) -> Idea | None:
    """Точечно обновляет одну идею в файле и сохраняет его."""
    ideas = load_ideas(path)
    target = None
    for idea in ideas:
        if idea.number == number:
            target = idea
            for key, value in changes.items():
                if hasattr(idea, key):
                    setattr(idea, key, value)
            break
    if target is None:
        return None
    save_ideas(path, ideas, make_backup=False)
    return target


def ideas_with_status(ideas: list[Idea], status: str) -> list[Idea]:
    return [i for i in ideas if i.status == status]


def stamp_today() -> str:
    return datetime.now().strftime("%Y-%m-%d")
