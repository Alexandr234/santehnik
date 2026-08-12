#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
СТАРТ — единственный скрипт, который нужно запускать.

Делает всё сам и в правильном порядке:
  1. проверяет версию Python;
  2. доустанавливает недостающие библиотеки (openai, requests, Pillow);
  3. проверяет ffmpeg и подсказывает, как поставить, если его нет;
  4. проверяет ключи OPENAI_API_KEY и FAST_GEN_API_KEY, при желании
     сохраняет их в ~/.zshrc, чтобы больше не спрашивать;
  5. проверяет, что на месте фото лица, видеосреднее.mp4 и папка мелодий;
  6. создаёт ИДЕИ.txt из образца, если его ещё нет;
  7. чистит забракованные идеи (парашюты, яхты, надписи от первого лица);
  8. досоздаёт идеи, если готовых не хватает;
  9. прогоняет фото -> оживление -> монтаж;
 10. показывает, что получилось и где лежит.

Запуск:
    python3 СТАРТ.py
    python3 СТАРТ.py --count 5      # сколько роликов сделать
    python3 СТАРТ.py --yes          # ничего не спрашивать, брать значения по умолчанию
    python3 СТАРТ.py --only-check   # только проверка окружения, без генерации

На маке можно просто дважды кликнуть по файлу СТАРТ.command рядом.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Библиотека -> как она называется при импорте
REQUIRED_PACKAGES = {
    "openai": "openai",
    "requests": "requests",
    "Pillow": "PIL",
}

ASK = True          # переключается флагом --yes
PROBLEMS: list[str] = []


# =============================================================================
# ВЫВОД
# =============================================================================

def title(text: str) -> None:
    print(f"\n{'=' * 68}\n{text}\n{'=' * 68}")


def ok(text: str) -> None:
    print(f"  [ОК]     {text}")


def warn(text: str) -> None:
    print(f"  [!]      {text}")


def fail(text: str) -> None:
    print(f"  [ОШИБКА] {text}")
    PROBLEMS.append(text)


def ask_yes(question: str, default: bool = True) -> bool:
    if not ASK:
        return default
    suffix = "[Д/н]" if default else "[д/Н]"
    try:
        answer = input(f"  {question} {suffix}: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return default
    if not answer:
        return default
    return answer[0] in ("д", "y", "1")


def ask_text(question: str) -> str:
    if not ASK:
        return ""
    try:
        return input(f"  {question}: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return ""


# =============================================================================
# 1. PYTHON
# =============================================================================

def check_python() -> None:
    title("1. ПРОВЕРКА PYTHON")
    version = sys.version_info
    print(f"  Версия: {version.major}.{version.minor}.{version.micro}")
    print(f"  Путь:   {sys.executable}")
    if version < (3, 9):
        fail("Нужен Python 3.9 или новее. Скачайте с python.org")
    else:
        ok("версия подходит")


# =============================================================================
# 2. БИБЛИОТЕКИ
# =============================================================================

def pip_install(package: str) -> bool:
    """Ставит пакет. Если система запрещает — пробует поставить в пользователя."""
    base = [sys.executable, "-m", "pip", "install", "--quiet", "--disable-pip-version-check"]
    for extra in ([], ["--user"], ["--break-system-packages"]):
        result = subprocess.run(base + extra + [package], capture_output=True, text=True)
        if result.returncode == 0:
            return True
    print(result.stdout.strip()[-800:])
    print(result.stderr.strip()[-800:])
    return False


def check_packages() -> None:
    title("2. ПРОВЕРКА БИБЛИОТЕК")
    import importlib

    for package, module in REQUIRED_PACKAGES.items():
        try:
            importlib.import_module(module)
            ok(f"{package} уже установлена")
            continue
        except ImportError:
            pass

        print(f"  [...]    {package} не найдена, устанавливаю...")
        if not pip_install(package):
            fail(f"не удалось установить {package}. Поставьте вручную: pip3 install {package}")
            continue

        importlib.invalidate_caches()
        try:
            importlib.import_module(module)
            ok(f"{package} установлена")
        except ImportError:
            fail(f"{package} установилась, но не импортируется. Перезапустите скрипт.")


# =============================================================================
# 3. FFMPEG
# =============================================================================

def check_ffmpeg() -> None:
    title("3. ПРОВЕРКА FFMPEG")

    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        fail("ffmpeg не найден")
        if shutil.which("brew"):
            if ask_yes("Установить ffmpeg через Homebrew прямо сейчас?", default=True):
                print("  Устанавливаю, это может занять несколько минут...")
                result = subprocess.run(["brew", "install", "ffmpeg"])
                if result.returncode == 0 and shutil.which("ffmpeg"):
                    PROBLEMS.remove("ffmpeg не найден")
                    ok("ffmpeg установлен")
                    return
                fail("установка ffmpeg не удалась")
        else:
            print("  Homebrew не найден. Установите его командой:")
            print('    /bin/bash -c "$(curl -fsSL '
                  'https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"')
            print("  затем:  brew install ffmpeg")
        return

    ok(f"ffmpeg найден: {shutil.which('ffmpeg')}")

    # drawtext нужен только как запасной путь — надпись рисуется через Pillow
    result = subprocess.run(["ffmpeg", "-v", "quiet", "-filters"], capture_output=True, text=True)
    if " drawtext " in result.stdout:
        ok("фильтр drawtext доступен")
    else:
        warn("в этой сборке ffmpeg нет drawtext — не страшно, надпись рисуется через Pillow")


# =============================================================================
# 4. КЛЮЧИ
# =============================================================================

def save_key_to_zshrc(name: str, value: str) -> None:
    zshrc = Path.home() / ".zshrc"
    line = f'export {name}="{value}"'
    try:
        existing = zshrc.read_text(encoding="utf-8") if zshrc.exists() else ""
        if line in existing:
            return
        with zshrc.open("a", encoding="utf-8") as f:
            f.write(f"\n# добавлено скриптом ВИТЯ АВТОМАТИЗАЦИЯ\n{line}\n")
        ok(f"{name} сохранён в ~/.zshrc — в следующий раз спрашивать не буду")
    except Exception as exc:  # noqa: BLE001
        warn(f"не смог записать в ~/.zshrc ({exc}). Ключ будет работать только в этом запуске.")


def check_key(name: str, purpose: str, aliases: tuple[str, ...] = ()) -> None:
    value = os.getenv(name, "").strip()
    for alias in aliases:
        if not value:
            value = os.getenv(alias, "").strip()

    if value:
        ok(f"{name} найден ({purpose})")
        os.environ[name] = value
        return

    warn(f"{name} не найден — нужен для: {purpose}")
    entered = ask_text(f"Вставьте {name} (или Enter, чтобы пропустить)")
    if not entered:
        fail(f"без {name} этот шаг работать не будет")
        return

    os.environ[name] = entered
    ok(f"{name} принят")
    if ask_yes("Сохранить его в ~/.zshrc, чтобы больше не вводить?", default=True):
        save_key_to_zshrc(name, entered)


def verify_openai_key() -> None:
    """Дешёвый запрос к OpenAI, чтобы поймать битый ключ до начала генерации."""
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        return
    try:
        from openai import OpenAI  # noqa: PLC0415

        OpenAI(api_key=key, timeout=20.0).models.list()
        ok("ключ OpenAI рабочий, связь есть")
    except Exception as exc:  # noqa: BLE001
        name = type(exc).__name__
        if "Authentication" in name or "PermissionDenied" in name:
            fail("ключ OPENAI_API_KEY недействителен — проверьте его на platform.openai.com")
        elif "Connection" in name or "Timeout" in name:
            fail("нет связи с OpenAI — проверьте интернет, VPN или прокси")
        elif "RateLimit" in name:
            warn("OpenAI отвечает 'превышен лимит' — возможно, закончились средства на балансе")
        else:
            warn(f"не смог проверить ключ OpenAI ({name}) — продолжаю, но генерация может упасть")


def check_keys() -> None:
    title("4. ПРОВЕРКА КЛЮЧЕЙ")
    check_key("OPENAI_API_KEY", "идеи, промпты, генерация фото")
    check_key("FAST_GEN_API_KEY", "оживление фото в видео", aliases=("FASTGEN_API_KEY", "MEDIA_GEN_API_KEY"))
    verify_openai_key()


# =============================================================================
# 5. ФАЙЛЫ ПРОЕКТА
# =============================================================================

def check_files():
    title("5. ПРОВЕРКА ФАЙЛОВ ПРОЕКТА")

    sys.path.insert(0, str(HERE))
    import config  # noqa: PLC0415 — импорт после установки библиотек

    print(f"  Рабочая папка: {config.BASE_DIR}")
    if not config.BASE_DIR.exists():
        fail(f"папка не найдена: {config.BASE_DIR}")
        print("  Поправьте путь в config.py или задайте:")
        print('    export VITYA_BASE_DIR="/путь/к/папке"')
        return None

    config.ensure_dirs()

    # ИДЕИ.txt — создаём из образца, если его нет
    if config.IDEAS_FILE.exists():
        ok(f"ИДЕИ.txt на месте")
    else:
        sample = HERE / "ИДЕИ_стартовый.txt"
        if sample.exists():
            shutil.copyfile(sample, config.IDEAS_FILE)
            ok("ИДЕИ.txt создан из образца")
        else:
            config.IDEAS_FILE.write_text("", encoding="utf-8")
            ok("ИДЕИ.txt создан пустым")

    # Фото лица: папка ЛИЦО с несколькими снимками либо одиночный файл
    face_photos = []
    if config.FACE_DIR.exists():
        face_photos = [
            p for p in config.FACE_DIR.iterdir()
            if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")
            and not p.name.startswith(".")
        ]
    if face_photos:
        ok(f"фото лица: {len(face_photos)} шт. из папки {config.FACE_DIR.name}")
        if len(face_photos) < 3:
            warn("для точного сходства лучше 3-4 снимка: анфас, полуоборот, разный свет")
    elif config.FACE_PHOTO.exists():
        ok(f"фото вашего лица: {config.FACE_PHOTO.name}")
        print(
            f"           Совет: положите 3-4 своих фото в папку «{config.FACE_DIR.name}»\n"
            "           (анфас, полуоборот, разный свет) — сходство станет заметно точнее"
        )
    else:
        fail(
            f"не найдено фото лица (нужно для генерации): {config.FACE_PHOTO}\n"
            f"             либо положите несколько снимков в папку {config.FACE_DIR}"
        )

    if config.MIDDLE_VIDEO.exists():
        ok(f"видеосреднее.mp4: {config.MIDDLE_VIDEO.name}")
    else:
        fail(f"не найден видеосреднее.mp4 (нужен для монтажа): {config.MIDDLE_VIDEO}")

    if config.MUSIC_DIR.exists():
        tracks = [
            p for p in config.MUSIC_DIR.iterdir()
            if p.is_file() and p.suffix.lower() in (".mp3", ".m4a", ".wav", ".aac", ".ogg")
            and not p.name.startswith(".")
        ]
        if tracks:
            ok(f"мелодии: {len(tracks)} шт.")
        else:
            warn("папка мелодий пустая — ролики будут без музыки")
    else:
        warn(f"папка мелодий не найдена: {config.MUSIC_DIR} — ролики будут без музыки")

    # Шрифт для надписи
    try:
        ok(f"шрифт надписи: {Path(config.font_path()).name}")
    except Exception as exc:  # noqa: BLE001
        fail(str(exc))

    return config


# =============================================================================
# ЗАПУСК ШАГОВ
# =============================================================================

def run_step(script: str, args: list[str]) -> bool:
    print(f"\n{'-' * 68}\n>>> {script} {' '.join(args)}\n{'-' * 68}")
    result = subprocess.run([sys.executable, str(HERE / script), *args], env=os.environ)
    if result.returncode != 0:
        print(f"\n  Шаг {script} завершился с ошибкой (код {result.returncode}).")
        return False
    return True


def count_ready_ideas(config) -> int:
    """Сколько идей готово к съёмке (статус НОВАЯ)."""
    from ideas_store import STATUS_NEW, load_ideas  # noqa: PLC0415

    return sum(1 for i in load_ideas(config.IDEAS_FILE) if i.status == STATUS_NEW)


def main() -> None:
    global ASK

    parser = argparse.ArgumentParser(description="Полный автоматический прогон конвейера.")
    parser.add_argument("--count", type=int, default=0, help="Сколько роликов сделать.")
    parser.add_argument("--yes", action="store_true", help="Ничего не спрашивать.")
    parser.add_argument("--only-check", action="store_true", help="Только проверка окружения.")
    parser.add_argument("--backend", choices=["flow", "openai"], default=None)
    args = parser.parse_args()

    ASK = not args.yes

    print("\n" + "=" * 68)
    print("  ВИТЯ АВТОМАТИЗАЦИЯ — полный прогон".center(68))
    print("=" * 68)

    check_python()
    check_packages()
    check_ffmpeg()
    check_keys()
    config = check_files()

    title("ИТОГ ПРОВЕРКИ")
    if PROBLEMS:
        print("  Найдены проблемы, из-за которых прогон не пойдёт:\n")
        for problem in PROBLEMS:
            print(f"    - {problem}")
        print("\n  Исправьте их и запустите скрипт снова.")
        sys.exit(1)

    ok("всё на месте, можно работать")

    if args.only_check or config is None:
        print("\n  Проверка окончена (запуск был с --only-check).")
        return

    # --- Сколько роликов делаем ---
    count = args.count
    if not count:
        answer = ask_text("Сколько роликов сделать? (Enter = 3)")
        count = int(answer) if answer.isdigit() and int(answer) > 0 else 3
    print(f"\n  Делаем роликов: {count}")

    # --- Шаг 0: чистка старых негодных идей ---
    title("ШАГ 1 из 5. ЧИСТКА СТАРЫХ ИДЕЙ")
    run_step("step1_ideas.py", ["--clean"])

    # --- Шаг 1: досоздать идеи, если не хватает ---
    title("ШАГ 2 из 5. ИДЕИ")
    ready = count_ready_ideas(config)
    print(f"  Готовых к съёмке идей: {ready}, нужно: {count}")
    if ready < count:
        need = count - ready
        print(f"  Досоздаю {need} шт.")
        if not run_step("step1_ideas.py", ["--count", str(need)]):
            sys.exit(1)
        ready = count_ready_ideas(config)
    else:
        print("  Идей достаточно, новые не придумываю.")

    if ready == 0:
        print("\n  Не удалось получить ни одной пригодной идеи. Запустите скрипт ещё раз.")
        sys.exit(1)

    count = min(count, ready)

    if ASK and not ask_yes(
        f"\n  Посмотрите идеи в ИДЕИ.txt. Продолжать съёмку {count} шт.?", default=True
    ):
        print("\n  Остановился. Поправьте ИДЕИ.txt и запустите скрипт снова.")
        return

    # --- Шаг 2: фото ---
    title("ШАГ 3 из 5. ГЕНЕРАЦИЯ ФОТО")
    photo_args = ["--limit", str(count)]
    if args.backend:
        photo_args += ["--backend", args.backend]
    if not run_step("step2_photo.py", photo_args):
        sys.exit(1)

    # --- Шаг 3: оживление ---
    title("ШАГ 4 из 5. ОЖИВЛЕНИЕ ФОТО")
    if not run_step("step3_animate.py", ["--limit", str(count)]):
        sys.exit(1)

    # --- Шаг 4: монтаж ---
    title("ШАГ 5 из 5. МОНТАЖ")
    if not run_step("step4_montage.py", ["--limit", str(count)]):
        sys.exit(1)

    # --- Итог ---
    title("ГОТОВО")
    reels = sorted(config.OUTPUT_DIR.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
    if reels:
        print(f"  Готовые ролики ({len(reels)} шт. всего) лежат в:")
        print(f"    {config.OUTPUT_DIR}\n")
        print("  Последние:")
        for reel in reels[:count]:
            print(f"    - {reel.name}")
    else:
        print("  Ролики не появились — посмотрите сообщения об ошибках выше.")

    print(
        "\n  Дальше: опубликуйте ролики, впишите просмотры в ИДЕИ.txt\n"
        "  (строка ПРОСМОТРЫ: 45000) и запустите этот скрипт снова —\n"
        "  новые идеи будут придумываться по тому, что реально зашло."
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n  Прервано пользователем.")
        sys.exit(130)
