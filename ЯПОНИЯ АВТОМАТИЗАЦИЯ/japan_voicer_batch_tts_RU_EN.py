#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Voicer API batch TTS generator для проекта «ЯПОНИЯ АВТОМАТИЗАЦИЯ» — японские привычки (RU / EN).

Что делает:
1. Берёт текстовые сценарии из папки:
   /Users/aleksandrtomilov/Desktop/ЯПОНИЯ АВТОМАТИЗАЦИЯ/СЦЕНАРИИ

2. Озвучивает файлы строго по очереди:
   RU -> EN

3. Для каждого языка использует свой UUID шаблона Voicer API:
   RU -> e024fe79-22eb-456f-acda-24469dc45070
   EN -> a2381e7f-19c9-420b-a0c9-e6150a8db140

4. Сохраняет результат в папку:
   /Users/aleksandrtomilov/Desktop/ЯПОНИЯ АВТОМАТИЗАЦИЯ/ОЗВУЧКА

5. Работает через актуальный Voicer API:
     POST /tasks                  -> создать задачу
     GET  /tasks/{task_id}/status -> проверить статус
     GET  /tasks/{task_id}/result -> скачать MP3 или ZIP
     GET  /balance                -> проверить рабочий домен API

Запуск:
   cd "/Users/aleksandrtomilov/Desktop/ЯПОНИЯ АВТОМАТИЗАЦИЯ"
   source venv/bin/activate
   python3 voicer_batch_tts_HISTORY_RU_EN.py
"""

from __future__ import annotations

import json
import re
import sys
import time
import zipfile
from pathlib import Path
from typing import Optional

import requests


# =========================
# НАСТРОЙКИ
# =========================

SCRIPT_VERSION = "JAPAN_HABITS_RU_EN_2026_07_TEMPLATE_VOICE"

# Ключ на озвучку. Передаётся в заголовке X-API-Key.
# Оставлен из исходного скрипта. При необходимости замени на новый ключ.
API_KEY = "550833620:537261493554306c5271754463367564526e78734d673d3d"

# Основной и резервный домены Voicer API.
PRIMARY_BASE_URL = "https://voiceapi.csv666.ru"
BACKUP_BASE_URL = "https://voiceapiru.csv666.ru"

# Папка проекта, папка сценариев и папка готовой озвучки.
BASE_DIR = Path("/Users/aleksandrtomilov/Desktop/ЯПОНИЯ АВТОМАТИЗАЦИЯ")
SCENARIOS_DIR = BASE_DIR / "СЦЕНАРИИ"
OUTPUT_DIR = BASE_DIR / "ОЗВУЧКА"

# UUID шаблонов Voicer API по языкам.
# Голос, модель, движок, водяной знак и прочие настройки берутся из шаблона.
TEMPLATE_UUIDS: dict[str, str] = {
    "RU": "e024fe79-22eb-456f-acda-24469dc45070",
    "EN": "a2381e7f-19c9-420b-a0c9-e6150a8db140",
}

# Порядок озвучки.
LANG_ORDER = ["RU", "EN"]

# Явные варианты имён файлов. Сначала скрипт ищет их.
# Если точного имени нет, включается авто-поиск по ключевым словам ниже.
SCENARIO_FILE_CANDIDATES: dict[str, list[str]] = {
    "RU": [
        "RU.txt",
        "ru.txt",
        "scenario_ru.txt",
        "script_ru.txt",
        "Сценарий_RU.txt",
        "Сценарий_RU.txt",
    ],
    "EN": [
        "EN.txt",
        "en.txt",
        "scenario_en.txt",
        "script_en.txt",
        "Сценарий_EN.txt",
        "Сценарий_EN.txt",
    ],
}

# Подсказки для авто-поиска файла, если точные имена не найдены.
# Проверяется имя файла без учёта регистра.
LANG_HINTS: dict[str, list[str]] = {
    "RU": ["ru", "rus", "russian", "рус", "русск"],
    "EN": ["en", "eng", "english", "англ", "англий"],
}

# Проверка статуса каждые N секунд.
POLL_SECONDS = 8

# Максимум ожидания одной озвучки, секунд. 0 = ждать бесконечно.
MAX_WAIT_SECONDS = 0

REQUEST_TIMEOUT = 120

# Сетевые ретраи. Voicer иногда рвёт соединение (Connection reset by peer).
# Скрипт повторяет запрос с нарастающей паузой, а не падает от первого обрыва.
MAX_NETWORK_RETRIES = 8
NETWORK_RETRY_BASE_DELAY = 5   # секунды: пауза = base * попытка (5,10,15,...), максимум 60

# Разбивка длинного текста на чанки (символы). None = не передавать поле.
CHUNK_SIZE: Optional[int] = 2000

# Настройки пауз (PauseSettings из API). None = не передавать, тогда паузы
# полностью определяются шаблоном.
PAUSE_SETTINGS: Optional[dict] = None

# Настройки ударений (StressSettings). None = не передавать, берётся из шаблона.
STRESS_SETTINGS: Optional[dict] = None

# True = озвучивать заново, даже если готовый файл уже лежит в OUTPUT_DIR.
# Поставь False, когда всё проверишь, чтобы не тратить символы повторно.
FORCE_REGENERATE = True


# =========================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =========================

def log(message: str) -> None:
    print(message, flush=True)


def json_headers() -> dict:
    return {
        "X-API-Key": API_KEY,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def download_headers() -> dict:
    return {
        "X-API-Key": API_KEY,
    }


def clean_text(text: str) -> str:
    """Убираем BOM и лишние пробелы по краям. Сам текст отправляем как есть."""
    text = text.replace("\ufeff", "")
    return text.strip()


def ensure_dirs() -> None:
    if not SCENARIOS_DIR.exists():
        raise FileNotFoundError(f"Не найдена папка со сценариями: {SCENARIOS_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def all_txt_files() -> list[Path]:
    files = []
    for pattern in ("*.txt", "*.TXT"):
        files.extend(SCENARIOS_DIR.glob(pattern))
    # Игнорируем служебные файлы-заглушки и readme, чтобы они не попадали в сценарии.
    ignore_prefixes = ("ПОЛОЖИ_СЮДА", "ЧИТАЙ_МЕНЯ", ".keep")
    files = [p for p in files if not p.name.upper().startswith(tuple(x.upper() for x in ignore_prefixes))]
    return sorted(set(files), key=lambda p: p.name.lower())


def has_language_marker(path: Path, lang: str) -> bool:
    """
    Проверяет, что язык явно указан в имени файла.
    Например: RU.txt, story_RU.txt, history-en-final.txt.
    """
    stem_low = path.stem.lower()
    lang_low = lang.lower()

    # Самый надёжный вариант: имя файла ровно RU / EN.
    if stem_low == lang_low:
        return True

    # Маркер языка как отдельный фрагмент: _ru, -ru, ru_, ru-, (ru).
    pattern = rf"(^|[^a-zа-яё0-9]){re.escape(lang_low)}([^a-zа-яё0-9]|$)"
    if re.search(pattern, stem_low, flags=re.IGNORECASE):
        return True

    # Дополнительные языковые подсказки.
    hints = LANG_HINTS.get(lang, [])
    return any(hint.lower() in stem_low for hint in hints)


def find_scenario_file(lang: str) -> Path:
    """
    Возвращает путь к сценарию для языка.

    Логика:
    1. Ищет точные имена из SCENARIO_FILE_CANDIDATES.
    2. Делает мягкую проверку регистра.
    3. Ищет .txt-файл, где язык явно указан в имени.
    """
    txt_files = all_txt_files()

    if not txt_files:
        raise FileNotFoundError(f"В папке нет .txt файлов: {SCENARIOS_DIR}")

    candidates = SCENARIO_FILE_CANDIDATES.get(lang, [])

    # 1. Точное имя.
    for filename in candidates:
        exact = SCENARIOS_DIR / filename
        if exact.exists():
            return exact

    # 2. Мягкая проверка регистра.
    candidate_names_low = {name.lower() for name in candidates}
    for candidate in txt_files:
        if candidate.name.lower() in candidate_names_low:
            return candidate

    # 3. Авто-поиск по языковому маркеру в имени.
    matched = [candidate for candidate in txt_files if has_language_marker(candidate, lang)]

    if len(matched) == 1:
        return matched[0]

    if len(matched) > 1:
        names = "\n".join(f"- {p.name}" for p in matched)
        raise RuntimeError(
            f"Найдено несколько файлов для {lang}. Переименуй нужный файл в {lang}.txt "
            f"или укажи точное имя в SCENARIO_FILE_CANDIDATES.\n{names}"
        )

    raise FileNotFoundError(
        f"Не найден файл для {lang}. Ожидались имена {candidates} "
        f"или .txt файл с явным маркером языка в имени в папке {SCENARIOS_DIR}"
    )


def output_base_for_file(scenario_path: Path) -> Path:
    """Готовый файл сохраняется с тем же именем, что и исходный сценарий."""
    return OUTPUT_DIR / scenario_path.stem


def template_uuid_for_language(lang: str) -> str:
    try:
        return TEMPLATE_UUIDS[lang]
    except KeyError as exc:
        raise RuntimeError(f"Не задан TEMPLATE_UUID для языка {lang}") from exc


def pretty_json(data) -> str:
    try:
        return json.dumps(data, ensure_ascii=False, indent=2)
    except Exception:
        return str(data)


def _retry_delay(attempt: int) -> int:
    return min(60, NETWORK_RETRY_BASE_DELAY * attempt)


def request_json(method: str, base_url: str, path: str, *, json_body: Optional[dict] = None) -> dict:
    url = base_url.rstrip("/") + path

    last_exc: Optional[Exception] = None
    for attempt in range(1, MAX_NETWORK_RETRIES + 1):
        try:
            response = requests.request(
                method=method,
                url=url,
                headers=json_headers(),
                json=json_body,
                timeout=REQUEST_TIMEOUT,
            )
        except requests.RequestException as exc:
            # Обрыв соединения / таймаут / reset — повторяем с паузой.
            last_exc = exc
            if attempt >= MAX_NETWORK_RETRIES:
                break
            delay = _retry_delay(attempt)
            log(f"[NET] {method} {url}: попытка {attempt}/{MAX_NETWORK_RETRIES} не удалась ({exc}). Повтор через {delay}s...")
            time.sleep(delay)
            continue

        # Ретраим временные серверные статусы (429 и 5xx); остальное отдаём как есть.
        if response.status_code in (429, 500, 502, 503, 504) and attempt < MAX_NETWORK_RETRIES:
            delay = _retry_delay(attempt)
            log(f"[NET] {method} {url}: HTTP {response.status_code}, попытка {attempt}/{MAX_NETWORK_RETRIES}. Повтор через {delay}s...")
            time.sleep(delay)
            continue

        try:
            data = response.json()
        except Exception:
            data = {"raw_text": response.text}

        if response.status_code >= 400:
            raise RuntimeError(f"HTTP {response.status_code} от {url}\nОтвет:\n{pretty_json(data)}")

        return data

    raise RuntimeError(f"Ошибка соединения с {url} после {MAX_NETWORK_RETRIES} попыток: {last_exc}")


def choose_working_base_url() -> str:
    """Проверяем домены через /balance. Первый ответивший становится рабочим."""
    for base_url in [PRIMARY_BASE_URL, BACKUP_BASE_URL]:
        try:
            data = request_json("GET", base_url, "/balance")
            balance = data.get("balance_text") or data.get("balance")
            log(f"[API] Использую {base_url} | balance: {balance}")
            return base_url
        except Exception as exc:
            log(f"[WARN] Не удалось проверить {base_url}: {exc}")

    raise RuntimeError("Не удалось подключиться ни к основному, ни к backup Voicer API.")


def build_task_payload(text: str, lang: str) -> dict:
    """
    Собираем тело POST /tasks строго по схеме TaskCreateRequest.
    Голос берётся из шаблона языка, поэтому voice/model/watermark тут НЕ передаём.
    """
    payload: dict = {
        "text": text,
        "template_uuid": template_uuid_for_language(lang),
    }

    if CHUNK_SIZE is not None:
        payload["chunk_size"] = CHUNK_SIZE

    if PAUSE_SETTINGS is not None:
        payload["pause_settings"] = PAUSE_SETTINGS

    if STRESS_SETTINGS is not None:
        payload["stress_settings"] = STRESS_SETTINGS

    return payload


def create_tts_task(base_url: str, text: str, lang: str) -> int:
    payload = build_task_payload(text, lang)
    data = request_json("POST", base_url, "/tasks", json_body=payload)

    task_id = data.get("task_id")
    if task_id is None:
        raise RuntimeError(f"API не вернул task_id. Ответ:\n{pretty_json(data)}")

    message = data.get("message", "")
    log(f"[TASK] {lang} | создана задача #{task_id}. {message}")
    return int(task_id)


def describe_task_error(data: dict) -> str:
    """Достаём человекочитаемую ошибку из TaskStatusResponse.error."""
    err = data.get("error")
    if isinstance(err, dict):
        ru = err.get("ru") or err.get("en") or err.get("code")
        if ru:
            return str(ru)
    return pretty_json(data)


def wait_for_task(base_url: str, task_id: int) -> str:
    """
    Ждём задачу по статусам:
      waiting -> processing -> ending -> ending_processed
      error / error_handled -> исключение
    Готовым считаем ending и ending_processed.
    """
    started = time.time()

    while True:
        data = request_json("GET", base_url, f"/tasks/{task_id}/status")
        status = data.get("status")
        label = data.get("status_label", "")

        log(f"[STATUS] task_id={task_id} | {status} {label}")

        if status in {"ending", "ending_processed"}:
            return status

        if status in {"error", "error_handled"}:
            raise RuntimeError(f"TTS задача #{task_id} завершилась ошибкой: {describe_task_error(data)}")

        if MAX_WAIT_SECONDS > 0 and time.time() - started > MAX_WAIT_SECONDS:
            raise TimeoutError(f"Превышено время ожидания задачи #{task_id}: {MAX_WAIT_SECONDS} секунд")

        time.sleep(POLL_SECONDS)


def filename_from_content_disposition(value: str) -> Optional[str]:
    if not value:
        return None

    # filename*=UTF-8''...
    m = re.search(r"filename\*=UTF-8''([^;]+)", value, flags=re.IGNORECASE)
    if m:
        from urllib.parse import unquote
        return unquote(m.group(1).strip().strip('"'))

    # filename="..."
    m = re.search(r'filename="?([^";]+)"?', value, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()

    return None


def guess_ext_from_response(response: requests.Response) -> str:
    cd_name = filename_from_content_disposition(response.headers.get("Content-Disposition", ""))
    if cd_name:
        suffix = Path(cd_name).suffix
        if suffix:
            return suffix

    content_type = response.headers.get("Content-Type", "").lower()

    if "zip" in content_type:
        return ".zip"
    if "mpeg" in content_type or "mp3" in content_type or "audio" in content_type:
        return ".mp3"

    return ".mp3"


def download_result(base_url: str, task_id: int, out_base: Path) -> Path:
    url = base_url.rstrip("/") + f"/tasks/{task_id}/result"

    response = None
    last_exc: Optional[Exception] = None
    for attempt in range(1, MAX_NETWORK_RETRIES + 1):
        try:
            response = requests.get(url, headers=download_headers(), timeout=REQUEST_TIMEOUT)
        except requests.RequestException as exc:
            last_exc = exc
            if attempt >= MAX_NETWORK_RETRIES:
                raise RuntimeError(f"Ошибка скачивания результата {url} после {MAX_NETWORK_RETRIES} попыток: {exc}") from exc
            delay = _retry_delay(attempt)
            log(f"[NET] download {url}: попытка {attempt}/{MAX_NETWORK_RETRIES} не удалась ({exc}). Повтор через {delay}s...")
            time.sleep(delay)
            continue

        # 202 = ещё не готово; сервер иногда так отвечает сразу после статуса. Ждём и повторяем.
        if response.status_code == 202 and attempt < MAX_NETWORK_RETRIES:
            delay = _retry_delay(attempt)
            log(f"[NET] результат ещё не готов (202), попытка {attempt}/{MAX_NETWORK_RETRIES}. Повтор через {delay}s...")
            time.sleep(delay)
            continue

        if response.status_code in (429, 500, 502, 503, 504) and attempt < MAX_NETWORK_RETRIES:
            delay = _retry_delay(attempt)
            log(f"[NET] download HTTP {response.status_code}, попытка {attempt}/{MAX_NETWORK_RETRIES}. Повтор через {delay}s...")
            time.sleep(delay)
            continue

        break

    if response is None:
        raise RuntimeError(f"Не удалось скачать результат {url}: {last_exc}")

    if response.status_code == 202:
        raise RuntimeError(f"Файл ещё не готов, хотя статус уже проверен. Ответ: {response.text}")

    if response.status_code >= 400:
        raise RuntimeError(f"HTTP {response.status_code} при скачивании результата.\nОтвет:\n{response.text}")

    ext = guess_ext_from_response(response)
    out_path = out_base.with_suffix(ext)
    out_path.write_bytes(response.content)

    if not out_path.exists() or out_path.stat().st_size == 0:
        raise RuntimeError(f"Файл не сохранился или пустой: {out_path}")

    log(f"[SAVE] {out_path} ({out_path.stat().st_size / 1024 / 1024:.2f} MB)")
    return out_path


def unzip_if_needed(path: Path) -> None:
    if path.suffix.lower() != ".zip":
        return

    extract_dir = path.with_suffix("")
    extract_dir.mkdir(parents=True, exist_ok=True)

    try:
        with zipfile.ZipFile(path, "r") as zf:
            zf.extractall(extract_dir)
        log(f"[ZIP] Распаковано: {extract_dir}")
    except Exception as exc:
        log(f"[WARN] ZIP скачался, но не распаковался: {exc}")


def process_one_language(base_url: str, lang: str) -> None:
    scenario_path = find_scenario_file(lang)
    text = clean_text(scenario_path.read_text(encoding="utf-8"))

    if not text:
        raise RuntimeError(f"Файл пустой: {scenario_path}")

    template_uuid = template_uuid_for_language(lang)

    log("")
    log("=" * 60)
    log(f"[START] {lang}")
    log(f"[FILE] {scenario_path}")
    log(f"[TEXT] {len(text)} символов")
    log(f"[TEMPLATE] {template_uuid}")

    out_base = output_base_for_file(scenario_path)
    existing_mp3 = out_base.with_suffix(".mp3")
    existing_zip = out_base.with_suffix(".zip")

    if not FORCE_REGENERATE:
        if existing_mp3.exists() and existing_mp3.stat().st_size > 0:
            log(f"[SKIP] Уже есть: {existing_mp3}")
            return
        if existing_zip.exists() and existing_zip.stat().st_size > 0:
            log(f"[SKIP] Уже есть: {existing_zip}")
            return
    else:
        if existing_mp3.exists() or existing_zip.exists():
            log("[REGEN] Старый файл найден, но FORCE_REGENERATE=True — озвучиваю заново.")

    task_id = create_tts_task(base_url, text, lang)
    wait_for_task(base_url, task_id)
    out_path = download_result(base_url, task_id, out_base)
    unzip_if_needed(out_path)

    log(f"[DONE] {lang}")


def main() -> None:
    if not API_KEY or API_KEY == "your-api-key-here":
        print("[ERROR] Вставь API_KEY в начале скрипта.")
        sys.exit(1)

    log(f"[SCRIPT_VERSION] {SCRIPT_VERSION}")
    log(f"[SCENARIOS] {SCENARIOS_DIR}")
    log(f"[OUTPUT] {OUTPUT_DIR}")

    ensure_dirs()
    base_url = choose_working_base_url()

    log(f"[ORDER] {' -> '.join(LANG_ORDER)}")
    for lang in LANG_ORDER:
        log(f"[TEMPLATE:{lang}] {template_uuid_for_language(lang)}")
    log(f"[CHUNK] size={CHUNK_SIZE}")
    log(f"[PAUSE_SETTINGS] {PAUSE_SETTINGS}")
    log(f"[STRESS_SETTINGS] {STRESS_SETTINGS}")
    log(f"[REGENERATE] {'ON' if FORCE_REGENERATE else 'OFF'}")

    errors = []

    for lang in LANG_ORDER:
        try:
            process_one_language(base_url, lang)
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            log(f"[ERROR] {lang}: {exc}")
            errors.append((lang, str(exc)))

    log("")
    log("=" * 60)
    if errors:
        log("[FINISH] Завершено с ошибками:")
        for lang, error in errors:
            log(f"- {lang}: {error}")
        sys.exit(1)

    log("[FINISH] Все озвучки готовы.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nОстановлено пользователем.")
        sys.exit(1)
