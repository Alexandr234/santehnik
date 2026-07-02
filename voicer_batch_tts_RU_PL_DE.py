#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Voicer API batch TTS generator (RU / PL / DE).

Что делает:
1. Берёт текстовые сценарии из папки:
   /Users/aleksandrtomilov/Desktop/ПСИХОЛОГИЯ ГЕРМАНИЯ ПОЛЬША/СЦЕНАРИИ

2. Озвучивает 3 файла строго по очереди:
   RU -> PL -> DE

3. Каждый файл озвучивается голосом шаблона (TEMPLATE_UUID) из Voicer API.
   Голос, модель, движок, снятие водяного знака и прочие настройки берутся
   из самого шаблона — их не нужно дублировать в теле задачи.

4. Сохраняет результат в папку:
   /Users/aleksandrtomilov/Desktop/ПСИХОЛОГИЯ ГЕРМАНИЯ ПОЛЬША/СКРИПТ ОЗВУЧКИ
   (именно отсюда следующий скрипт — psych_prompt_pipeline_v2 — берёт озвучки,
   поэтому папки должны совпадать: scenario_ru.mp3 / scenario_pl.mp3 / scenario_de.mp3)

5. Работает через актуальный Voicer API (OAS 3.1, версия 1.1.0):
     POST /tasks                     -> TaskCreateResponse { task_id, message }
     GET  /tasks/{task_id}/status    -> TaskStatusResponse { task_id, status, status_label, ... }
     GET  /tasks/{task_id}/result    -> бинарный файл (MP3 или ZIP)
     GET  /balance                   -> BalanceResponse
     GET  /templates                 -> список шаблонов пользователя

   Тело POST /tasks (TaskCreateRequest) принимает ТОЛЬКО эти поля:
     template_uuid, template, text, chunk_size, pause_settings, stress_settings, task_type
   Поэтому voice_id / model_id / voice_settings / remove_watermark на верхнем
   уровне задачи НЕ передаются — всё это часть шаблона.

Запуск:
   cd "/Users/aleksandrtomilov/Desktop/ПСИХОЛОГИЯ ГЕРМАНИЯ ПОЛЬША"
   source venv/bin/activate
   python3 voicer_batch_tts_RU_PL_DE.py
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

SCRIPT_VERSION = "RU_PL_DE_2026_07_02_TEMPLATE_VOICE"

# Ключ на озвучку. Передаётся в заголовке X-API-Key.
API_KEY = "550833620:537261493554306c5271754463367564526e78734d673d3d"

# Основной и резервный домены Voicer API.
PRIMARY_BASE_URL = "https://voiceapi.csv666.ru"
BACKUP_BASE_URL = "https://voiceapiru.csv666.ru"

# Папка с исходными текстами и папка для готовой озвучки.
BASE_DIR = Path("/Users/aleksandrtomilov/Desktop/ПСИХОЛОГИЯ ГЕРМАНИЯ ПОЛЬША")
SCENARIOS_DIR = BASE_DIR / "СЦЕНАРИИ"
# ВАЖНО: сюда же смотрит следующий скрипт (psych_prompt_pipeline_v2 ->
# DEFAULT_VOICEOVER_FOLDER). Папки обязаны совпадать, иначе пайплайн возьмёт
# не те (старые) озвучки. Имена файлов: scenario_ru.mp3 / scenario_pl.mp3 / scenario_de.mp3.
OUTPUT_DIR = BASE_DIR / "СКРИПТ ОЗВУЧКИ"

# UUID шаблона из API. Именно его голосом озвучиваются все файлы.
# Голос / модель / движок / водяной знак / настройки берутся из шаблона.
TEMPLATE_UUID: str = "c552325d-9b67-43ad-abe6-d82b946c860b"

# Порядок озвучки.
LANG_ORDER = ["RU", "PL", "DE"]

# Явные имена файлов для каждого языка. Если файла с таким именем нет,
# скрипт попытается найти подходящий .txt в папке по ключевым словам
# (см. LANG_HINTS ниже).
SCENARIO_FILES = {
    "RU": "scenario_ru.txt",
    "PL": "scenario_pl.txt",
    "DE": "scenario_de.txt",
}

# Подсказки для авто-поиска файла, если точное имя из SCENARIO_FILES не найдено.
# Проверяется вхождение любого из этих кусочков в имя файла (без учёта регистра).
LANG_HINTS = {
    "RU": ["ru", "rus", "рус", "russ"],
    "PL": ["pl", "pol", "поль", "polsk"],
    "DE": ["de", "ger", "deu", "нем", "germ"],
}

# Проверка статуса каждые N секунд.
POLL_SECONDS = 8

# Максимум ожидания одной озвучки, секунд. 0 = ждать бесконечно.
MAX_WAIT_SECONDS = 0

REQUEST_TIMEOUT = 120

# Разбивка длинного текста на чанки (символы). None = не передавать поле.
CHUNK_SIZE: Optional[int] = 2000

# Настройки пауз (PauseSettings из API). None = не передавать, тогда паузы
# полностью определяются шаблоном. Если хочешь управлять паузами вручную —
# заполни поля по схеме:
#   enabled: bool, max_pause_symb: int, pause_time: float, auto_paragraph_pause: bool
PAUSE_SETTINGS: Optional[dict] = None

# Настройки ударений (StressSettings). None = не передавать (берётся из шаблона).
#   {"enabled": True}
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
    """Убираем BOM и лишние пробелы по краям. Текст отправляем как есть."""
    text = text.replace("﻿", "")
    return text.strip()


def ensure_dirs() -> None:
    if not SCENARIOS_DIR.exists():
        raise FileNotFoundError(f"Не найдена папка со сценариями: {SCENARIOS_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def find_scenario_file(lang: str) -> Path:
    """
    Возвращает путь к сценарию для языка.

    Сначала пробуем точное имя из SCENARIO_FILES (с мягкой проверкой регистра),
    затем ищем по ключевым словам из LANG_HINTS среди всех .txt в папке.
    """
    filename = SCENARIO_FILES.get(lang)
    txt_files = sorted(SCENARIOS_DIR.glob("*.txt")) + sorted(SCENARIOS_DIR.glob("*.TXT"))

    if filename:
        exact = SCENARIOS_DIR / filename
        if exact.exists():
            return exact
        # Мягкая проверка регистра (macOS обычно case-insensitive, но подстрахуемся).
        for candidate in txt_files:
            if candidate.name.lower() == filename.lower():
                return candidate

    # Авто-поиск по ключевым словам.
    hints = LANG_HINTS.get(lang, [])
    for candidate in txt_files:
        name_low = candidate.name.lower()
        if any(hint in name_low for hint in hints):
            return candidate

    raise FileNotFoundError(
        f"Не найден файл для {lang}. Ожидалось имя '{filename}' "
        f"или файл, содержащий одно из {hints}, в папке {SCENARIOS_DIR}"
    )


def output_base_for_language(lang: str) -> Path:
    """Имя выходного файла без расширения: scenario_ru / scenario_pl / scenario_de."""
    filename = SCENARIO_FILES.get(lang, f"scenario_{lang.lower()}")
    return OUTPUT_DIR / Path(filename).stem


def pretty_json(data) -> str:
    try:
        return json.dumps(data, ensure_ascii=False, indent=2)
    except Exception:
        return str(data)


def request_json(method: str, base_url: str, path: str, *, json_body: Optional[dict] = None) -> dict:
    url = base_url.rstrip("/") + path

    try:
        response = requests.request(
            method=method,
            url=url,
            headers=json_headers(),
            json=json_body,
            timeout=REQUEST_TIMEOUT,
        )
    except requests.RequestException as exc:
        raise RuntimeError(f"Ошибка соединения с {url}: {exc}") from exc

    try:
        data = response.json()
    except Exception:
        data = {"raw_text": response.text}

    if response.status_code >= 400:
        raise RuntimeError(f"HTTP {response.status_code} от {url}\nОтвет:\n{pretty_json(data)}")

    return data


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


def build_task_payload(text: str) -> dict:
    """
    Собираем тело POST /tasks строго по схеме TaskCreateRequest:
      template_uuid, template, text, chunk_size, pause_settings, stress_settings, task_type
    Голос берётся из шаблона, поэтому voice/model/watermark тут НЕ передаём.
    """
    payload: dict = {
        "text": text,
        "template_uuid": TEMPLATE_UUID,
    }

    if CHUNK_SIZE is not None:
        payload["chunk_size"] = CHUNK_SIZE

    if PAUSE_SETTINGS is not None:
        payload["pause_settings"] = PAUSE_SETTINGS

    if STRESS_SETTINGS is not None:
        payload["stress_settings"] = STRESS_SETTINGS

    return payload


def create_tts_task(base_url: str, text: str) -> int:
    payload = build_task_payload(text)
    data = request_json("POST", base_url, "/tasks", json_body=payload)

    task_id = data.get("task_id")
    if task_id is None:
        raise RuntimeError(f"API не вернул task_id. Ответ:\n{pretty_json(data)}")

    message = data.get("message", "")
    log(f"[TASK] Создана задача #{task_id}. {message}")
    return int(task_id)


def describe_task_error(data: dict) -> str:
    """Достаём человекочитаемую ошибку из TaskStatusResponse.error (TaskError)."""
    err = data.get("error")
    if isinstance(err, dict):
        ru = err.get("ru") or err.get("en") or err.get("code")
        if ru:
            return str(ru)
    return pretty_json(data)


def wait_for_task(base_url: str, task_id: int) -> str:
    """
    Ждём задачу по статусам OrderStatus:
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

    # fallback
    return ".mp3"


def download_result(base_url: str, task_id: int, out_base: Path) -> Path:
    url = base_url.rstrip("/") + f"/tasks/{task_id}/result"

    try:
        response = requests.get(url, headers=download_headers(), timeout=REQUEST_TIMEOUT)
    except requests.RequestException as exc:
        raise RuntimeError(f"Ошибка скачивания результата {url}: {exc}") from exc

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

    log("")
    log("=" * 60)
    log(f"[START] {lang}")
    log(f"[FILE] {scenario_path}")
    log(f"[TEXT] {len(text)} символов")

    out_base = output_base_for_language(lang)
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

    task_id = create_tts_task(base_url, text)
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
    log(f"[TEMPLATE] {TEMPLATE_UUID}")
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
