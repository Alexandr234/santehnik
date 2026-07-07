#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
translate_and_voice.py
======================

Берёт англоязычный сценарий, качественно ЛОКАЛИЗУЕТ его на 3 языка
(русский, португальский, испанский) через Chat API OpenAI, а затем
ОЗВУЧИВАЕТ каждый перевод через Voicer API (https://voiceapi.csv666.ru).

Главное требование к переводу: это НЕ дословный перевод, а локализация —
идиомы, фразеологизмы, культурный контекст и интонация адаптируются под
целевой язык так, чтобы текст звучал как написанный носителем.

Порядок работы:
    1) читаем входной .txt (по умолчанию scenario.txt);
    2) режем на чанки по границам абзацев (длинный текст → несколько запросов);
    3) переводим каждый чанк, склеиваем, сохраняем scenario_<lang>.txt;
    4) отправляем перевод в TTS (POST /tasks), ждём готовности,
       скачиваем результат scenario_<lang>.mp3 (или .zip).

Зависимости:  pip install requests

Примеры запуска:
    # всё сразу (перевод + озвучка) для файла по умолчанию
    python3 translate_and_voice.py

    # указать свой файл
    python3 translate_and_voice.py "/Users/aleksandrtomilov/Desktop/ПРОМПТЫ/scenario.txt"

    # только перевод, без озвучки
    python3 translate_and_voice.py --skip-tts

    # озвучить уже готовые переводы (scenario_ru.txt и т.д.), не переводя заново
    python3 translate_and_voice.py --skip-translate

    # только часть языков
    python3 translate_and_voice.py --langs ru,es

    # своя озвучка: конкретный голос или готовый шаблон из Telegram-бота
    python3 translate_and_voice.py --voice-id 21m00Tcm4TlvDq8ikWAM
    python3 translate_and_voice.py --template-uuid 123e4567-e89b-12d3-a456-426614174000
"""

import argparse
import os
import re
import sys
import time
import json

import requests


# ---------------------------------------------------------------------------
# КЛЮЧИ И БАЗОВЫЕ НАСТРОЙКИ
# ---------------------------------------------------------------------------
# TTS-ключ вписан прямо в файл по просьбе владельца.
#
# ⚠️ Ключ OpenAI СЮДА ВПИСАТЬ НЕЛЬЗЯ: GitHub Push Protection блокирует любой push,
# в котором есть ключ формата OpenAI, — файл просто не зальётся в репозиторий.
# Поэтому OpenAI-ключ берётся из переменной окружения. Два способа задать его:
#
#   1) экспортом (рекомендуется — ключ не попадёт в git):
#        export OPENAI_API_KEY="sk-proj-...ваш ключ..."
#        python3 scripts/translate_and_voice.py
#
#   2) вписать ключ в ЛОКАЛЬНУЮ копию строки ниже (замените "" на "sk-proj-...").
#      Только не коммитьте этот файл обратно в GitHub — push будет отклонён.
#
# Обе переменные окружения имеют приоритет над значениями по умолчанию.

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")  # ← сюда можно вставить ключ ЛОКАЛЬНО
TTS_API_KEY = os.environ.get(
    "TTS_API_KEY",
    "550833620:537261493554306c5271754463367564526e78734d673d3d",
)

OPENAI_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_MODEL_DEFAULT = "gpt-4o"

# Основной хост Voicer API; если основной домен заблокирован — резервный.
TTS_BASE_URL = "https://voiceapi.csv666.ru"
TTS_BACKUP_URL = "https://voiceapiru.csv666.ru"

# Голос по умолчанию (eleven_v3 — мультиязычная модель, один голос звучит на всех
# трёх языках). Можно переопределить флагом --voice-id или своим --template-uuid.
DEFAULT_VOICE_ID = "iP95p4xoKVk53GoZ742B"
DEFAULT_TTS_MODEL = "eleven_v3"


# ---------------------------------------------------------------------------
# ЯЗЫКИ ЛОКАЛИЗАЦИИ
# ---------------------------------------------------------------------------
# code -> (человекочитаемое название для промпта, подпись для логов)
LANGUAGES = {
    "ru": ("Russian", "Русский"),
    "pt": ("Brazilian Portuguese", "Português (BR)"),
    "es": ("Spanish (neutral Latin American)", "Español"),
}


def build_system_prompt(target_lang_name: str) -> str:
    """Системный промпт для локализатора: качество и адаптация идиом важнее буквальности."""
    return (
        f"You are a professional localization specialist and native {target_lang_name} "
        f"copywriter. You translate marketing / video scripts from English into "
        f"{target_lang_name}.\n\n"
        "Rules:\n"
        f"1. Produce natural, fluent {target_lang_name} that reads as if originally written "
        "by a native speaker — NOT a literal word-for-word translation.\n"
        "2. Adapt idioms, phraseology, humor, cultural references, units and examples so they "
        f"land naturally for a {target_lang_name}-speaking audience. Replace English idioms with "
        "equivalent native idioms rather than translating them literally.\n"
        "3. Preserve the original meaning, tone, register and emotional intent of every line.\n"
        "4. Keep the structure and formatting: line breaks, paragraph breaks, lists, and any "
        "speaker labels or timestamps must stay in the same places.\n"
        "5. Keep proper names, brand names, URLs, emails and code untouched unless a well-known "
        "localized form exists.\n"
        "6. Keep terminology consistent throughout.\n"
        "7. This text is meant to be READ ALOUD by a text-to-speech engine, so write clean, "
        "speakable sentences (spell out things that should be spoken naturally).\n"
        "8. Output ONLY the translated text. No preamble, no notes, no explanations, "
        "no quotes around it, no markdown fences."
    )


# ---------------------------------------------------------------------------
# ЧАНКИНГ
# ---------------------------------------------------------------------------
def split_into_chunks(text: str, max_chars: int) -> list:
    """
    Режет текст на чанки не длиннее max_chars, стараясь рвать по границам абзацев,
    а если абзац слишком большой — по предложениям. Это даёт связный перевод и
    держит размер каждого запроса под контролем.
    """
    paragraphs = re.split(r"(\n\s*\n)", text)  # сохраняем разделители-абзацы
    chunks = []
    current = ""

    def flush():
        nonlocal current
        if current.strip():
            chunks.append(current)
        current = ""

    for part in paragraphs:
        if len(current) + len(part) <= max_chars:
            current += part
            continue
        # часть не влезает — сбрасываем накопленное
        flush()
        if len(part) <= max_chars:
            current = part
        else:
            # один огромный абзац — режем по предложениям
            for sentence in _split_sentences(part, max_chars):
                if len(current) + len(sentence) <= max_chars:
                    current += sentence
                else:
                    flush()
                    current = sentence
    flush()
    return chunks


def _split_sentences(text: str, max_chars: int) -> list:
    """Грубое деление на предложения; если предложение всё равно длиннее лимита — режем жёстко."""
    sentences = re.split(r"(?<=[.!?…])\s+", text)
    out = []
    for s in sentences:
        if len(s) <= max_chars:
            out.append(s + " ")
        else:
            for i in range(0, len(s), max_chars):
                out.append(s[i:i + max_chars])
    return out


# ---------------------------------------------------------------------------
# ПЕРЕВОД (OpenAI Chat Completions)
# ---------------------------------------------------------------------------
def translate_chunk(chunk: str, target_lang_name: str, model: str,
                    max_retries: int = 5) -> str:
    """Переводит один чанк, с ретраями на сетевые ошибки и 429/5xx."""
    payload = {
        "model": model,
        "temperature": 0.3,
        "max_tokens": 16000,
        "messages": [
            {"role": "system", "content": build_system_prompt(target_lang_name)},
            {"role": "user", "content": chunk},
        ],
    }
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    delay = 2
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(OPENAI_URL, headers=headers, json=payload, timeout=180)
            if resp.status_code == 200:
                data = resp.json()
                return data["choices"][0]["message"]["content"].strip()
            if resp.status_code in (429, 500, 502, 503, 504):
                print(f"    OpenAI {resp.status_code}, retry {attempt}/{max_retries} "
                      f"через {delay}s…", file=sys.stderr)
            else:
                raise RuntimeError(f"OpenAI error {resp.status_code}: {resp.text[:500]}")
        except requests.RequestException as e:
            print(f"    Сетевая ошибка ({e}), retry {attempt}/{max_retries} "
                  f"через {delay}s…", file=sys.stderr)
        time.sleep(delay)
        delay = min(delay * 2, 60)

    raise RuntimeError(f"Не удалось перевести чанк после {max_retries} попыток.")


def translate_text(text: str, lang_code: str, model: str, chunk_size: int) -> str:
    lang_name, label = LANGUAGES[lang_code]
    chunks = split_into_chunks(text, chunk_size)
    print(f"  [{label}] чанков: {len(chunks)} (модель {model})")
    translated_parts = []
    for i, chunk in enumerate(chunks, 1):
        print(f"    → перевод чанка {i}/{len(chunks)} ({len(chunk)} симв.)…")
        translated_parts.append(translate_chunk(chunk, lang_name, model))
    # Абзацные разделители уже внутри чанков, склеиваем встык.
    return "".join(translated_parts).strip() + "\n"


# ---------------------------------------------------------------------------
# ОЗВУЧКА (Voicer API)
# ---------------------------------------------------------------------------
class TTSClient:
    def __init__(self, api_key: str, base_url: str = TTS_BASE_URL):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({"X-API-Key": api_key})

    def _request(self, method: str, path: str, **kwargs):
        """HTTP c ретраями и авто-фолбэком на резервный домен при сетевых сбоях."""
        url = self.base_url + path
        delay = 2
        last_exc = None
        for attempt in range(1, 6):
            try:
                resp = self.session.request(method, url, timeout=120, **kwargs)
                if resp.status_code == 503:
                    print(f"    503 (сервис недоступен), retry через {delay}s…", file=sys.stderr)
                    time.sleep(delay)
                    delay = min(delay * 2, 30)
                    continue
                return resp
            except requests.RequestException as e:
                last_exc = e
                # пробуем резервный домен
                if self.base_url != TTS_BACKUP_URL.rstrip("/"):
                    print(f"    Сетевая ошибка ({e}); переключаюсь на резервный домен…",
                          file=sys.stderr)
                    self.base_url = TTS_BACKUP_URL.rstrip("/")
                    url = self.base_url + path
                print(f"    retry {attempt}/5 через {delay}s…", file=sys.stderr)
                time.sleep(delay)
                delay = min(delay * 2, 30)
        raise RuntimeError(f"TTS запрос не удался: {last_exc}")

    def get_balance(self):
        resp = self._request("GET", "/balance")
        resp.raise_for_status()
        return resp.json()

    def list_templates(self):
        resp = self._request("GET", "/templates")
        resp.raise_for_status()
        return resp.json()

    def create_task(self, text: str, template_uuid=None, template=None,
                    chunk_size=None) -> int:
        body = {"text": text}
        if template_uuid:
            body["template_uuid"] = template_uuid
        elif template:
            body["template"] = template
        if chunk_size:
            body["chunk_size"] = chunk_size
        resp = self._request("POST", "/tasks", json=body)
        if resp.status_code != 200:
            raise RuntimeError(f"POST /tasks -> {resp.status_code}: {resp.text[:500]}")
        data = resp.json()
        print(f"    задача создана: id={data['task_id']} — {data.get('message', '')}")
        return data["task_id"]

    def wait_for_result(self, task_id: int, poll_interval: int = 5,
                        timeout: int = 1800) -> str:
        """Ждёт статус 'ending'. Возвращает статус или бросает исключение при ошибке."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            resp = self._request("GET", f"/tasks/{task_id}/status")
            resp.raise_for_status()
            data = resp.json()
            status = data.get("status")
            if status == "ending":
                return status
            if status in ("error", "error_handled"):
                err = data.get("error") or {}
                raise RuntimeError(
                    f"Задача {task_id} завершилась ошибкой: "
                    f"{err.get('code')} — {err.get('ru') or err.get('en')}"
                )
            print(f"    задача {task_id}: {status} ({data.get('status_label', '')})…")
            time.sleep(poll_interval)
        raise TimeoutError(f"Задача {task_id} не готова за {timeout}s.")

    def download_result(self, task_id: int, out_path_no_ext: str) -> str:
        resp = self._request("GET", f"/tasks/{task_id}/result")
        if resp.status_code != 200:
            raise RuntimeError(f"GET result -> {resp.status_code}: {resp.text[:300]}")
        ext = ".zip" if "zip" in resp.headers.get("Content-Type", "") else ".mp3"
        out_path = out_path_no_ext + ext
        with open(out_path, "wb") as f:
            f.write(resp.content)
        return out_path


def synthesize(client: TTSClient, text: str, out_path_no_ext: str,
               template_uuid=None, voice_id=None, tts_model=DEFAULT_TTS_MODEL,
               chunk_size=2000) -> str:
    """Создаёт TTS-задачу, ждёт готовности и скачивает аудио."""
    template = None
    if not template_uuid:
        template = {
            "model_id": tts_model,
            "voice_id": voice_id or DEFAULT_VOICE_ID,
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75,
                "use_speaker_boost": True,
                "style": 0.0,
                "speed": 1.0,
            },
        }
    task_id = client.create_task(
        text=text,
        template_uuid=template_uuid,
        template=template,
        chunk_size=chunk_size,
    )
    client.wait_for_result(task_id)
    path = client.download_result(task_id, out_path_no_ext)
    print(f"    ✓ аудио сохранено: {path}")
    return path


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Локализация англ. сценария на ru/pt/es + озвучка через Voicer API.")
    p.add_argument("input", nargs="?",
                   default="/Users/aleksandrtomilov/Desktop/ПРОМПТЫ/scenario.txt",
                   help="путь к англоязычному .txt (по умолчанию scenario.txt на рабочем столе)")
    p.add_argument("--langs", default="ru,pt,es",
                   help="языки через запятую (ru,pt,es)")
    p.add_argument("--model", default=OPENAI_MODEL_DEFAULT,
                   help=f"модель OpenAI (по умолчанию {OPENAI_MODEL_DEFAULT})")
    p.add_argument("--chunk-size", type=int, default=8000,
                   help="макс. символов в одном чанке для перевода (по умолчанию 8000)")
    p.add_argument("--out-dir", default=None,
                   help="куда складывать результаты (по умолчанию рядом с входным файлом)")
    p.add_argument("--skip-translate", action="store_true",
                   help="не переводить, использовать уже готовые scenario_<lang>.txt")
    p.add_argument("--skip-tts", action="store_true",
                   help="только перевод, без озвучки")
    p.add_argument("--template-uuid", default=None,
                   help="UUID готового шаблона озвучки (из Telegram-бота)")
    p.add_argument("--voice-id", default=None,
                   help="ElevenLabs voice_id для инлайн-шаблона озвучки")
    p.add_argument("--tts-chunk-size", type=int, default=2000,
                   help="chunk_size для TTS (500-2000, по умолчанию 2000)")
    return p.parse_args()


def main():
    args = parse_args()

    langs = [c.strip() for c in args.langs.split(",") if c.strip()]
    for c in langs:
        if c not in LANGUAGES:
            sys.exit(f"Неизвестный язык: {c}. Доступно: {', '.join(LANGUAGES)}")

    input_path = args.input
    out_dir = args.out_dir or os.path.dirname(os.path.abspath(input_path))
    os.makedirs(out_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(input_path))[0]

    def out_txt(lang):
        return os.path.join(out_dir, f"{base_name}_{lang}.txt")

    def out_audio_base(lang):
        return os.path.join(out_dir, f"{base_name}_{lang}")

    # --- перевод ---
    if not args.skip_translate:
        if not OPENAI_API_KEY:
            sys.exit(
                "Не задан ключ OpenAI. Экспортируйте его перед запуском:\n"
                '    export OPENAI_API_KEY="sk-proj-...ваш ключ..."\n'
                "или впишите в локальную копию строки OPENAI_API_KEY в начале скрипта.\n"
                "(Впрямую в закоммиченный файл его вписать нельзя — GitHub блокирует такой push.)"
            )
        if not os.path.isfile(input_path):
            sys.exit(f"Входной файл не найден: {input_path}")
        with open(input_path, "r", encoding="utf-8") as f:
            source_text = f.read()
        print(f"Прочитан входной файл: {input_path} ({len(source_text)} символов)\n")

        print("=== ПЕРЕВОД ===")
        for lang in langs:
            translated = translate_text(source_text, lang, args.model, args.chunk_size)
            with open(out_txt(lang), "w", encoding="utf-8") as f:
                f.write(translated)
            print(f"  ✓ сохранено: {out_txt(lang)} ({len(translated)} символов)\n")
    else:
        print("=== ПЕРЕВОД ПРОПУЩЕН (--skip-translate) ===\n")

    # --- озвучка ---
    if args.skip_tts:
        print("=== ОЗВУЧКА ПРОПУЩЕНА (--skip-tts) ===")
        return

    print("=== ОЗВУЧКА ===")
    client = TTSClient(TTS_API_KEY)

    # баланс + примерная оценка стоимости
    try:
        bal = client.get_balance()
        total_chars = 0
        for lang in langs:
            if os.path.isfile(out_txt(lang)):
                total_chars += len(open(out_txt(lang), encoding="utf-8").read())
        print(f"  Баланс: {bal.get('balance')} символов. "
              f"Ориентировочно потребуется ~{total_chars} символов на {len(langs)} языка(ов).")
        if isinstance(bal.get("balance"), int) and total_chars > bal["balance"]:
            print("  ⚠ Похоже, баланса не хватит на все языки — пополните или уберите часть языков.",
                  file=sys.stderr)
    except Exception as e:
        print(f"  (не удалось получить баланс: {e})", file=sys.stderr)

    for lang in langs:
        label = LANGUAGES[lang][1]
        path = out_txt(lang)
        if not os.path.isfile(path):
            print(f"  [{label}] нет файла перевода {path}, пропускаю.", file=sys.stderr)
            continue
        text = open(path, encoding="utf-8").read().strip()
        if not text:
            print(f"  [{label}] пустой перевод, пропускаю.", file=sys.stderr)
            continue
        print(f"  [{label}] озвучиваю ({len(text)} символов)…")
        try:
            synthesize(
                client, text, out_audio_base(lang),
                template_uuid=args.template_uuid,
                voice_id=args.voice_id,
                chunk_size=args.tts_chunk_size,
            )
        except Exception as e:
            print(f"  ✗ [{label}] ошибка озвучки: {e}", file=sys.stderr)

    print("\nГотово.")


if __name__ == "__main__":
    main()
