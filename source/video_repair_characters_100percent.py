# -*- coding: utf-8 -*-
"""
ДОБИВКА видео до 100% для связки скриптов с ПЕРСОНАЖАМИ.

Связка (проект /Users/aleksandrtomilov/Desktop/ПРОМПТЫ):
  1) master_prompt_pipeline_...            -> generated_prompts.txt / generated_image_prompts.txt
       (строки сцен с маркерами персонажей [Alias]) + character_generation_state.json.
  2) media_gen_character_image_pipeline    -> ПЕРСОНАЖИ/[Alias].png (портреты-референсы)
       и КАРТИНКИ/0001.png ... (сцены, сгенерированные с учётом референсов персонажей).
  3) video_from_local_images_base64.py     -> ВИДЕО/0001.mp4 (оживление картинок сцены).

Этот скрипт добивает видео, которые НЕ сгенерировались в шаге 3.

Провайдер видео: flower (модель flower-video / Veo 3.1), operation flower_video_from_image.

Важно про персонажей:
  • В видео-шаге референс-картинка персонажа НЕ передаётся — персонаж уже "вшит" в картинку
    сцены. Связь с персонажем живёт в ТЕКСТЕ промпта ([Alias] + имя, напр. "King Solomon...").
    Именно имена (особенно именованные исторические/религиозные фигуры) чаще всего валят
    video-фильтр.
  • Требование: сначала пробуем сгенерировать видео С ПЕРСОНАЖЕМ. Если по какой-то причине
    3 раза подряд не вышло — ОТВЯЗЫВАЕМ сцену от персонажа (убираем имя/маркер, анонимизируем)
    и продолжаем. Дальше — эскалация до гарантированного "оживить только движением".

Ступени для каждой пропущенной сцены:
  A. Если у сцены есть персонаж(и):
       попытка 1: промпт с персонажем как есть;
       попытка 2: с персонажем, но без сцен насилия (санитайз, имена сохранены);
       попытка 3: с персонажем, усиленный санитайз (имена сохранены);
       -> если все 3 не вышли, ОТВЯЗКА персонажа.
  B. Отвязанные / нейтральные ступени (и вход для сцен без персонажа):
       - анонимизированная сцена (имена персонажей заменены на "an anonymous person");
       - + санитайз насилия;
       - motion-preserve (только движение, кадр без изменений);
       - motion-minimal (гарантированно проходит фильтр).
  + Доп. заходы motion-minimal на случай сетевых сбоев -> итог 100%.

Безопасность:
  • FAST_GEN_API_KEY — ключ media_gen (из окружения, не хардкодим).
  • OPENAI_API_KEY — опционально, для умного переписывания промптов.

Запуск:
   export FAST_GEN_API_KEY="ТВОЙ_КЛЮЧ"
   export OPENAI_API_KEY="sk-..."          # опционально
   python video_repair_characters_100percent.py
   python video_repair_characters_100percent.py --dry-run
   python video_repair_characters_100percent.py --limit 20
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import re
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests


# =========================================================
# НАСТРОЙКИ
# =========================================================

API_KEY = (
    os.getenv("FAST_GEN_API_KEY")
    or os.getenv("FASTGEN_API_KEY")
    or os.getenv("MEDIA_GEN_API_KEY")
    or ""
)
BASE_URL = os.getenv("FAST_GEN_API_BASE", "https://api.fast-gen.ai")

BASE_DIR = Path(os.getenv("BASE_FOLDER", "/Users/aleksandrtomilov/Desktop/ПРОМПТЫ"))
IMAGES_DIR = BASE_DIR / "КАРТИНКИ"
VIDEOS_DIR = BASE_DIR / "ВИДЕО"

# Файлы промптов (те же, что читает video_from_local_images_base64.py) — со строками [Alias].
PROMPTS_FILE_PRIMARY = BASE_DIR / "generated_prompts_detailed.txt"
PROMPTS_FILE_FALLBACK = BASE_DIR / "generated_prompts.txt"

# State персонажей из шага 2 (alias -> {path, file_hash, ...}) — используем только чтобы
# лучше знать список имён персонажей для анонимизации.
CHARACTER_STATE_FILE = BASE_DIR / "character_generation_state.json"

LOG_FILE = BASE_DIR / "video_repair_characters_log.json"
DETACHED_FILE = BASE_DIR / "video_repair_detached_characters.json"
RATE_LIMIT_FILE = BASE_DIR / "video_rate_limit_state.json"

REQUEST_TIMEOUT = int(os.getenv("FAST_GEN_REQUEST_TIMEOUT", "300"))
POLL_INTERVAL = int(os.getenv("FAST_GEN_OPERATION_POLL_SEC", "20"))
POLL_TIMEOUT = int(os.getenv("FAST_GEN_POLL_TIMEOUT", str(60 * 60 * 4)))
RETRY_DELAY_SEC = int(os.getenv("FAST_GEN_RETRY_DELAY_SEC", "8"))
MAX_HTTP_ATTEMPTS = max(1, int(os.getenv("FAST_GEN_MAX_ATTEMPTS", "4")))

# Сколько раз пробуем сгенерировать видео С ПЕРСОНАЖЕМ перед отвязкой.
CHARACTER_ATTEMPTS = max(1, int(os.getenv("FAST_GEN_CHARACTER_ATTEMPTS", "3")))
# Доп. заходы на гарантированном motion-minimal (страховка от сети/перегрузки провайдера).
EXTRA_GUARANTEED_ROUNDS = int(os.getenv("FAST_GEN_EXTRA_GUARANTEED_ROUNDS", "6"))

# Провайдер часто отвечает "Generation failed, please try again later" — это ВРЕМЕННАЯ
# перегрузка, НЕ safety-блок. Такое повторяем на ТОМ ЖЕ промпте с нарастающей паузой,
# а не эскалируем и не отвязываем персонажа. Каждая "попытка" = до TRANSIENT_RETRIES под-повторов.
TRANSIENT_RETRIES = max(1, int(os.getenv("FAST_GEN_TRANSIENT_RETRIES", "5")))
BACKOFF_CAP_SEC = int(os.getenv("FAST_GEN_BACKOFF_CAP_SEC", "90"))

MAX_VIDEO_WORKERS = int(os.getenv("FAST_GEN_VIDEO_WORKERS", "6"))

ASPECT_RATIO = os.getenv("FAST_GEN_VIDEO_ASPECT_RATIO", "16:9")
# Canonical V6 operation id для image->video на flower (Veo 3.1).
# См. GET /api/v6/capabilities — flower_video_from_image, provider=flower,
# model=flower-video, 1 кредит.
VIDEO_OPERATION = os.getenv("FAST_GEN_VIDEO_OPERATION", "flower_video_from_image")
# Необязательный shorthand-model. По умолчанию flower-video (Veo 3.1).
VIDEO_MODEL = os.getenv("FAST_GEN_VIDEO_MODEL") or None

_SEED_ENV = os.getenv("FAST_GEN_SEED", "").strip()
GENERATION_SEED: Optional[int] = int(_SEED_ENV) if _SEED_ENV.lstrip("-").isdigit() else None
_DURATION_ENV = os.getenv("FAST_GEN_VIDEO_DURATION_SECONDS", "").strip()
VIDEO_DURATION_SECONDS: Optional[int] = int(_DURATION_ENV) if _DURATION_ENV.isdigit() else None
VIDEO_RESOLUTION = os.getenv("FAST_GEN_VIDEO_RESOLUTION") or None
# ВНИМАНИЕ: ultra и keyframes — только flow-video, для flower не используются.

MAX_IMAGE_BYTES = 5 * 1024 * 1024
MIN_VIDEO_BYTES = int(os.getenv("FAST_GEN_MIN_VIDEO_BYTES", "1024"))

MAX_VIDEO_STARTS_PER_HOUR = int(os.getenv("FAST_GEN_MAX_VIDEO_STARTS_PER_HOUR", "150"))
RATE_WINDOW_SECONDS = 3600

PROMPT_MODEL = os.getenv("PROMPT_MODEL", "gpt-4o-mini")

V6_GENERATIONS_ENDPOINT = "/api/v6/generations"
V6_GENERATION_STATUS_ENDPOINT = "/api/v6/generations/{generation_id}"

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}

DEFAULT_ANIMATION_PROMPT = (
    "Animate this image with gentle realistic motion, subtle cinematic camera movement, "
    "natural details, preserve the original composition, subject, lighting and style."
)

MOTION_PRESERVE_PROMPT = (
    "Animate the existing scene in this image with subtle, natural, realistic motion. "
    "Preserve exactly the same subject, composition, colors, lighting and style. "
    "Add only gentle ambient movement and a slow cinematic camera push-in. "
    "Do not add, remove or change any content. No text, no logos, no watermark."
)
MOTION_MINIMAL_PROMPT = (
    "Gently animate this image with very subtle motion and a slow calm camera push-in, "
    "keeping everything exactly as it is. No new elements. No text, no watermark."
)

PERMANENT_ERROR_MARKERS = (
    "safety", "blocked", "moderation", "content policy",
    "prohibited", "not allowed", "violat",
)

RISKY_REPLACEMENTS = [
    (re.compile(r"\bblood[a-z]*\b", re.I), ""),
    (re.compile(r"\bgore?\b", re.I), ""),
    (re.compile(r"\bgory\b", re.I), ""),
    (re.compile(r"\bkill(?:s|ing|ed|er|ers)?\b", re.I), "confronting"),
    (re.compile(r"\bmurder(?:s|ing|ed)?\b", re.I), "confronting"),
    (re.compile(r"\bslaughter(?:s|ing|ed)?\b", re.I), "gathering"),
    (re.compile(r"\bbehead(?:s|ing|ed)?\b", re.I), "facing"),
    (re.compile(r"\bexecut(?:e|ion|ing|ed)\b", re.I), "ceremony"),
    (re.compile(r"\bcorpse(?:s)?\b", re.I), "figure"),
    (re.compile(r"\bdead body\b", re.I), "resting figure"),
    (re.compile(r"\bdead\b", re.I), "still"),
    (re.compile(r"\bdeath\b", re.I), "stillness"),
    (re.compile(r"\bwound(?:s|ed|ing)?\b", re.I), ""),
    (re.compile(r"\bfight(?:s|ing)?\b", re.I), "gathering"),
    (re.compile(r"\bbattle(?:s|field)?\b", re.I), "encampment"),
    (re.compile(r"\bwar(?:s|riors?|fare)?\b", re.I), "assembly"),
    (re.compile(r"\baggressive(?:ly)?\b", re.I), "solemn"),
    (re.compile(r"\bviolen(?:t|tly|ce)\b", re.I), "solemn"),
    (re.compile(r"\bweapon(?:s)?\b", re.I), "tools"),
    (re.compile(r"\bsword(?:s)?\b", re.I), "staff"),
    (re.compile(r"\bspear(?:s)?\b", re.I), "staff"),
    (re.compile(r"\bknife\b|\bknives\b", re.I), "tool"),
    (re.compile(r"\bblade(?:s)?\b", re.I), "tool"),
    (re.compile(r"\btorture(?:s|d|ing)?\b", re.I), "hardship"),
    (re.compile(r"\bsuffer(?:s|ing|ed)?\b", re.I), ""),
    (re.compile(r"\bnaked\b", re.I), "modestly clothed"),
    (re.compile(r"\bnude\b", re.I), "modestly clothed"),
]

SAFE_SUFFIX = (
    "fully clothed, modest, non-sexualized, calm documentary reconstruction, "
    "no violence, no gore, no weapons in use, respectful historical scene"
)

# Известные имена/титулы, которые надо анонимизировать при отвязке (в дополнение к [Alias]).
KNOWN_FIGURE_NAMES = {
    "jesus", "jesus christ", "christ", "messiah", "saviour", "savior", "son of god",
    "solomon", "king solomon", "david", "king david", "moses", "abraham", "noah",
    "mary", "virgin mary", "mother mary", "joseph", "peter", "paul", "john the baptist",
}
HONORIFICS = (
    r"(?:king|queen|saint|st\.?|prophet|lord|rabbi|apostle|pharaoh|emperor|"
    r"prince|princess|holy|blessed|the)"
)


# =========================================================
# LOCKS / ERRORS
# =========================================================

print_lock = threading.Lock()
results_lock = threading.Lock()
rate_limit_lock = threading.Lock()


class PermanentError(Exception):
    """safety-фильтр / 4xx — повторять с тем же контентом бессмысленно."""


class TransientError(Exception):
    """сеть / 429 / 5xx — можно повторить."""


def log(message: str) -> None:
    with print_lock:
        print(message, flush=True)


def classify_error_message(message: str) -> bool:
    low = str(message).lower()
    return any(marker in low for marker in PERMANENT_ERROR_MARKERS)


# =========================================================
# УТИЛИТЫ
# =========================================================

def normalize_base_url(url: str) -> str:
    return url.rstrip("/")


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def safe_json(resp: requests.Response) -> Dict[str, Any]:
    try:
        return resp.json()
    except Exception:
        return {"raw_text": resp.text}


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip(" ,")


def infer_mime_type(path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    return mime or "image/png"


def path_sort_key(path: Path):
    stem = path.stem
    if stem.isdigit():
        return (0, int(stem), "")
    return (1, 0, stem.lower())


# =========================================================
# PROMPTS + ПЕРСОНАЖИ
# =========================================================

@dataclass
class Scene:
    index: int          # порядковый индекс сцены (= номер строки промпта = stem картинки)
    aliases: List[str] = field(default_factory=list)
    body: str = ""      # текст сцены без [Alias]-скобок (имена внутри текста сохранены)


def parse_prompt_line(line: str) -> Tuple[List[str], str]:
    """Возвращает (aliases, body_без_скобок)."""
    m = re.match(r"^\[([^\]]+)\]\s*(.*)$", line)
    if m:
        aliases = [a.strip() for a in m.group(1).split(",") if a.strip()]
        return aliases, m.group(2).strip()
    return [], line.strip()


def load_scenes_map() -> Dict[int, Scene]:
    """Индекс сцены -> Scene. Индексация как в video_from_local_images_base64.py:
    пропускаем пустые и ### строки, порядковый счётчик 1,2,3..."""
    prompts_file = None
    if PROMPTS_FILE_PRIMARY.exists():
        prompts_file = PROMPTS_FILE_PRIMARY
    elif PROMPTS_FILE_FALLBACK.exists():
        prompts_file = PROMPTS_FILE_FALLBACK
    if not prompts_file:
        log("[INFO] Файл промптов не найден — сцены без персонажей, DEFAULT-анимация.")
        return {}

    scenes: Dict[int, Scene] = {}
    idx = 0
    for raw in prompts_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("###"):
            continue
        idx += 1
        aliases, body = parse_prompt_line(line)
        scenes[idx] = Scene(index=idx, aliases=aliases, body=body)
    log(f"[INFO] Файл промптов: {prompts_file.name} ({len(scenes)} сцен, "
        f"с персонажами: {sum(1 for s in scenes.values() if s.aliases)})")
    return scenes


def load_known_names() -> set:
    names = set(KNOWN_FIGURE_NAMES)
    state = load_json(CHARACTER_STATE_FILE, {})
    if isinstance(state, dict):
        for alias in state.keys():
            a = str(alias).strip("[] ").strip().lower()
            if a:
                names.add(a)
    return names


def list_images() -> List[Path]:
    if not IMAGES_DIR.exists():
        return []
    images = [p for p in IMAGES_DIR.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    images.sort(key=path_sort_key)
    return images


def scene_index_for(image_path: Path, order: int) -> int:
    return int(image_path.stem) if image_path.stem.isdigit() else order


# =========================================================
# ПЕРЕПИСЫВАНИЕ / АНОНИМИЗАЦИЯ
# =========================================================

def remove_brackets(text: str) -> str:
    return clean_text(re.sub(r"\[[^\]]*\]", "", text))


def heuristic_sanitize(text: str, level: int) -> str:
    out = text
    for pattern, repl in RISKY_REPLACEMENTS:
        out = pattern.sub(repl, out)
    out = clean_text(out)
    if level >= 2:
        out = f"Calm, respectful, non-violent documentary reconstruction. {out}. {SAFE_SUFFIX}."
    return clean_text(out)


def anonymize(body: str, aliases: List[str], known_names: set) -> str:
    """Убирает имена персонажей (alias + известные фигуры) -> 'an anonymous person'."""
    text = remove_brackets(body)
    # Собираем все имена: alias из строки + известные фигуры, встречающиеся в тексте.
    names = set()
    for a in aliases:
        a = a.strip()
        if a:
            names.add(a)
    low = text.lower()
    for n in known_names:
        if n and n in low:
            names.add(n)
    # Заменяем длинные имена раньше коротких ("king solomon" перед "solomon").
    for name in sorted(names, key=len, reverse=True):
        esc = re.escape(name)
        text = re.sub(rf"\b{HONORIFICS}\s+{esc}\b", "an anonymous person", text, flags=re.I)
        text = re.sub(rf"\b{esc}\b", "an anonymous person", text, flags=re.I)
    # Убираем оставшиеся "character reference for ..." и дубли.
    text = re.sub(r"\bcharacter reference for\b[^,]*", "", text, flags=re.I)
    text = re.sub(r"(an anonymous person)(\s*,?\s*(an anonymous person))+", r"\1", text, flags=re.I)
    text = clean_text(text)
    return text or "an anonymous historical person in a calm documentary scene"


_openai_lock = threading.Lock()
_openai_cache: List[Any] = []


def get_openai_client():
    with _openai_lock:
        if _openai_cache:
            return _openai_cache[0]
        client = None
        if os.getenv("OPENAI_API_KEY"):
            try:
                from openai import OpenAI
                client = OpenAI()
            except Exception as e:
                log(f"[WARN] OpenAI недоступен ({e}); переписываю эвристикой.")
        _openai_cache.append(client)
        return client


def _llm(messages) -> Optional[str]:
    client = get_openai_client()
    if client is None:
        return None
    try:
        response = client.chat.completions.create(
            model=PROMPT_MODEL, temperature=0.4, messages=messages,
        )
        return clean_text(response.choices[0].message.content or "") or None
    except Exception as e:
        log(f"[WARN] LLM-переписывание не удалось: {e}")
        return None


def rewrite_keep_character(body: str) -> str:
    """Санитайз насилия, но персонаж и его имя сохранены."""
    out = _llm([
        {"role": "system", "content": "You edit video-animation prompts to pass strict content filters "
                                       "while staying visual and documentary-realistic."},
        {"role": "user", "content": "Rewrite this prompt so it safely passes strict content moderation. "
                                     "Remove any violence, gore, blood, killing, weapons in use, torture or "
                                     "distressing content, but KEEP the same character(s), their name(s) and "
                                     "identity, and keep it a calm respectful historical documentary scene. "
                                     "People must be fully clothed and non-sexualized. "
                                     "Return ONLY the rewritten prompt in English, no quotes.\n\n" + body},
    ])
    return out or heuristic_sanitize(body, 2)


def rewrite_anonymize(body: str, aliases: List[str], known_names: set) -> str:
    """Отвязка от персонажа через LLM (или эвристику)."""
    out = _llm([
        {"role": "system", "content": "You edit prompts so they contain NO named real person and pass strict "
                                       "content filters, while staying a calm documentary scene."},
        {"role": "user", "content": "Rewrite this prompt so that there is NO specific named person. "
                                     "Replace any personal name, especially any historical or religious figure, "
                                     "with 'an anonymous person'. Remove honorific titles. "
                                     "Remove any violence, gore, weapons in use or distressing content. "
                                     "Keep it a calm, respectful, fully-clothed historical documentary scene. "
                                     "Return ONLY the rewritten prompt in English, no quotes.\n\n" + body},
    ])
    return out or heuristic_sanitize(anonymize(body, aliases, known_names), 2)


# =========================================================
# IMAGE -> DATA URI / VIDEO SAVE
# =========================================================

def image_to_data_uri(image_path: Path) -> str:
    size = image_path.stat().st_size
    if size > MAX_IMAGE_BYTES:
        raise PermanentError(
            f"Картинка слишком большая для inline data URI: {image_path.name} "
            f"({size/1024/1024:.2f} MB > 5 MB)."
        )
    raw = image_path.read_bytes()
    return f"data:{infer_mime_type(image_path)};base64,{base64.b64encode(raw).decode('ascii')}"


def recursive_find_strings(obj: Any) -> List[str]:
    found: List[str] = []

    def _walk(x: Any) -> None:
        if isinstance(x, dict):
            for v in x.values():
                _walk(v)
        elif isinstance(x, list):
            for v in x:
                _walk(v)
        elif isinstance(x, str):
            found.append(x)

    _walk(obj)
    return found


def extract_video_source(data: Dict[str, Any]) -> str:
    results = data.get("results")
    if isinstance(results, list):
        for item in results:
            if not isinstance(item, dict):
                continue
            inline = item.get("data")
            if isinstance(inline, str) and inline.startswith("data:video/"):
                return inline
            url = item.get("download_url")
            if isinstance(url, str) and url:
                return url
    for s in recursive_find_strings(data):
        if s.startswith("data:video/"):
            return s
    for s in recursive_find_strings(data):
        if re.match(r"^https?://.+", s, flags=re.IGNORECASE):
            return s
    raise TransientError(f"Не найден video source: {json.dumps(data, ensure_ascii=False)[:1200]}")


def save_video_from_source(source: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if source.startswith("data:video/"):
        _, b64 = source.split(",", 1)
        out_path.write_bytes(base64.b64decode(b64))
    elif source.startswith(("http://", "https://")):
        with requests.get(source, stream=True, headers={"X-API-Key": API_KEY}, timeout=REQUEST_TIMEOUT) as r:
            if 400 <= r.status_code < 500:
                raise PermanentError(f"HTTP {r.status_code} при скачивании видео")
            r.raise_for_status()
            with out_path.open("wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
    else:
        raise TransientError(f"Неподдерживаемый video source: {source[:200]}")
    if not out_path.exists() or out_path.stat().st_size < MIN_VIDEO_BYTES:
        raise TransientError(f"Видео не сохранилось/слишком маленькое: {out_path}")


# =========================================================
# HTTP + V6
# =========================================================

def request_post(url: str, payload: dict) -> dict:
    last_error: Optional[Exception] = None
    for attempt in range(1, MAX_HTTP_ATTEMPTS + 1):
        try:
            resp = requests.post(
                url, headers={"X-API-Key": API_KEY, "Content-Type": "application/json"},
                json=payload, timeout=REQUEST_TIMEOUT,
            )
            if resp.status_code == 429:
                raise TransientError(f"429: {resp.text[:200]}")
            if 400 <= resp.status_code < 500:
                raise PermanentError(f"HTTP {resp.status_code}: {resp.text[:200]}")
            if resp.status_code >= 500:
                raise TransientError(f"HTTP {resp.status_code}: {resp.text[:200]}")
            return safe_json(resp)
        except PermanentError:
            raise
        except KeyboardInterrupt:
            raise
        except Exception as e:
            last_error = e
            if attempt < MAX_HTTP_ATTEMPTS:
                time.sleep(RETRY_DELAY_SEC)
    raise TransientError(f"POST не удался: {last_error}")


def request_get(url: str) -> dict:
    last_error: Optional[Exception] = None
    for attempt in range(1, MAX_HTTP_ATTEMPTS + 1):
        try:
            resp = requests.get(url, headers={"X-API-Key": API_KEY}, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 429:
                raise TransientError(f"429: {resp.text[:200]}")
            if 400 <= resp.status_code < 500:
                raise PermanentError(f"HTTP {resp.status_code}: {resp.text[:200]}")
            if resp.status_code >= 500:
                raise TransientError(f"HTTP {resp.status_code}: {resp.text[:200]}")
            return safe_json(resp)
        except PermanentError:
            raise
        except KeyboardInterrupt:
            raise
        except Exception as e:
            last_error = e
            if attempt < MAX_HTTP_ATTEMPTS:
                time.sleep(RETRY_DELAY_SEC)
    raise TransientError(f"GET не удался: {last_error}")


def extract_generation_id(data: Dict[str, Any]) -> Optional[str]:
    for c in (data.get("id"), data.get("operation_id")):
        if isinstance(c, str) and c.strip():
            return c.strip()
    if isinstance(data.get("data"), dict):
        c = data["data"].get("id")
        if isinstance(c, str) and c.strip():
            return c.strip()
    return None


def build_payload(prompt: str, image_data_uri: str) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "operation": VIDEO_OPERATION,
        "prompt": prompt,
        "aspect_ratio": ASPECT_RATIO,
        "inputs": [image_data_uri],
    }
    if VIDEO_MODEL:
        payload["model"] = VIDEO_MODEL
    if GENERATION_SEED is not None:
        payload["seed"] = GENERATION_SEED
    if VIDEO_DURATION_SECONDS is not None:
        payload["duration_seconds"] = VIDEO_DURATION_SECONDS
    if VIDEO_RESOLUTION:
        payload["resolution"] = VIDEO_RESOLUTION
    # ВНИМАНИЕ: ultra и keyframes — только flow-video, для flower не отправляем.
    return payload


def poll_generation(generation_id: str, tag: str) -> Dict[str, Any]:
    url = normalize_base_url(BASE_URL) + V6_GENERATION_STATUS_ENDPOINT.format(generation_id=generation_id)
    started = time.time()
    while True:
        if time.time() - started > POLL_TIMEOUT:
            raise TransientError(f"{tag}: {generation_id} не завершилась за {POLL_TIMEOUT}s")
        data = request_get(url)
        status = str(data.get("status") or "").lower()
        if status in ("queued", "running", ""):
            time.sleep(POLL_INTERVAL)
            continue
        if status in ("succeeded", "success", "completed", "done"):
            return data
        if status in ("failed", "error", "cancelled", "canceled"):
            error_str = str(data.get("error") or data)
            if classify_error_message(error_str):
                raise PermanentError(f"{tag} заблокировано: {error_str[:300]}")
            raise TransientError(f"{tag} ошибка: {error_str[:300]}")
        time.sleep(POLL_INTERVAL)


def generate_once(prompt: str, image_data_uri: str, out_path: Path) -> str:
    """Одна попытка: старт + ожидание + сохранение. Кидает Permanent/Transient."""
    video_rate_limiter.acquire(out_path.name)
    url = normalize_base_url(BASE_URL) + V6_GENERATIONS_ENDPOINT
    data = request_post(url, build_payload(prompt, image_data_uri))
    generation_id = extract_generation_id(data)
    if not generation_id:
        raise PermanentError(f"API не вернул generation id: {data}")
    result = poll_generation(generation_id, out_path.name)
    save_video_from_source(extract_video_source(result), out_path)
    return generation_id


def generate_attempt(prompt: str, image_data_uri: str, out_path: Path,
                     retries: Optional[int] = None) -> str:
    """Одна "попытка" уровня, устойчивая к ВРЕМЕННЫМ ошибкам провайдера.

    "Generation failed, please try again later" и сетевые сбои повторяются на ТОМ ЖЕ
    промпте с нарастающей паузой (до `retries` под-повторов). PermanentError (safety-блок)
    пробрасывается сразу — это сигнал эскалировать/отвязать персонажа.
    """
    retries = retries or TRANSIENT_RETRIES
    delay = float(RETRY_DELAY_SEC)
    last_error: Optional[Exception] = None
    for i in range(1, retries + 1):
        try:
            return generate_once(prompt, image_data_uri, out_path)
        except PermanentError:
            raise
        except KeyboardInterrupt:
            raise
        except Exception as e:
            last_error = e
            if i < retries:
                log(f"      временная ошибка {i}/{retries}: {e} — повтор через {int(delay)}с")
                time.sleep(delay)
                delay = min(delay * 1.7, BACKOFF_CAP_SEC)
    raise TransientError(str(last_error))


# =========================================================
# RATE LIMITER (150/час, состояние на диске)
# =========================================================

class HourlyRateLimiter:
    def __init__(self, path: Path, max_events: int, window_seconds: int):
        self.path = path
        self.max_events = max_events
        self.window_seconds = window_seconds
        self.events: deque = deque()
        data = load_json(path, [])
        now = time.time()
        if isinstance(data, list):
            for ts in data:
                try:
                    ts = float(ts)
                    if now - ts < window_seconds:
                        self.events.append(ts)
                except Exception:
                    pass

    def _save(self) -> None:
        try:
            save_json(self.path, list(self.events))
        except Exception:
            pass

    def acquire(self, tag: str) -> None:
        while True:
            with rate_limit_lock:
                now = time.time()
                while self.events and (now - self.events[0] >= self.window_seconds):
                    self.events.popleft()
                if len(self.events) < self.max_events:
                    self.events.append(now)
                    self._save()
                    return
                wait_time = max(1, int(self.window_seconds - (now - self.events[0])))
            log(f"[RATE-WAIT] {tag}: лимит {self.max_events}/час, жду {min(wait_time, 60)} сек")
            time.sleep(min(wait_time, 60))


video_rate_limiter = HourlyRateLimiter(RATE_LIMIT_FILE, MAX_VIDEO_STARTS_PER_HOUR, RATE_WINDOW_SECONDS)


# =========================================================
# ДОБИВКА ОДНОЙ СЦЕНЫ
# =========================================================

def repair_one(image_path: Path, scene: Optional[Scene], known_names: set) -> dict:
    out_path = VIDEOS_DIR / f"{image_path.stem}.mp4"
    aliases = list(scene.aliases) if scene else []
    body = (scene.body if scene and scene.body else "").strip()
    original_prompt = remove_brackets(body) if body else DEFAULT_ANIMATION_PROMPT

    log(f"[REPAIR] {image_path.name} -> {out_path.name} "
        f"({'персонаж: ' + ', '.join(aliases) if aliases else 'без персонажа'})")

    try:
        image_data_uri = image_to_data_uri(image_path)
    except PermanentError as e:
        log(f"    IMAGE PROBLEM {image_path.name}: {e}")
        return {"image": str(image_path), "output": str(out_path), "status": "failed",
                "reason": "image_too_big", "error": str(e), "aliases": aliases}

    detached = False

    # ---- Фаза A: пытаемся С ПЕРСОНАЖЕМ (до CHARACTER_ATTEMPTS раз) ----
    if aliases:
        for a in range(1, CHARACTER_ATTEMPTS + 1):
            if a == 1:
                prompt = original_prompt
            elif a == 2:
                prompt = heuristic_sanitize(original_prompt, 1)   # имена сохранены
            else:
                prompt = rewrite_keep_character(original_prompt)  # LLM/усиленный, имена сохранены
            log(f"    [С ПЕРСОНАЖЕМ] попытка {a}/{CHARACTER_ATTEMPTS}")
            try:
                gen = generate_attempt(prompt, image_data_uri, out_path)
                log(f"    OK (с персонажем, попытка {a}): {out_path.name}")
                return {"image": str(image_path), "output": str(out_path), "status": "success",
                        "mode": "with_character", "character_attempt": a, "detached": False,
                        "aliases": aliases, "prompt": prompt, "generation_id": gen}
            except PermanentError as e:
                log(f"    [С ПЕРСОНАЖЕМ] попытка {a} заблокирована: {e}")
            except TransientError as e:
                log(f"    [С ПЕРСОНАЖЕМ] попытка {a} не удалась: {e}")
        detached = True
        log(f"    3 неудачи подряд -> ОТВЯЗЫВАЮ от персонажа ({', '.join(aliases)})")

    # ---- Фаза B: отвязанные / нейтральные ступени (+ вход для сцен без персонажа) ----
    if aliases:
        detached_base = rewrite_anonymize(original_prompt, aliases, known_names)
    else:
        detached_base = original_prompt

    ladder = [
        ("detached" if aliases else "original", detached_base),
        ("detached-sanitized" if aliases else "sanitized", heuristic_sanitize(detached_base, 2)),
        ("motion-preserve", MOTION_PRESERVE_PROMPT),
        ("motion-minimal", MOTION_MINIMAL_PROMPT),
    ]
    for mode, prompt in ladder:
        log(f"    [{mode}]")
        try:
            gen = generate_attempt(prompt, image_data_uri, out_path)
            log(f"    OK ({mode}): {out_path.name}")
            return {"image": str(image_path), "output": str(out_path), "status": "success",
                    "mode": mode, "detached": detached or bool(aliases), "aliases": aliases,
                    "prompt": prompt, "generation_id": gen}
        except PermanentError as e:
            log(f"    [{mode}] заблокировано: {e} — эскалирую")
        except TransientError as e:
            log(f"    [{mode}] не удалось: {e} — эскалирую")

    # ---- Последний рубеж: motion-minimal несколько раз (страховка от перегрузки/сети) ----
    # Каждый заход сам переживает временные ошибки; ловим ещё и Permanent на случай
    # редкого интермиттент-safety от перегруженного провайдера.
    for extra in range(1, EXTRA_GUARANTEED_ROUNDS + 1):
        log(f"    [motion-minimal доп.заход {extra}/{EXTRA_GUARANTEED_ROUNDS}]")
        try:
            gen = generate_attempt(MOTION_MINIMAL_PROMPT, image_data_uri, out_path)
            log(f"    OK (доп {extra}): {out_path.name}")
            return {"image": str(image_path), "output": str(out_path), "status": "success",
                    "mode": "motion-minimal", "detached": detached or bool(aliases),
                    "aliases": aliases, "prompt": MOTION_MINIMAL_PROMPT, "generation_id": gen}
        except Exception as e:
            log(f"    доп.заход {extra} не удался: {e}")

    log(f"    FAILED окончательно: {out_path.name}")
    return {"image": str(image_path), "output": str(out_path), "status": "failed",
            "reason": "exhausted", "aliases": aliases, "detached": detached,
            "error": "не удалось даже motion-minimal (проверь сеть/ключ/лимиты)"}


# =========================================================
# MAIN
# =========================================================

results_log: List[Dict[str, Any]] = []


def append_result(item: Dict[str, Any]) -> None:
    with results_lock:
        results_log.append(item)
        results_log.sort(key=lambda x: str(x.get("output", "")))
        save_json(LOG_FILE, results_log)


def existing_video(image_path: Path, videos_dir: Path) -> bool:
    out_path = videos_dir / f"{image_path.stem}.mp4"
    try:
        return out_path.exists() and out_path.stat().st_size >= MIN_VIDEO_BYTES
    except OSError:
        return False


def main() -> None:
    global BASE_URL, IMAGES_DIR, VIDEOS_DIR, results_log

    parser = argparse.ArgumentParser(
        description="Добивка видео до 100% (image->video flower/Veo 3.1) с логикой персонажей: 3 попытки с персонажем, затем отвязка."
    )
    parser.add_argument("--images-dir", default=str(IMAGES_DIR))
    parser.add_argument("--videos-dir", default=str(VIDEOS_DIR))
    parser.add_argument("--api-base", default=BASE_URL)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="Максимум пропусков за запуск (0 = все).")
    args = parser.parse_args()

    if not API_KEY:
        raise SystemExit('Не найден API ключ. export FAST_GEN_API_KEY="ТВОЙ_КЛЮЧ"')

    BASE_URL = args.api_base
    IMAGES_DIR = Path(args.images_dir).expanduser()
    VIDEOS_DIR = Path(args.videos_dir).expanduser()
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

    images = list_images()
    if not images:
        raise SystemExit(f"В папке нет картинок: {IMAGES_DIR}")

    scenes_map = load_scenes_map()
    known_names = load_known_names()

    missing: List[Tuple[Path, Optional[Scene]]] = []
    for order, image_path in enumerate(images, start=1):
        if existing_video(image_path, VIDEOS_DIR):
            continue
        scene = scenes_map.get(scene_index_for(image_path, order)) or scenes_map.get(order)
        missing.append((image_path, scene))

    with_char = sum(1 for _p, s in missing if s and s.aliases)

    log("ДОБИВКА видео с логикой персонажей (image->video flower / Veo 3.1)")
    log(f"BASE_URL: {BASE_URL}")
    log(f"Картинки: {IMAGES_DIR}")
    log(f"Видео:    {VIDEOS_DIR}")
    log(f"Video operation: {VIDEO_OPERATION} (flower / flower-video / Veo 3.1)")
    log(f"OpenAI переписывание: {'ДА' if os.getenv('OPENAI_API_KEY') else 'нет (эвристика)'}")
    log(f"Попыток с персонажем до отвязки: {CHARACTER_ATTEMPTS}")
    log(f"Всего картинок: {len(images)}; уже с видео: {len(images) - len(missing)}; пропущено: {len(missing)}")
    log(f"Из пропусков с персонажем: {with_char}")

    if missing:
        preview = ", ".join(p.stem for p, _s in missing[:30])
        more = "" if len(missing) <= 30 else f" … (+{len(missing) - 30})"
        log(f"Индексы пропусков: {preview}{more}")

    if not missing:
        log("Всё на месте — 100% ✅")
        return
    if args.dry_run:
        log("[DRY-RUN] генерацию не запускаю.")
        return
    if args.limit and args.limit > 0:
        missing = missing[:args.limit]
        log(f"[LIMIT] Обрабатываю первые {len(missing)} пропусков.")

    loaded = load_json(LOG_FILE, [])
    results_log = loaded if isinstance(loaded, list) else []

    workers = max(1, min(MAX_VIDEO_WORKERS, len(missing)))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(repair_one, img, scene, known_names): img for img, scene in missing}
        for future in as_completed(futures):
            try:
                append_result(future.result())
            except Exception as e:
                log(f"[FUTURE-ERROR] {e}")

    # Список отвязанных сцен -> отдельный файл.
    detached_list = [
        {"image": r.get("image"), "output": r.get("output"), "aliases": r.get("aliases"), "mode": r.get("mode")}
        for r in results_log if r.get("status") == "success" and r.get("detached")
    ]
    if detached_list:
        save_json(DETACHED_FILE, detached_list)

    ok = sum(1 for r in results_log if r.get("status") == "success")
    fail = sum(1 for r in results_log if r.get("status") != "success")
    still_missing = [p for p in list_images() if not existing_video(p, VIDEOS_DIR)]

    log("\n========== ИТОГ ==========")
    log(f"  Успехов в логе: {ok}, неудач: {fail}")
    log(f"  Отвязано от персонажа: {len(detached_list)}"
        + (f" (см. {DETACHED_FILE.name})" if detached_list else ""))
    log(f"  Итоговое покрытие: {len(list_images()) - len(still_missing)}/{len(list_images())}")
    if not still_missing:
        log("  ГОТОВО: 100% видео ✅")
    else:
        preview = ", ".join(p.stem for p in still_missing[:30])
        log(f"  Осталось пропусков: {len(still_missing)} ({preview}). Обычно сеть/лимиты — запусти ещё раз.")


if __name__ == "__main__":
    main()
