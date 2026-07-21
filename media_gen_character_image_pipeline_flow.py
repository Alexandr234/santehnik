from __future__ import annotations

import base64
import json
import os
import re
import struct
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

import requests

# =========================
# CONFIG
# =========================
# Переработано под media_gen_api V6 (openapi 3.1.0):
#   • Единый endpoint POST /api/v6/generations (GenerationCreateRequest):
#       operation, prompt, aspect_ratio (n:n, напр. 16:9 | 9:16 | 1:1), inputs[],
#       + опционально seed, quality, generation_config.upscale (2x).
#       - Текст->картинка: operation = nano_banana_pro_image_generate (flow/nano-banana).
#       - Картинка+референс: та же operation nano_banana_pro_image_generate + inputs=[<ref>].
#         У flow НЕТ отдельной edit-операции: nano-banana принимает референсные картинки
#         прямо в _image_generate через inputs[] (это его основной кейс — консистентность
#         персонажа). Обе операции переопределяются через env (см. ниже).
#   • inputs[] (V6GenerationInput) — теперь два вида элемента:
#       1) строка: "data:<mime>;base64,<...>" ИЛИ сырой 32-символьный storage id (без file:);
#       2) объект V6NamedMediaInput: {"filename": "<имя>", "input": "<data-uri|storage_id>"} —
#          позволяет ссылаться на референс по имени файла прямо в промпте.
#     Скрипт использует именованные инпуты, чтобы промпт сцены мог явно ссылаться
#     на референс персонажа по имени файла.
#   • Ответ на создание — GenerationAcceptedResponse: id (generation id), operation,
#       provider, model, media_type, operation_type, billing.
#   • Статус: GET /api/v6/generations/{generation_id} -> GenerationStatusResponse:
#       status (queued|running|succeeded|failed),
#       results[] (GenerationResultItem: index, type, download_url, data, text,
#                  mime_type, metadata.storage_id),
#       usage, warnings, error, translations.
#   • metadata.storage_id: для файловых результатов сервер уже отдаёт storage id —
#       поэтому сгенерированный портрет персонажа можно сразу переиспользовать как
#       референс без повторной загрузки в storage.
#   • Storage: POST https://storage.fast-gen.ai/v2/upload -> сырой storage id
#       (нужен только для уже существующих на диске файлов, которых нет в state).

# 1) API-ключ берётся из окружения (не хардкодим секрет в файл, чтобы не утёк в git).
#    export FAST_GEN_API_KEY="ТВОЙ_КЛЮЧ"
API_KEY = (
    os.getenv("FAST_GEN_API_KEY")
    or os.getenv("FASTGEN_API_KEY")
    or os.getenv("MEDIA_GEN_API_KEY")
    or ""
)

# 2) Set your API base URL here
BASE_URL = os.getenv("FAST_GEN_API_BASE", "https://api.fast-gen.ai")

# Paths
BASE_DIR = Path("/Users/aleksandrtomilov/Desktop/ПРОМПТЫ")
PROMPT_FILE_CANDIDATES = [
    BASE_DIR / "generated_image_prompts.txt",
    BASE_DIR / "image_prompts.txt",
]
SCENES_DIR = BASE_DIR / "КАРТИНКИ"
CHARACTERS_DIR = BASE_DIR / "ПЕРСОНАЖИ"

# Generation options
# Flow (nano-banana) принимает соотношения сторон в форме n:n: "16:9", "9:16", "1:1".
SCENE_ASPECT_RATIO = "16:9"       # сцены — ландшафт
CHARACTER_ASPECT_RATIO = "16:9"   # портреты персонажей — только горизонталь (вертикаль запрещена)
REQUEST_TIMEOUT = 300
RETRY_COUNT = 4
RETRY_DELAY_SEC = 4
OPERATION_POLL_SEC = 6
OPERATION_TIMEOUT_SEC = 1800
SKIP_EXISTING = True
MAX_WORKERS = 25

# Сколько референсов персонажей максимум отдавать flow (nano-banana) на одну сцену.
# По умолчанию один референс (как раньше на flower); nano-banana умеет и несколько —
# лимит поднимается через env, инпуты уходят как именованные V6NamedMediaInput.
# Поддерживаем и новое имя переменной, и старое (обратная совместимость).
FLOW_MAX_REFERENCES = int(
    os.getenv("FAST_GEN_FLOW_MAX_REFERENCES")
    or os.getenv("FAST_GEN_FLOWER_MAX_REFERENCES")
    or "1"
)

# Опциональный seed (детерминизм), если модель поддерживает. Пусто -> не отправляем.
_SEED_ENV = os.getenv("FAST_GEN_SEED", "").strip()
GENERATION_SEED: Optional[int] = int(_SEED_ENV) if _SEED_ENV.isdigit() else None

# Storage server (V6).
STORAGE_UPLOAD_URL = os.getenv("FAST_GEN_STORAGE_UPLOAD_URL", "https://storage.fast-gen.ai/v2/upload")

# V6 endpoints.
V6_GENERATIONS_ENDPOINT = "/api/v6/generations"                       # POST create generation
V6_GENERATION_STATUS_ENDPOINT = "/api/v6/generations/{generation_id}"  # GET status

# Canonical V6 operation ids (см. GET /api/v6/capabilities). Переопределяются через env.
# Flow: и текст->картинка, и генерация-с-референсом идут через один и тот же
# nano-banana _image_generate (отдельной edit-операции у flow нет).
OP_IMAGE_GENERATE = os.getenv("FAST_GEN_IMAGE_OPERATION", "nano_banana_pro_image_generate")
OP_IMAGE_EDIT = os.getenv("FAST_GEN_IMAGE_EDIT_OPERATION", "") or OP_IMAGE_GENERATE

# File names
ALIAS_MAP_CANDIDATES = [
    BASE_DIR / "image_character_alias_map.json",
    BASE_DIR / "character_alias_map.json",
]
STATE_FILE = BASE_DIR / "character_generation_state.json"
SCENE_LOG_FILE = BASE_DIR / "scene_generation_log.json"

# V6 media input (V6MediaInput): либо data URI, либо сырой 32-символьный storage id.
DATA_URI_RE = re.compile(r"^data:[^;]+;base64,")
STORAGE_ID_RE = re.compile(r"^[a-f0-9]{32}$")
# aspect_ratio по контракту V6: n:n, где обе части — положительные целые.
ASPECT_RATIO_RE = re.compile(r"^([1-9]\d*):([1-9]\d*)$")

# Разрешены только горизонтальные картинки. Любой вертикальный (или квадратный)
# aspect_ratio запрещён и принудительно заменяется на этот дефолт.
DEFAULT_HORIZONTAL_ASPECT_RATIO = "16:9"

# Добавка к промпту, усиливающая горизонтальную (ландшафтную) композицию.
# Нужна потому, что модель может игнорировать поле aspect_ratio и решать
# ориентацию по содержанию промпта.
HORIZONTAL_PROMPT_SUFFIX = (
    "horizontal landscape composition, wide cinematic framing, image wider than tall, "
    "16:9 aspect ratio, full scene visible left to right, "
    "not vertical, not portrait orientation, not a tall image"
)

# Если True — сохранять картинку, даже если её не удалось получить горизонтальной
# (по умолчанию False: вертикаль сохранять запрещено, сцена помечается ошибкой).
ALLOW_SAVE_VERTICAL = os.getenv("FAST_GEN_ALLOW_VERTICAL", "").strip().lower() in ("1", "true", "yes")

# HTTP-коды, при которых повтор бессмыслен (ошибка запроса/валидации).
FATAL_HTTP_CODES = {400, 401, 403, 404, 422}


# =========================
# DATA MODELS
# =========================
@dataclass
class PromptLine:
    index: int
    raw_line: str
    aliases: List[str]
    body: str


@dataclass
class CharacterRef:
    """Референс персонажа для передачи в inputs[] как именованный V6NamedMediaInput."""
    alias: str
    storage_id: str
    filename: str
    # Локальный путь к файлу персонажа — источник правды для перезагрузки в storage,
    # когда storage id протух (TTL ~1 час с последнего использования).
    path: Optional[str] = None


class FatalApiError(RuntimeError):
    pass


# Storage id референсов имеют TTL и в длинном параллельном прогоне протухают.
# Обновляем их из локального файла по требованию, синхронизируя доступ к state
# и не допуская дублирующих загрузок одного и того же персонажа.
STATE_LOCK = Lock()
_alias_locks_guard = Lock()
_alias_locks: Dict[str, Lock] = {}

# Модель может по-разному трактовать aspect_ratio (иногда как height:width),
# поэтому рабочее значение, дающее ГОРИЗОНТАЛЬНЫЙ результат, определяем на лету
# по первой удачной генерации и переиспользуем, чтобы не тратить лишние запросы.
_HORIZONTAL_RATIO_LOCK = Lock()
_LEARNED_HORIZONTAL_RATIO: Optional[str] = None


def _alias_lock(alias: str) -> Lock:
    with _alias_locks_guard:
        lk = _alias_locks.get(alias)
        if lk is None:
            lk = Lock()
            _alias_locks[alias] = lk
        return lk


def is_expired_reference_error(exc: Exception) -> bool:
    """True, если ошибка про протухший/ненайденный референс в storage."""
    msg = str(exc)
    return (
        "file_not_found_or_expired" in msg
        or "not found or expired" in msg
        or "не найден или истек" in msg
    )


# =========================
# HELPERS
# =========================
def log(msg: str) -> None:
    print(msg, flush=True)


def ensure_api_ready() -> None:
    if not API_KEY:
        print("[ERROR] Не найден API ключ.")
        print('Задай его так: export FAST_GEN_API_KEY="ТВОЙ_КЛЮЧ"')
        sys.exit(1)
    if not BASE_URL or "YOUR_MEDIA_GEN_API_HOST" in BASE_URL:
        print("[ERROR] Укажи BASE_URL, например https://api.fast-gen.ai")
        sys.exit(1)


def ensure_dirs() -> None:
    SCENES_DIR.mkdir(parents=True, exist_ok=True)
    CHARACTERS_DIR.mkdir(parents=True, exist_ok=True)


def clean_base_url(url: str) -> str:
    return url.rstrip("/")


def headers() -> Dict[str, str]:
    return {
        "X-API-Key": API_KEY,
        "Content-Type": "application/json",
    }


def resolve_first_existing_file(candidates: List[Path], label: str) -> Path:
    for path in candidates:
        if path.exists():
            return path
    pretty = "\n".join(f"- {p}" for p in candidates)
    print(f"[ERROR] Не найден {label}. Проверил:\n{pretty}")
    sys.exit(1)


def parse_prompt_line(line: str, idx: int) -> Optional[PromptLine]:
    stripped = line.strip()
    if not stripped:
        return None
    m = re.match(r"^\[([^\]]+)\]\s*(.*)$", stripped)
    if m:
        alias_part = m.group(1).strip()
        body = m.group(2).strip()
        aliases = [a.strip() for a in alias_part.split(",") if a.strip()]
        return PromptLine(index=idx, raw_line=stripped, aliases=aliases, body=body)
    return PromptLine(index=idx, raw_line=stripped, aliases=[], body=stripped)


def load_prompts(prompt_file: Path) -> List[PromptLine]:
    lines: List[PromptLine] = []
    for idx, raw in enumerate(prompt_file.read_text(encoding="utf-8").splitlines(), start=1):
        parsed = parse_prompt_line(raw, idx)
        if parsed:
            lines.append(parsed)
    return lines


def load_json(path: Path, default):
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return default
    return default


def save_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def first_alias_occurrences(prompts: List[PromptLine]) -> Dict[str, PromptLine]:
    found: Dict[str, PromptLine] = {}
    for p in prompts:
        for alias in p.aliases:
            if alias not in found:
                found[alias] = p
    return found


def remove_alias_tokens(text: str) -> str:
    text = re.sub(r"\[[^\]]+\]", "", text)
    text = re.sub(r"\s+", " ", text).strip(" ,")
    return text


def character_portrait_prompt(alias: str, line_body: str) -> str:
    base = remove_alias_tokens(line_body)
    return (
        "photorealistic historical character reference portrait, single person only, upper body or half-body, "
        "facing camera, neutral expression, clear facial features, visible clothing details, plain unobtrusive background, "
        "natural lighting, realistic skin texture, documentary realism, historically grounded appearance, no other people, "
        "no action scene, no crowded background, no text, no logos, no watermark, "
        f"character reference for {alias}, {base}"
    )


def scene_prompt_without_aliases(body: str) -> str:
    cleaned = remove_alias_tokens(body)
    cleaned = re.sub(
        r"\bcharacter reference for\s+(First|Second|Third|Fourth|Fifth|Sixth|Seventh|Eighth|Ninth|Tenth)\b",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,")
    return cleaned


def compose_scene_prompt_with_refs(base_prompt: str, refs: List[CharacterRef]) -> str:
    """
    Вплетает подсказку про именованные референсы в промпт сцены.
    V6 позволяет ссылаться на инпут по filename, что улучшает узнаваемость персонажей.
    """
    if not refs:
        return base_prompt
    parts = "; ".join(f"{r.alias} as shown in reference image {r.filename}" for r in refs)
    return f"{base_prompt}. Use the provided reference images for character likeness: {parts}"


def decode_data_uri_to_bytes(data_uri: str) -> bytes:
    if not data_uri.startswith("data:"):
        raise ValueError("Ответ API не похож на data:<mime>;base64,... URI")
    _, b64 = data_uri.split(",", 1)
    return base64.b64decode(b64)


def image_dimensions(data: bytes) -> Optional[Tuple[int, int]]:
    """
    Возвращает (width, height) картинки, читая заголовок PNG/JPEG/WEBP без сторонних
    библиотек. None — если формат не распознан (тогда ориентацию не проверяем).
    """
    # PNG: сигнатура + IHDR(width,height) на фиксированном смещении.
    if data[:8] == b"\x89PNG\r\n\x1a\n" and len(data) >= 24:
        w, h = struct.unpack(">II", data[16:24])
        return int(w), int(h)

    # JPEG: ищем маркер SOF, в нём height, width.
    if data[:2] == b"\xff\xd8":
        i, n = 2, len(data)
        while i + 9 < n:
            if data[i] != 0xFF:
                i += 1
                continue
            marker = data[i + 1]
            if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
                          0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
                h, w = struct.unpack(">HH", data[i + 5:i + 9])
                return int(w), int(h)
            if marker == 0xD8 or marker == 0xD9 or 0xD0 <= marker <= 0xD7:
                i += 2
                continue
            seg_len = struct.unpack(">H", data[i + 2:i + 4])[0]
            i += 2 + seg_len
        return None

    # WEBP (VP8/VP8L/VP8X) — на всякий случай.
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP" and len(data) >= 30:
        chunk = data[12:16]
        try:
            if chunk == b"VP8 ":
                w = struct.unpack("<H", data[26:28])[0] & 0x3FFF
                h = struct.unpack("<H", data[28:30])[0] & 0x3FFF
                return int(w), int(h)
            if chunk == b"VP8L":
                b = data[21:25]
                bits = b[0] | (b[1] << 8) | (b[2] << 16) | (b[3] << 24)
                w = (bits & 0x3FFF) + 1
                h = ((bits >> 14) & 0x3FFF) + 1
                return int(w), int(h)
            if chunk == b"VP8X":
                w = 1 + (data[24] | (data[25] << 8) | (data[26] << 16))
                h = 1 + (data[27] | (data[28] << 8) | (data[29] << 16))
                return int(w), int(h)
        except Exception:
            return None
    return None


def is_horizontal_image(data: bytes) -> bool:
    """
    True, если картинка горизонтальная (ширина > высоты) ИЛИ формат нераспознан
    (тогда не блокируем). Вертикаль и квадрат считаем НЕ горизонтальными.
    """
    dims = image_dimensions(data)
    if dims is None:
        log("[WARN] Не удалось определить размеры картинки — пропускаю проверку ориентации.")
        return True
    w, h = dims
    return w > h


def normalize_storage_id(value: str) -> str:
    """
    V6 inputs используют сырой 32-символьный storage id без префикса.
    Срезаем устаревший префикс file:, если он придёт из старого state или storage.
    """
    sid = value.strip()
    if sid.startswith("file:"):
        sid = sid[len("file:"):]
    return sid


def validate_media_input(value: str) -> str:
    """
    Проверяет соответствие строки схеме V6MediaInput:
      data:<mime>;base64,<...>   ИЛИ   сырой 32-символьный storage id.
    Иначе генерация вернёт 422.
    """
    if DATA_URI_RE.match(value) or STORAGE_ID_RE.match(value):
        return value
    raise FatalApiError(
        "media input не соответствует схеме V6MediaInput (нужно 'data:<mime>;base64,...' или "
        f"32-символьный storage id): {value[:60]}..."
    )


def named_media_input(storage_id: str, filename: str) -> Dict[str, str]:
    """Собирает V6NamedMediaInput: {'filename', 'input'} с валидацией input."""
    return {
        "filename": filename,
        "input": validate_media_input(normalize_storage_id(storage_id)),
    }


def validate_aspect_ratio(aspect_ratio: str) -> str:
    """
    Проверяет aspect_ratio по контракту V6 (n:n) и разрешает ТОЛЬКО горизонтальные картинки.

    Вертикальные (высота > ширины) и квадратные (1:1) соотношения запрещены и
    принудительно заменяются на горизонтальный дефолт — вертикаль генерировать нельзя.
    """
    m = ASPECT_RATIO_RE.match(aspect_ratio)
    if not m:
        log(f"[WARN] aspect_ratio {aspect_ratio!r} не в формате n:n — использую {DEFAULT_HORIZONTAL_ASPECT_RATIO!r}.")
        return DEFAULT_HORIZONTAL_ASPECT_RATIO

    width, height = int(m.group(1)), int(m.group(2))
    if width <= height:
        log(
            f"[WARN] aspect_ratio {aspect_ratio!r} не горизонтальный (нужна ширина > высоты) — "
            f"разрешены только горизонтальные картинки, использую {DEFAULT_HORIZONTAL_ASPECT_RATIO!r}."
        )
        return DEFAULT_HORIZONTAL_ASPECT_RATIO

    return aspect_ratio


def swap_aspect_ratio(aspect_ratio: str) -> str:
    """Меняет местами части n:m -> m:n (для перебора трактовки aspect_ratio моделью)."""
    m = ASPECT_RATIO_RE.match(aspect_ratio)
    if not m:
        return aspect_ratio
    return f"{m.group(2)}:{m.group(1)}"


def with_horizontal_hint(prompt: str) -> str:
    """Добавляет к промпту подсказку про горизонтальную композицию (если её ещё нет)."""
    if HORIZONTAL_PROMPT_SUFFIX in prompt:
        return prompt
    return f"{prompt}. {HORIZONTAL_PROMPT_SUFFIX}"


def upload_file_to_storage(image_path: Path) -> str:
    file_size_mb = image_path.stat().st_size / (1024 * 1024)
    log(f"[DEBUG] Uploading to storage: {image_path.name} ({file_size_mb:.2f} MB)")

    with image_path.open("rb") as f:
        resp = requests.post(
            STORAGE_UPLOAD_URL,
            headers={"X-API-Key": API_KEY},
            files={"file": (image_path.name, f, "image/png")},
            timeout=REQUEST_TIMEOUT,
        )

    log(f"[DEBUG] Storage response status: {resp.status_code}")
    if resp.status_code >= 400:
        log(f"[DEBUG] Storage response text: {resp.text}")

    resp.raise_for_status()
    data = resp.json()

    # storage V6 может вернуть id в разных полях — берём первый непустой.
    storage_id = (
        data.get("id")
        or data.get("storage_id")
        or data.get("file_hash")
        or data.get("file_id")
    )
    if not storage_id and isinstance(data.get("result"), dict):
        nested = data["result"]
        storage_id = (
            nested.get("id")
            or nested.get("storage_id")
            or nested.get("file_hash")
            or nested.get("file_id")
        )

    if not storage_id:
        raise RuntimeError(f"Не удалось загрузить файл в storage: {data}")

    # Схема V6MediaInput требует сырой storage id — валидируем сразу,
    # чтобы не словить 422 уже на генерации.
    return validate_media_input(normalize_storage_id(str(storage_id)))


def refresh_alias_storage_id(alias: str, state: dict, stale_id: Optional[str]) -> Optional[str]:
    """
    Перезагружает локальный файл персонажа в storage и возвращает свежий storage id.
    Потокобезопасно: под per-alias локом; если другой поток уже обновил id — используем его.
    """
    lk = _alias_lock(alias)
    with lk:
        entry = state.get(alias) or {}
        current = normalize_storage_id(entry.get("file_hash") or "")
        stale = normalize_storage_id(stale_id or "")
        if current and current != stale:
            # Другой поток уже обновил референс — переиспользуем.
            return current

        # Путь из state, а если его нет (старый state) — детерминированный путь по alias.
        path_str = entry.get("path")
        p = Path(path_str) if path_str else character_output_path(alias)
        if not p.exists():
            log(f"[WARN] Локальный файл персонажа {alias} отсутствует: {p} — не могу обновить референс.")
            return None

        try:
            new_id = upload_file_to_storage(p)
        except Exception as e:
            log(f"[WARN] Не удалось перезагрузить референс {alias}: {e}")
            return None

        entry["file_hash"] = new_id
        entry["path"] = str(p)
        with STATE_LOCK:
            state[alias] = entry
            save_json(STATE_FILE, state)
        log(f"[INFO] Референс {alias} обновлён в storage: {new_id}")
        return new_id


def request_with_retries(url: str, payload: dict) -> dict:
    last_err: Optional[Exception] = None
    for attempt in range(1, RETRY_COUNT + 1):
        try:
            resp = requests.post(url, headers=headers(), json=payload, timeout=REQUEST_TIMEOUT)
            if resp.status_code in FATAL_HTTP_CODES:
                raise FatalApiError(f"HTTP {resp.status_code}: {resp.text}")
            if resp.status_code == 429:
                raise RuntimeError(f"Rate limit 429: {resp.text}")
            resp.raise_for_status()
            return resp.json()
        except FatalApiError:
            raise
        except Exception as e:
            last_err = e
            if attempt < RETRY_COUNT:
                log(f"[WARN] Попытка {attempt}/{RETRY_COUNT} не удалась: {e}")
                time.sleep(RETRY_DELAY_SEC * attempt)
            else:
                break
    raise RuntimeError(f"Запрос к API провалился после {RETRY_COUNT} попыток: {last_err}")


def request_get_with_retries(url: str, params: Optional[dict] = None) -> dict:
    last_err: Optional[Exception] = None
    for attempt in range(1, RETRY_COUNT + 1):
        try:
            resp = requests.get(url, headers={"X-API-Key": API_KEY}, params=params, timeout=REQUEST_TIMEOUT)
            if resp.status_code in FATAL_HTTP_CODES:
                raise FatalApiError(f"HTTP {resp.status_code}: {resp.text}")
            if resp.status_code == 429:
                raise RuntimeError(f"Rate limit 429: {resp.text}")
            resp.raise_for_status()
            return resp.json()
        except FatalApiError:
            raise
        except Exception as e:
            last_err = e
            if attempt < RETRY_COUNT:
                log(f"[WARN] GET попытка {attempt}/{RETRY_COUNT} не удалась: {e}")
                time.sleep(RETRY_DELAY_SEC * attempt)
            else:
                break
    raise RuntimeError(f"GET запрос к API провалился после {RETRY_COUNT} попыток: {last_err}")


def download_url_to_bytes(url: str) -> bytes:
    """Скачивает файл результата по download_url (с ключом на случай приватного стораджа)."""
    last_err: Optional[Exception] = None
    for attempt in range(1, RETRY_COUNT + 1):
        try:
            resp = requests.get(url, headers={"X-API-Key": API_KEY}, timeout=REQUEST_TIMEOUT)
            if resp.status_code in FATAL_HTTP_CODES:
                raise FatalApiError(f"HTTP {resp.status_code}: {resp.text[:200]}")
            resp.raise_for_status()
            if not resp.content:
                raise RuntimeError("Пустой ответ при скачивании результата")
            return resp.content
        except FatalApiError:
            raise
        except Exception as e:
            last_err = e
            if attempt < RETRY_COUNT:
                log(f"[WARN] download result попытка {attempt}/{RETRY_COUNT} не удалась: {e}")
                time.sleep(RETRY_DELAY_SEC * attempt)
            else:
                break
    raise RuntimeError(f"Скачивание результата провалилось после {RETRY_COUNT} попыток: {last_err}")


def result_item_to_bytes(result_item: Dict[str, Any]) -> bytes:
    """
    Достаёт байты картинки из GenerationResultItem.
    Приоритет источников: inline data URI -> download_url.
    """
    item_type = result_item.get("type")
    if item_type not in (None, "image"):
        log(f"[WARN] Ожидали image, а результат type={item_type!r} — беру как есть")

    inline = result_item.get("data")
    if isinstance(inline, str) and inline.startswith("data:"):
        return decode_data_uri_to_bytes(inline)

    download_url = result_item.get("download_url")
    if not (isinstance(download_url, str) and download_url):
        raise RuntimeError(f"Result item без data и download_url: {result_item}")
    return download_url_to_bytes(download_url)


def save_image_bytes_to_file(data: bytes, path: Path) -> None:
    """Пишет байты картинки в path и проверяет, что файл не пустой."""
    path.write_bytes(data)
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"Файл не сохранился или пустой: {path}")


def poll_generation(generation_id: str) -> Dict[str, Any]:
    """Ждёт завершения генерации и возвращает первый result item."""
    url = clean_base_url(BASE_URL) + V6_GENERATION_STATUS_ENDPOINT.format(generation_id=generation_id)
    started = time.time()
    while True:
        data = request_get_with_retries(url)
        status = data.get("status")

        if status in ("queued", "running"):
            if time.time() - started > OPERATION_TIMEOUT_SEC:
                raise RuntimeError(f"Генерация {generation_id} не завершилась за {OPERATION_TIMEOUT_SEC}s")
            time.sleep(OPERATION_POLL_SEC)
            continue

        if status == "succeeded":
            results = data.get("results") or []
            if not results:
                raise RuntimeError(f"Генерация {generation_id} завершилась, но results пустой: {data}")
            for w in (data.get("warnings") or []):
                log(f"[WARN] gen {generation_id}: {w}")
            return results[0]

        if status == "failed":
            err = data.get("error") or data
            translations = data.get("translations") or {}
            if isinstance(translations, dict) and translations:
                tr = translations.get("ru") or next(iter(translations.values()), None)
                if tr:
                    err = f"{err} | {tr}"
            raise RuntimeError(f"Генерация {generation_id} закончилась ошибкой: {err}")

        raise RuntimeError(f"Неизвестный статус генерации {generation_id}: {data}")


def generate_image_flow(
    prompt: str,
    aspect_ratio: str,
    references: Optional[List[CharacterRef]] = None,
    state: Optional[dict] = None,
) -> bytes:
    """
    Генерация ГОРИЗОНТАЛЬНОЙ картинки через flow (nano-banana).

    Без референса — operation nano_banana_pro_image_generate (текст->картинка).
    С референсом — та же operation + inputs[] (у flow нет отдельной edit-операции).
    Референсы уходят в inputs[] как именованные V6NamedMediaInput, а промпт ссылается
    на них по filename. Число референсов ограничено FLOW_MAX_REFERENCES.

    Гарантия горизонтальной ориентации:
      • промпт усиливается подсказкой про ландшафтную композицию;
      • после генерации проверяются реальные размеры картинки;
      • если модель проигнорировала aspect_ratio и вернула вертикаль — пробуем
        перевёрнутое значение (модель может трактовать aspect_ratio как height:width);
      • рабочее значение запоминается и переиспользуется, чтобы не тратить запросы;
      • если горизонталь получить не удалось — вертикаль НЕ сохраняется (RuntimeError),
        кроме случая FAST_GEN_ALLOW_VERTICAL=1.

    Storage id референсов имеют TTL: если API вернул "file_not_found_or_expired",
    перезагружаем файлы персонажей из локальных копий (через state) и повторяем.
    Возвращает байты картинки после ожидания завершения и проверки ориентации.
    """
    global _LEARNED_HORIZONTAL_RATIO

    url = clean_base_url(BASE_URL) + V6_GENERATIONS_ENDPOINT

    refs = list(references or [])
    if len(refs) > FLOW_MAX_REFERENCES:
        log(
            f"[WARN] Референсов {len(refs)}, лимит FLOW_MAX_REFERENCES={FLOW_MAX_REFERENCES} — "
            f"беру первые {FLOW_MAX_REFERENCES}."
        )
        refs = refs[:FLOW_MAX_REFERENCES]

    def build_payload(ar_value: str) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "operation": OP_IMAGE_GENERATE,
            "prompt": with_horizontal_hint(prompt),
            "aspect_ratio": ar_value,
        }
        if GENERATION_SEED is not None:
            payload["seed"] = GENERATION_SEED
        if refs:
            payload["operation"] = OP_IMAGE_EDIT
            payload["prompt"] = with_horizontal_hint(compose_scene_prompt_with_refs(prompt, refs))
            payload["inputs"] = [named_media_input(r.storage_id, r.filename) for r in refs]
        return payload

    def generate_once(ar_value: str) -> bytes:
        """Одна генерация с заданным aspect_ratio (+1 повтор после протухших референсов)."""
        for attempt in range(2):
            try:
                data = request_with_retries(url, build_payload(ar_value))
                generation_id = data.get("id")
                if not generation_id:
                    raise RuntimeError(f"Flow image API не вернул generation id: {data}")
                return result_item_to_bytes(poll_generation(generation_id))
            except (FatalApiError, RuntimeError) as e:
                can_refresh = refs and state is not None and attempt == 0 and is_expired_reference_error(e)
                if not can_refresh:
                    raise
                log("[INFO] Референс(ы) протухли в storage — перезагружаю из локальных файлов и повторяю.")
                refreshed = False
                for r in refs:
                    new_id = refresh_alias_storage_id(r.alias, state, r.storage_id)
                    if new_id and new_id != normalize_storage_id(r.storage_id):
                        r.storage_id = new_id
                        refreshed = True
                if not refreshed:
                    raise
        raise RuntimeError("generate_once: исчерпаны попытки после обновления референсов")

    # Кандидаты aspect_ratio для получения горизонтали.
    # Если рабочее значение уже выяснено в этом прогоне — используем только его.
    horizontal_ar = validate_aspect_ratio(aspect_ratio)   # напр. 16:9
    with _HORIZONTAL_RATIO_LOCK:
        learned = _LEARNED_HORIZONTAL_RATIO
    if learned:
        candidates = [learned]
    else:
        swapped = swap_aspect_ratio(horizontal_ar)         # напр. 9:16
        candidates = [horizontal_ar] + ([swapped] if swapped != horizontal_ar else [])

    last_bytes: Optional[bytes] = None
    for ar_value in candidates:
        data_bytes = generate_once(ar_value)
        last_bytes = data_bytes
        if is_horizontal_image(data_bytes):
            with _HORIZONTAL_RATIO_LOCK:
                if _LEARNED_HORIZONTAL_RATIO != ar_value:
                    _LEARNED_HORIZONTAL_RATIO = ar_value
                    log(f"[INFO] Горизонтальный результат при aspect_ratio={ar_value!r} — запоминаю это значение.")
            return data_bytes
        log(f"[WARN] Результат получился вертикальным при aspect_ratio={ar_value!r} — пробую другое значение.")

    if ALLOW_SAVE_VERTICAL and last_bytes is not None:
        log("[WARN] FAST_GEN_ALLOW_VERTICAL=1 — сохраняю вертикальную картинку, хотя горизонталь получить не удалось.")
        return last_bytes

    raise RuntimeError(
        "Не удалось получить горизонтальную картинку: модель вернула вертикаль для всех "
        "вариантов aspect_ratio. Вертикаль не сохраняю (FAST_GEN_ALLOW_VERTICAL=1 снимет запрет)."
    )


def scene_output_path(index: int) -> Path:
    return SCENES_DIR / f"{index:04d}.png"


def character_output_path(alias: str) -> Path:
    safe = alias.strip("[]")
    return CHARACTERS_DIR / f"[{safe}].png"


def reference_filename(alias: str) -> str:
    """Стабильное имя файла референса для ссылки из промпта (V6NamedMediaInput.filename)."""
    safe = re.sub(r"[^0-9A-Za-zА-Яа-яёЁ_-]+", "_", alias.strip("[]")).strip("_") or "character"
    return f"{safe}.png"


def maybe_existing_file_hash(state: dict, alias: str) -> Optional[str]:
    entry = state.get(alias)
    if isinstance(entry, dict):
        fh = entry.get("file_hash")
        return normalize_storage_id(fh) if fh else None
    return None


def build_character_refs(prompts: List[PromptLine], state: dict) -> dict:
    firsts = first_alias_occurrences(prompts)

    for alias, pline in sorted(firsts.items(), key=lambda kv: kv[1].index):
        out_path = character_output_path(alias)
        file_hash = maybe_existing_file_hash(state, alias)

        if out_path.exists() and file_hash:
            log(f"[SKIP] Персонаж {alias} уже есть: {out_path.name}")
            continue

        if out_path.exists() and not file_hash:
            log(f"[INFO] Загружаю существующий файл персонажа в storage: {out_path.name}")
            file_hash = upload_file_to_storage(out_path)
            state[alias] = {
                "path": str(out_path),
                "file_hash": file_hash,
                "first_line": pline.index,
            }
            save_json(STATE_FILE, state)
            continue

        portrait_prompt = character_portrait_prompt(alias, pline.body)
        log(f"[CHAR] Генерирую персонажа {alias} из строки {pline.index}")
        image_bytes = generate_image_flow(portrait_prompt, CHARACTER_ASPECT_RATIO)
        save_image_bytes_to_file(image_bytes, out_path)

        # Референс грузим как отдельный файл в storage (TTL ~1 час с последнего
        # использования). Storage id самого результата НЕ переиспользуем как референс:
        # у сгенерированного результата TTL короче (~30 мин), и он быстро протухает.
        # Если id всё же истечёт в длинном прогоне — refresh_alias_storage_id перезальёт
        # файл из локальной копии по пути ниже.
        file_hash = upload_file_to_storage(out_path)

        state[alias] = {
            "path": str(out_path),
            "file_hash": file_hash,
            "first_line": pline.index,
            "portrait_prompt": portrait_prompt,
        }
        save_json(STATE_FILE, state)
    return state


def collect_scene_references(p: PromptLine, state: dict) -> List[CharacterRef]:
    refs: List[CharacterRef] = []
    for alias in p.aliases:
        entry = state.get(alias) or {}
        fh = entry.get("file_hash")
        if fh:
            refs.append(
                CharacterRef(
                    alias=alias,
                    storage_id=normalize_storage_id(fh),
                    filename=reference_filename(alias),
                    path=entry.get("path"),
                )
            )
    return refs


def generate_one_scene(p: PromptLine, state: dict) -> dict:
    out_path = scene_output_path(p.index)

    if SKIP_EXISTING and out_path.exists():
        log(f"[SKIP] Сцена {p.index} уже существует")
        return {
            "line": p.index,
            "output": str(out_path),
            "skipped": True,
            "aliases": p.aliases,
        }

    clean_prompt = scene_prompt_without_aliases(p.body)
    refs = collect_scene_references(p, state)

    try:
        if refs:
            log(f"[SCENE] {p.index}: flow + ref через {', '.join(p.aliases[:3])}")
            image_bytes = generate_image_flow(clean_prompt, SCENE_ASPECT_RATIO, refs, state)
            mode = "flow_with_ref"
        else:
            log(f"[SCENE] {p.index}: flow без ref")
            image_bytes = generate_image_flow(clean_prompt, SCENE_ASPECT_RATIO)
            mode = "flow_from_text"

        save_image_bytes_to_file(image_bytes, out_path)
        return {
            "line": p.index,
            "output": str(out_path),
            "aliases": p.aliases,
            "mode": mode,
            "prompt": clean_prompt,
        }
    except Exception as e:
        log(f"[ERROR] Сцена {p.index} не сгенерировалась: {e}")
        return {
            "line": p.index,
            "output": str(out_path),
            "aliases": p.aliases,
            "error": str(e),
            "prompt": clean_prompt,
        }


def generate_scene_images_parallel(prompts: List[PromptLine], state: dict) -> List[dict]:
    logs: List[dict] = []
    write_lock = Lock()

    def handle_result(result: dict) -> None:
        with write_lock:
            logs.append(result)
            logs.sort(key=lambda x: x.get("line", 0))
            save_json(SCENE_LOG_FILE, logs)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_prompt = {executor.submit(generate_one_scene, p, state): p for p in prompts}
        for future in as_completed(future_to_prompt):
            result = future.result()
            handle_result(result)

    logs.sort(key=lambda x: x.get("line", 0))
    save_json(SCENE_LOG_FILE, logs)
    return logs


def main() -> None:
    ensure_api_ready()
    ensure_dirs()

    prompt_file = resolve_first_existing_file(PROMPT_FILE_CANDIDATES, "файл промптов")
    alias_map_file = resolve_first_existing_file(ALIAS_MAP_CANDIDATES, "файл карты персонажей") if any(p.exists() for p in ALIAS_MAP_CANDIDATES) else None

    prompts = load_prompts(prompt_file)
    if not prompts:
        print(f"[ERROR] В {prompt_file.name} нет строк для обработки.")
        sys.exit(1)

    alias_lines = sum(1 for p in prompts if p.aliases)
    unique_aliases = sorted({alias for p in prompts for alias in p.aliases})

    log(f"[INFO] Использую файл промптов: {prompt_file.name}")
    log(f"[INFO] Найдено промптов: {len(prompts)}")
    log(f"[INFO] Строк с alias: {alias_lines}")
    log(f"[INFO] Уникальных alias: {len(unique_aliases)} -> {', '.join(unique_aliases[:12]) if unique_aliases else 'нет'}")
    log(f"[INFO] Режим: media_gen_api V6 flow (nano-banana) image, потоков: {MAX_WORKERS}")
    log(f"[INFO] Операции: text->{OP_IMAGE_GENERATE}, ref->{OP_IMAGE_EDIT} (именованные inputs), max ref={FLOW_MAX_REFERENCES}")
    if GENERATION_SEED is not None:
        log(f"[INFO] Seed: {GENERATION_SEED}")

    alias_map = load_json(alias_map_file, {}) if alias_map_file else {}
    if alias_map_file and alias_map:
        log(f"[INFO] Найдена карта персонажей: {alias_map_file.name}")
    elif alias_map_file:
        log(f"[INFO] Файл карты персонажей найден, но пуст или не прочитался: {alias_map_file.name}")
    else:
        log("[INFO] Карта персонажей не найдена, продолжаю только по alias в строках промптов")

    state = load_json(STATE_FILE, {})

    # Pass 1: create/upload character references
    state = build_character_refs(prompts, state)

    if unique_aliases and not any(alias in state for alias in unique_aliases):
        log("[WARN] Alias в промптах есть, но ни один персонаж не попал в state после Pass 1")

    # Pass 2: generate all scene images in parallel
    logs = generate_scene_images_parallel(prompts, state)

    success_count = sum(1 for x in logs if not x.get("error"))
    error_count = sum(1 for x in logs if x.get("error"))
    log(f"[DONE] Готово. Успешно: {success_count}, с ошибками: {error_count}")
    log(f"[OUT] Сцены: {SCENES_DIR}")
    log(f"[OUT] Персонажи: {CHARACTERS_DIR}")


if __name__ == "__main__":
    main()
