from __future__ import annotations

import base64
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

import requests

# =========================
# CONFIG
# =========================
# 1) Insert your media generation API key here
API_KEY = "CPemyQY7NuiZrCjEgvQYu32U7pT6X1c7"

# 2) Set your API base URL here
BASE_URL = "https://api.fast-gen.ai"

# Paths
BASE_DIR = Path("/Users/aleksandrtomilov/Desktop/ПРОМПТЫ")
PROMPT_FILE_CANDIDATES = [
    BASE_DIR / "generated_image_prompts.txt",
    BASE_DIR / "image_prompts.txt",
]
SCENES_DIR = BASE_DIR / "КАРТИНКИ"
CHARACTERS_DIR = BASE_DIR / "ПЕРСОНАЖИ"

# Generation options
# Flower принимает простые соотношения сторон: "16:9", "9:16", "1:1".
SCENE_ASPECT_RATIO = "16:9"       # сцены — ландшафт
CHARACTER_ASPECT_RATIO = "9:16"   # портреты персонажей — вертикаль
REQUEST_TIMEOUT = 300
RETRY_COUNT = 4
RETRY_DELAY_SEC = 4
OPERATION_POLL_SEC = 6
SKIP_EXISTING = True
MAX_WORKERS = 25

# Storage server from the spec
STORAGE_UPLOAD_URL = "https://storage.fast-gen.ai/upload"

# V4 Flower endpoints.
FLOWER_IMAGE_ENDPOINT = "/api/v4/flower/image/generate"   # Nano Banana 2 via Flower
V4_OPERATION_ENDPOINT = "/api/v4/operations/{operation_id}"

# File names
ALIAS_MAP_CANDIDATES = [
    BASE_DIR / "image_character_alias_map.json",
    BASE_DIR / "character_alias_map.json",
]
STATE_FILE = BASE_DIR / "character_generation_state.json"
SCENE_LOG_FILE = BASE_DIR / "scene_generation_log.json"


# =========================
# DATA MODELS
# =========================
@dataclass
class PromptLine:
    index: int
    raw_line: str
    aliases: List[str]
    body: str


# =========================
# HELPERS
# =========================
def log(msg: str) -> None:
    print(msg, flush=True)


def ensure_api_ready() -> None:
    if not API_KEY or API_KEY == "PASTE_YOUR_MEDIA_GEN_API_KEY_HERE":
        print("[ERROR] Вставь свой API ключ в переменную API_KEY в начале файла.")
        sys.exit(1)
    if not BASE_URL or "YOUR_MEDIA_GEN_API_HOST" in BASE_URL:
        print("[ERROR] Укажи BASE_URL в начале файла, например https://api.fast-gen.ai")
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


def decode_data_uri_to_png_bytes(data_uri: str) -> bytes:
    if not data_uri.startswith("data:image"):
        raise ValueError("Ответ API не похож на data:image/... URI")
    _, b64 = data_uri.split(",", 1)
    return base64.b64decode(b64)


def save_data_uri_to_file(data_uri: str, path: Path) -> None:
    path.write_bytes(decode_data_uri_to_png_bytes(data_uri))


def normalize_file_ref(file_hash: str) -> str:
    fh = file_hash.strip()
    return fh if fh.startswith("file:") else f"file:{fh}"


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

    # Support both {"file_hash": "..."} and {"result": {"file_hash": "..."}}
    file_hash = data.get("file_hash")
    if not file_hash and isinstance(data.get("result"), dict):
        file_hash = data["result"].get("file_hash")

    if not data.get("success", True) or not file_hash:
        raise RuntimeError(f"Не удалось загрузить файл в storage: {data}")

    return normalize_file_ref(file_hash)


def request_with_retries(url: str, payload: dict) -> dict:
    last_err: Optional[Exception] = None
    for attempt in range(1, RETRY_COUNT + 1):
        try:
            resp = requests.post(url, headers=headers(), json=payload, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 429:
                raise RuntimeError(f"Rate limit 429: {resp.text}")
            resp.raise_for_status()
            return resp.json()
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
            if resp.status_code == 429:
                raise RuntimeError(f"Rate limit 429: {resp.text}")
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            last_err = e
            if attempt < RETRY_COUNT:
                log(f"[WARN] GET попытка {attempt}/{RETRY_COUNT} не удалась: {e}")
                time.sleep(RETRY_DELAY_SEC * attempt)
            else:
                break
    raise RuntimeError(f"GET запрос к API провалился после {RETRY_COUNT} попыток: {last_err}")


def resolve_result_data_uri(value: Any) -> str:
    """
    Flower возвращает result как список:
      ["data:image/png;base64,..."]  при result_format=data_uri
      ["file:abc..."]                при result_format=ref
    """
    if isinstance(value, list):
        if not value:
            raise RuntimeError("Пустой список result")
        value = value[0]

    if not isinstance(value, str):
        raise RuntimeError(f"Неожиданный result от API: {value}")

    if value.startswith("data:"):
        return value

    if value.startswith("file:"):
        file_hash = value.replace("file:", "", 1)
        url = f"https://storage.fast-gen.ai/file/{file_hash}"
        resp = requests.get(url, headers={"X-API-Key": API_KEY}, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        text = resp.text.strip()
        if text.startswith("data:"):
            return text
        try:
            data = resp.json()
            for key in ("data_uri", "result", "file", "content"):
                candidate = data.get(key)
                if isinstance(candidate, str) and candidate.startswith("data:"):
                    return candidate
        except ValueError:
            pass
        raise RuntimeError(f"Storage не вернул data URI: {text[:200]}")

    raise RuntimeError(f"Неизвестный формат result: {value[:200]}")


def poll_operation(operation_id: str) -> str:
    url = clean_base_url(BASE_URL) + V4_OPERATION_ENDPOINT.format(operation_id=operation_id)
    while True:
        data = request_get_with_retries(url, params={"result_format": "data_uri"})
        status = data.get("status")

        if status in ("pending", "processing"):
            time.sleep(OPERATION_POLL_SEC)
            continue

        if status == "success":
            result = data.get("result")
            if not result:
                raise RuntimeError(f"Операция {operation_id} завершилась, но result пустой: {data}")
            return resolve_result_data_uri(result)

        if status == "error":
            raise RuntimeError(f"Операция {operation_id} закончилась ошибкой: {data.get('error') or data}")

        raise RuntimeError(f"Неизвестный статус операции {operation_id}: {data}")


def generate_image_flower(prompt: str, aspect_ratio: str, reference_hashes: Optional[List[str]] = None) -> str:
    """
    Генерация картинки через Flower (Nano Banana 2).

    Flower image generate принимает один опциональный reference_image (img2img edit).
    В отличие от старого flow (до 10 reference_images), здесь берётся только первый
    референс; при наличии нескольких alias предупреждаем и используем первый.
    """
    url = clean_base_url(BASE_URL) + FLOWER_IMAGE_ENDPOINT
    payload: Dict[str, Any] = {
        "prompt": prompt,
        "aspect_ratio": aspect_ratio,
    }
    if reference_hashes:
        if len(reference_hashes) > 1:
            log(f"[WARN] Flower использует только 1 референс — беру первый из {len(reference_hashes)}.")
        # reference_image принимает data URI или file:ref. Когда он задан,
        # aspect_ratio игнорируется (размер берётся из исходной картинки).
        payload["reference_image"] = normalize_file_ref(reference_hashes[0])

    data = request_with_retries(url, payload)
    operation_id = data.get("operation_id")
    if not operation_id:
        raise RuntimeError(f"Flower image API не вернул operation_id: {data}")
    return poll_operation(operation_id)


def scene_output_path(index: int) -> Path:
    return SCENES_DIR / f"{index:04d}.png"


def character_output_path(alias: str) -> Path:
    safe = alias.strip("[]")
    return CHARACTERS_DIR / f"[{safe}].png"


def maybe_existing_file_hash(state: dict, alias: str) -> Optional[str]:
    entry = state.get(alias)
    if isinstance(entry, dict):
        fh = entry.get("file_hash")
        return normalize_file_ref(fh) if fh else None
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
        data_uri = generate_image_flower(portrait_prompt, CHARACTER_ASPECT_RATIO)
        save_data_uri_to_file(data_uri, out_path)
        file_hash = upload_file_to_storage(out_path)
        state[alias] = {
            "path": str(out_path),
            "file_hash": file_hash,
            "first_line": pline.index,
            "portrait_prompt": portrait_prompt,
        }
        save_json(STATE_FILE, state)
    return state


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
    alias_hashes: List[str] = []
    for alias in p.aliases:
        entry = state.get(alias) or {}
        fh = entry.get("file_hash")
        if fh:
            alias_hashes.append(normalize_file_ref(fh))

    try:
        if alias_hashes:
            log(f"[SCENE] {p.index}: flower + ref через {', '.join(p.aliases[:3])}")
            data_uri = generate_image_flower(clean_prompt, SCENE_ASPECT_RATIO, alias_hashes)
            mode = "flower_with_ref"
        else:
            log(f"[SCENE] {p.index}: flower без ref")
            data_uri = generate_image_flower(clean_prompt, SCENE_ASPECT_RATIO)
            mode = "flower_from_text"

        save_data_uri_to_file(data_uri, out_path)
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
    log(f"[INFO] Режим: Nano Banana 2 via Flower, потоков: {MAX_WORKERS}")

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
