from __future__ import annotations

import json
import math
import os
import random
import re
import subprocess
import sys
import time
import wave
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from openai import OpenAI

# ============================================================
# 1) OPENAI KEY / MODEL SETTINGS
# ============================================================
# Ключ можно хранить прямо здесь, как в исходной версии скрипта.
# ВАЖНО: если файл куда-то отправляется или выкладывается, такой ключ лучше перевыпустить.
# Если API_KEY оставить пустым, скрипт попробует взять ключ из переменной окружения OPENAI_API_KEY.
API_KEY = 'YOUR_OPENAI_API_KEY_HERE'

# При желании можно задать модель через переменную OPENAI_MODEL:
# macOS / Linux:
#   export OPENAI_MODEL="gpt-4.1-mini"
# Windows PowerShell:
#   setx OPENAI_MODEL "gpt-4.1-mini"

ENV_MODEL = os.getenv("OPENAI_MODEL", "").strip()
MODEL_CANDIDATES = [m for m in [ENV_MODEL, "gpt-4.5", "gpt-4.1-mini", "gpt-4o-mini"] if m]

# Более высокая температура и штрафы за повторы дают заметно более разные сцены.
GENERATION_TEMPERATURE = 0.85
PRESENCE_PENALTY = 0.65
FREQUENCY_PENALTY = 0.35
MAX_RETRIES = 3

# ============================================================
# 2) BASE SETTINGS
# ============================================================
try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()

OUTPUT_JSON = BASE_DIR / "generated_prompts.json"
OUTPUT_TXT = BASE_DIR / "generated_prompts.txt"
OUTPUT_CONTEXT = BASE_DIR / "global_context.json"
OUTPUT_BLOCKS = BASE_DIR / "scene_blocks.json"
OUTPUT_STATS = BASE_DIR / "prompt_generation_stats.json"

TARGET_SCENE_SECONDS = 5.0
MIN_BLOCK_SECONDS = 3.4
MAX_BLOCK_SECONDS = 7.2
MAX_SCENES_PER_BLOCK = 3
SECONDS_PER_100_CHARS = 7.8

MIN_KEY_DETAILS = 3
MIN_BACKGROUND_DETAILS = 2
MIN_BODY_LANGUAGE_DETAILS = 2

# Контроль разнообразия.
DIVERSITY_MEMORY_WINDOW = 14
RECENT_SCENES_FOR_PROMPT = 12
MAX_SAME_SCENE_TYPE_IN_RECENT = 2
MAX_SAME_ENVIRONMENT_IN_RECENT = 1
MAX_SAME_CAMERA_IN_RECENT = 2
RANDOM_SEED = 42

# Если модель всё равно выдаёт слишком общие сцены, постобработка заменит generic-поля.
GENERIC_ENVIRONMENT_PATTERNS = [
    r"\broom\b",
    r"\binterior\b",
    r"\bspace\b",
    r"\bquiet place\b",
    r"\bsimple interior\b",
    r"\bgrounded interior\b",
]
GENERIC_ACTION_PATTERNS = [
    r"subtle emotional movement",
    r"holding still",
    r"looking away",
    r"standing quietly",
    r"sitting quietly",
    r"breathing deeply",
]

# ============================================================
# 3) SINGLE-CHARACTER VIDEO SETTINGS
# ============================================================
REFERENCE_CHARACTER_TAG = "this character"
REFERENCE_CHARACTER_HINT = ""  # Например: "adult woman", "tired office worker"

GLOBAL_NEGATIVE_TEMPLATE = (
    "single character only, no extra people, no crowd, no duplicate person, "
    "no text, no subtitles, no watermark, no logo"
)

REALISM_TEMPLATE = (
    "grounded psychological realism, natural human behavior, believable body language, "
    "clear emotional readability, realistic environment, layered foreground and background, 16:9"
)

DETAIL_BOOST_TEMPLATE = (
    "specific physical behavior, concrete small details, visible emotion in posture and gesture, "
    "subtle facial tension, tactile props, non-generic scene construction"
)

# ============================================================
# 4) SCENE TYPES
# ============================================================
# Расширенный набор типов сцен. Чем больше разных физических ситуаций, тем меньше однообразия.
SCENE_TYPES: Dict[str, Dict[str, str]] = {
    "inner_portrait": {
        "description": "Close or medium-close emotional portrait of one recurring character.",
        "template": "emotion-focused portrait shot, subtle facial changes, readable eyes and micro-expressions",
    },
    "body_language": {
        "description": "The character reveals their state through posture, gesture, and body tension.",
        "template": "single-character body language shot, expressive posture and restrained movement, clear emotional readability",
    },
    "room_interaction": {
        "description": "The character interacts with meaningful objects in a grounded interior.",
        "template": "single-character interaction shot in a grounded interior, meaningful object contact, psychological realism",
    },
    "walking_reflection": {
        "description": "The character walks, stops, turns, or moves through space with emotional intent.",
        "template": "single-character reflective movement shot, walking or pausing with emotional intent, grounded spatial motion",
    },
    "mirror_moment": {
        "description": "Mirror, glass, or reflective surface used for self-observation.",
        "template": "single-character reflection shot with mirror or glass, introspective visual tension, no duplicated person",
    },
    "window_moment": {
        "description": "The character near a window, balcony, curtain, or outside light.",
        "template": "single-character window-side shot, inward emotion contrasted with outside space, contemplative stillness",
    },
    "symbolic_object": {
        "description": "The character and one meaningful object connected to the theme.",
        "template": "single-character symbolic object shot, object and gesture carry psychological meaning, tactile detail",
    },
    "memory_fragment": {
        "description": "A grounded memory-like fragment with only the same character visible.",
        "template": "single-character memory-like scene, emotionally charged but grounded, intimate psychological atmosphere",
    },
    "emotional_transition": {
        "description": "Visible change from one emotional state to another.",
        "template": "single-character emotional transition shot, visible change in posture expression and rhythm",
    },
    "close_detail_action": {
        "description": "Tight detail of hands, face, shoulders, breathing, or object action.",
        "template": "tight psychological detail shot, hands face breath or small object action clearly readable",
    },
    "domestic_ritual": {
        "description": "Ordinary domestic routine reveals inner pressure or emotional avoidance.",
        "template": "single-character domestic ritual shot, ordinary action revealing emotional pressure",
    },
    "threshold_moment": {
        "description": "Doorway, stairs, hallway, exit, entrance, or border between two spaces as a decision point.",
        "template": "single-character threshold shot, decision point shown through space and posture",
    },
    "public_isolation": {
        "description": "The character alone in an empty public or semi-public space.",
        "template": "single-character public isolation shot, empty public space emphasizing inner distance",
    },
    "object_decision": {
        "description": "The character chooses, hides, leaves, releases, or protects an object.",
        "template": "single-character decision-through-object shot, clear choice expressed through gesture",
    },
    "routine_break": {
        "description": "A normal routine is interrupted by emotional weight, hesitation, or realization.",
        "template": "single-character interrupted routine shot, small failure or hesitation reveals conflict",
    },
    "environmental_pressure": {
        "description": "The surrounding space visually presses on the character: narrow, cluttered, empty, or oversized.",
        "template": "single-character environmental pressure shot, space composition expresses psychological tension",
    },
}

DEFAULT_CAMERA_BY_TYPE = {
    "inner_portrait": "medium close-up at eye level",
    "body_language": "medium shot with full upper-body posture visible",
    "room_interaction": "medium shot close to the action",
    "walking_reflection": "tracking medium shot or side profile walk",
    "mirror_moment": "over-shoulder or angled mirror framing",
    "window_moment": "three-quarter medium shot near the window",
    "symbolic_object": "medium close-up with object in hands or foreground",
    "memory_fragment": "intimate medium close shot with gentle spatial separation",
    "emotional_transition": "medium close-up capturing change in face and body",
    "close_detail_action": "tight close-up on hands face shoulders or object",
    "domestic_ritual": "medium side-profile shot close to the domestic action",
    "threshold_moment": "wide shot framed through a doorway or corridor",
    "public_isolation": "wide shot with the character small in the frame",
    "object_decision": "over-the-shoulder shot focused on the object and hands",
    "routine_break": "static medium shot showing the interrupted routine",
    "environmental_pressure": "wide or high-angle shot emphasizing the surrounding space",
}

DETAILED_SCENE_TYPES = {
    "inner_portrait",
    "body_language",
    "room_interaction",
    "mirror_moment",
    "symbolic_object",
    "emotional_transition",
    "close_detail_action",
    "domestic_ritual",
    "threshold_moment",
    "object_decision",
    "routine_break",
    "environmental_pressure",
}

# ============================================================
# 5) VARIATION BANKS
# ============================================================
# Эти банки не задают стиль. Они задают физическую вариативность сцен.
ENVIRONMENT_BANK = [
    "small kitchen at night with a sink and one cup",
    "narrow hallway with shoes, keys, and a half-open door",
    "bathroom sink with fogged mirror and folded towel",
    "quiet bedroom with an unmade bed and chair near the wall",
    "empty stairwell between floors",
    "laundromat with spinning machines and plastic chairs",
    "small cafe corner near a cold window, no other people visible",
    "parked car interior in a quiet parking lot",
    "office desk after everyone has left",
    "balcony with railings and distant city lights",
    "empty train platform with the character alone",
    "messy living room with scattered papers and a silent phone",
    "dim elevator with reflective metal walls",
    "quiet chapel corner or empty church bench",
    "small grocery aisle after closing time, empty and still",
    "bus stop shelter after rain with wet pavement",
    "attic corner with boxes and old fabric",
    "basement laundry room with humming pipes",
    "rooftop access door with concrete floor",
    "plain waiting room with one chair and a wall clock",
    "kitchen table covered with unopened letters",
    "bedside table with a lamp and a glass of water",
    "coat closet with hanging jackets and a small shelf",
    "empty classroom with chairs pushed under desks",
    "long corridor with numbered doors and cold light",
]

ACTION_BANK = [
    "folding and unfolding a small note",
    "washing the same cup longer than necessary",
    "standing still with keys in hand",
    "slowly packing a bag and then stopping",
    "touching an old photo without looking directly at it",
    "sitting on the floor beside the bed",
    "walking down stairs and pausing halfway",
    "opening a drawer and closing it without taking anything",
    "holding a cup with both hands without drinking",
    "leaning against a door after closing it",
    "placing a small object into a drawer",
    "turning a phone face down on the table",
    "pulling a sleeve over the wrist as a self-protective gesture",
    "tying and untying a shoelace without leaving",
    "straightening objects on a table too carefully",
    "stopping in front of a door but not knocking",
    "pressing both palms on the sink edge",
    "writing one sentence and crossing it out",
    "holding a coat but not putting it on",
    "switching a lamp on and off once, then freezing",
    "counting coins or small objects mechanically",
    "pulling a chair back and then pushing it in again",
    "opening a window slightly and immediately closing it",
    "sitting inside a parked car with both hands on the steering wheel",
]

OBJECT_BANK = [
    "keys",
    "folded note",
    "turned-down phone",
    "old photograph",
    "half-empty cup",
    "unopened letter",
    "coat",
    "small notebook",
    "drawer handle",
    "lamp switch",
    "towel",
    "chair",
    "bag zipper",
    "shoelace",
    "watch",
    "glass of water",
    "bus ticket",
    "paper envelope",
    "simple ring or small accessory",
    "book with no readable text",
]

CAMERA_VARIATION_BANK = [
    "wide shot with the character small in the frame",
    "medium side-profile shot",
    "over-the-shoulder shot focused on the object",
    "low angle from table height",
    "static frontal medium shot",
    "slow tracking side shot",
    "tight close-up on hands and lower face",
    "high angle showing the character isolated in space",
    "three-quarter shot with foreground object partially blocking the view",
    "wide doorway framing from the next room",
    "close profile shot with background depth",
    "medium shot from behind the character looking into the space",
]

COMPOSITION_BANK = [
    "character placed at the far left with empty space on the right",
    "character centered but partly blocked by foreground objects",
    "character framed through a doorway",
    "character separated from the background by a clear line of light",
    "character surrounded by ordinary objects that feel emotionally heavy",
    "character small against a larger environment",
    "hands and object in the foreground, face slightly behind",
    "diagonal line of hallway or furniture leading toward the character",
    "character paused at the border between two spaces",
    "foreground shadow or furniture creates depth without hiding the action",
]

BODY_LANGUAGE_BANK = [
    "one shoulder slightly raised as if guarding the body",
    "hands held too tightly around a small object",
    "jaw held firm while the eyes soften",
    "weight shifted onto one foot, ready to leave but unable to move",
    "head lowered while the torso remains tense",
    "arms close to the body in a protective posture",
    "slow exhale visible in the chest and shoulders",
    "fingers tapping once and then stopping",
    "back pressed against a wall or door",
    "hands open slowly after holding tension",
    "neck and shoulders gradually relaxing",
    "eyes fixed on an object instead of the room",
]

EMOTION_MICRO_BEATS = [
    "suppressed anxiety",
    "quiet resistance",
    "painful hesitation",
    "emotional numbness",
    "private shame",
    "guarded vulnerability",
    "controlled anger",
    "tired acceptance",
    "fragile hope",
    "inner conflict",
    "reluctant release",
    "self-protective calm",
    "loneliness without drama",
    "a decision forming silently",
]

FALLBACK_SCENES = [
    {
        "scene_type": "domestic_ritual",
        "subject": "this character beside a kitchen sink",
        "environment": "small kitchen at night with a sink and one cup",
        "action": "washing the same cup longer than necessary",
        "emotion": "suppressed anxiety",
        "body_pose": "slightly hunched shoulders, hands moving mechanically",
        "key_details": ["water running over one cup", "fingers gripping the cup too tightly", "silent counter with one towel"],
        "background_details": ["small kitchen surfaces", "one chair pulled slightly away from the table"],
        "body_language_details": ["shoulders held high", "jaw tense while hands keep moving"],
        "camera_framing": "medium side-profile shot close to the domestic action",
        "composition_notes": "sink and hands in foreground, character slightly turned away",
        "reason": "varied fallback domestic scene",
    },
    {
        "scene_type": "threshold_moment",
        "subject": "this character standing near a half-open door",
        "environment": "narrow hallway with shoes, keys, and a half-open door",
        "action": "holding the door handle but not leaving",
        "emotion": "hesitation before a difficult choice",
        "body_pose": "one foot forward, body frozen between staying and going",
        "key_details": ["hand on door handle", "keys hanging near the door", "coat sleeve slightly twisted"],
        "background_details": ["narrow hallway depth", "shoes lined unevenly near the wall"],
        "body_language_details": ["weight shifted forward", "shoulders tight and still"],
        "camera_framing": "wide shot framed through a doorway or corridor",
        "composition_notes": "character paused at the border between two spaces",
        "reason": "varied fallback threshold scene",
    },
    {
        "scene_type": "object_decision",
        "subject": "this character at a cluttered desk",
        "environment": "office desk after everyone has left",
        "action": "placing a small object into a drawer and closing it slowly",
        "emotion": "painful acceptance",
        "body_pose": "controlled hands, tight jaw, lowered gaze",
        "key_details": ["drawer closing slowly", "object partly hidden by the hand", "phone turned face down nearby"],
        "background_details": ["papers pushed to one side", "empty chair behind the desk"],
        "body_language_details": ["controlled hand movement", "tight jaw and lowered eyes"],
        "camera_framing": "over-the-shoulder shot focused on the object",
        "composition_notes": "hands and drawer dominate the frame, face partly visible",
        "reason": "varied fallback decision scene",
    },
    {
        "scene_type": "public_isolation",
        "subject": "this character alone under a bus stop shelter",
        "environment": "bus stop shelter after rain with wet pavement",
        "action": "holding a coat closed while looking at the empty street",
        "emotion": "loneliness without drama",
        "body_pose": "arms close to the body, shoulders slightly lifted",
        "key_details": ["wet pavement reflection", "coat held closed", "empty bench beside the character"],
        "background_details": ["quiet street beyond the shelter", "rain marks on transparent panels"],
        "body_language_details": ["arms close to torso", "still posture with guarded breathing"],
        "camera_framing": "wide shot with the character small in the frame",
        "composition_notes": "empty shelter space around the character emphasizes isolation",
        "reason": "varied fallback public isolation scene",
    },
]

# ============================================================
# 6) DATA STRUCTURES
# ============================================================
@dataclass
class SentenceUnit:
    index: int
    text: str
    est_seconds: float


@dataclass
class SceneBlock:
    block_id: int
    text: str
    sentence_indexes: List[int]
    est_seconds: float
    scene_budget: int


# ============================================================
# 7) BASIC UTILITIES
# ============================================================
def fail(msg: str) -> None:
    print(f"[ERROR] {msg}")
    sys.exit(1)


def info(msg: str) -> None:
    print(f"[INFO] {msg}")


def ensure_base_dir() -> None:
    BASE_DIR.mkdir(parents=True, exist_ok=True)


def get_client() -> OpenAI:
    # Сначала используем ключ, прописанный прямо в скрипте.
    # Если API_KEY пустой, берём ключ из переменной окружения OPENAI_API_KEY.
    key = (API_KEY or "").strip() or os.getenv("OPENAI_API_KEY", "").strip()
    if not key or key == "PASTE_YOUR_OPENAI_API_KEY_HERE":
        fail(
            "Не найден API_KEY. Вставь ключ в переменную API_KEY внутри файла "
            "или задай переменную окружения OPENAI_API_KEY."
        )
    return OpenAI(api_key=key)


def read_text(path: Path) -> str:
    if not path.exists():
        fail(f"Не найден файл сценария: {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        fail(f"Файл сценария пустой: {path.name}")
    return text


def find_scenario_file() -> Path:
    preferred_names = ["scenario.txt", "script.txt", "scenariu.txt", "scenario.md", "script.md"]
    for name in preferred_names:
        path = BASE_DIR / name
        if path.exists() and path.is_file():
            return path

    excluded_names = {
        "generated_prompts.txt",
        "generated_prompts.json",
        "global_context.json",
        "scene_blocks.json",
        "prompt_generation_stats.json",
        Path(__file__).name if "__file__" in globals() else "",
    }

    candidates: List[Tuple[Path, int]] = []
    for p in BASE_DIR.iterdir():
        if not p.is_file():
            continue
        if p.name in excluded_names:
            continue
        if p.suffix.lower() not in {".txt", ".md"}:
            continue
        try:
            content = p.read_text(encoding="utf-8").strip()
        except Exception:
            continue
        if len(content) < 200:
            continue
        candidates.append((p, len(content)))

    if not candidates:
        fail(
            "Не найден файл сценария в папке скрипта. Положи сценарий в эту же папку, "
            "лучше всего под именем scenario.txt или script.txt."
        )

    candidates.sort(key=lambda x: (-x[1], x[0].name.lower()))
    return candidates[0][0]


def find_audio_file() -> Optional[Path]:
    preferred_names = [
        "voice.mp3",
        "voice.wav",
        "voice.m4a",
        "voice.aac",
        "audio.mp3",
        "audio.wav",
        "audio.m4a",
        "audio.aac",
    ]
    for name in preferred_names:
        path = BASE_DIR / name
        if path.exists():
            return path

    audio_exts = {".mp3", ".wav", ".m4a", ".aac"}
    found = sorted(
        [p for p in BASE_DIR.iterdir() if p.is_file() and p.suffix.lower() in audio_exts],
        key=lambda x: x.name.lower(),
    )
    return found[0] if found else None


def ffprobe_duration(path: Path) -> Optional[float]:
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        value = result.stdout.strip()
        return float(value)
    except Exception:
        return None


def wave_duration(path: Path) -> Optional[float]:
    if path.suffix.lower() != ".wav":
        return None
    try:
        with wave.open(str(path), "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            return frames / float(rate)
    except Exception:
        return None


def get_audio_duration(path: Optional[Path], scenario_text: str) -> float:
    if path is not None:
        dur = ffprobe_duration(path)
        if dur is not None:
            info(f"Длина аудио через ffprobe: {dur:.2f} сек")
            return dur
        dur = wave_duration(path)
        if dur is not None:
            info(f"Длина аудио через wave: {dur:.2f} сек")
            return dur
        info("Не удалось прочитать длину аудио, использую расчёт по символам.")
    chars = len(scenario_text)
    dur = (chars / 100.0) * SECONDS_PER_100_CHARS
    info(f"Оценочная длина по символам: {dur:.2f} сек")
    return dur


def split_into_sentences(text: str) -> List[str]:
    normalized = re.sub(r"\s+", " ", text).strip()
    parts = re.split(r"(?<=[.!?])\s+(?=[A-ZА-ЯЁ0-9\"'])", normalized)
    sentences = [p.strip() for p in parts if p.strip()]
    return sentences or [normalized]


def estimate_sentence_durations(scenario_text: str, audio_seconds: float) -> List[SentenceUnit]:
    sentences = split_into_sentences(scenario_text)
    total_chars = sum(max(len(s), 1) for s in sentences)
    units: List[SentenceUnit] = []
    for idx, s in enumerate(sentences, start=1):
        share = max(len(s), 1) / total_chars
        est = max(0.8, audio_seconds * share)
        units.append(SentenceUnit(index=idx, text=s, est_seconds=est))
    return units


def build_blocks(units: List[SentenceUnit]) -> List[SceneBlock]:
    blocks: List[SceneBlock] = []
    current_texts: List[str] = []
    current_ids: List[int] = []
    current_seconds = 0.0
    block_id = 1

    def flush_block() -> None:
        nonlocal block_id, current_texts, current_ids, current_seconds
        if not current_texts:
            return
        text = " ".join(current_texts).strip()
        scene_budget = max(1, min(MAX_SCENES_PER_BLOCK, math.ceil(current_seconds / TARGET_SCENE_SECONDS)))
        blocks.append(
            SceneBlock(
                block_id=block_id,
                text=text,
                sentence_indexes=current_ids[:],
                est_seconds=current_seconds,
                scene_budget=scene_budget,
            )
        )
        block_id += 1
        current_texts = []
        current_ids = []
        current_seconds = 0.0

    for unit in units:
        if unit.est_seconds >= MAX_BLOCK_SECONDS:
            flush_block()
            blocks.append(
                SceneBlock(
                    block_id=block_id,
                    text=unit.text,
                    sentence_indexes=[unit.index],
                    est_seconds=unit.est_seconds,
                    scene_budget=max(1, min(MAX_SCENES_PER_BLOCK, math.ceil(unit.est_seconds / TARGET_SCENE_SECONDS))),
                )
            )
            block_id += 1
            continue

        if not current_texts:
            current_texts.append(unit.text)
            current_ids.append(unit.index)
            current_seconds = unit.est_seconds
            continue

        if current_seconds < MIN_BLOCK_SECONDS:
            current_texts.append(unit.text)
            current_ids.append(unit.index)
            current_seconds += unit.est_seconds
            continue

        if current_seconds + unit.est_seconds <= MAX_BLOCK_SECONDS:
            current_texts.append(unit.text)
            current_ids.append(unit.index)
            current_seconds += unit.est_seconds
        else:
            flush_block()
            current_texts.append(unit.text)
            current_ids.append(unit.index)
            current_seconds = unit.est_seconds

    flush_block()
    return blocks


# ============================================================
# 8) JSON / TEXT CLEANING
# ============================================================
def safe_json_loads(text: str) -> Any:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    if not (text.startswith("{") or text.startswith("[")):
        match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
        if match:
            text = match.group(1)
    return json.loads(text)


def call_model_json(client: OpenAI, system_prompt: str, user_prompt: str) -> Any:
    last_error: Optional[Exception] = None

    for model in MODEL_CANDIDATES:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=GENERATION_TEMPERATURE,
                    presence_penalty=PRESENCE_PENALTY,
                    frequency_penalty=FREQUENCY_PENALTY,
                    response_format={"type": "json_object"},
                )
                content = response.choices[0].message.content or "{}"
                return safe_json_loads(content)
            except Exception as e:
                last_error = e
                info(f"Модель {model}, попытка {attempt}/{MAX_RETRIES} не удалась: {e}")
                time.sleep(1.2 * attempt)
                continue

    raise RuntimeError(f"Не удалось получить JSON от модели. Последняя ошибка: {last_error}")


def clean_text_fragment(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    text = re.sub(r"\s+", " ", text)
    text = text.rstrip(".,")
    return text


def clean_string_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        value = []
    cleaned: List[str] = []
    for item in value:
        text = clean_text_fragment(item)
        if text:
            cleaned.append(text)
    return cleaned


def dedupe_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for item in items:
        key = re.sub(r"\s+", " ", item.lower()).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def is_generic_text(text: str, patterns: Sequence[str]) -> bool:
    low = clean_text_fragment(text).lower()
    if not low:
        return True
    return any(re.search(pattern, low, flags=re.IGNORECASE) for pattern in patterns)


# ============================================================
# 9) DIVERSITY HELPERS
# ============================================================
def build_scene_type_reference() -> str:
    lines = []
    for key, meta in SCENE_TYPES.items():
        lines.append(f"- {key}: {meta['description']}")
    return "\n".join(lines)


def build_character_anchor() -> str:
    hint = clean_text_fragment(REFERENCE_CHARACTER_HINT)
    if hint:
        return f"{REFERENCE_CHARACTER_TAG}, {hint}"
    return REFERENCE_CHARACTER_TAG


def rotate_list(items: Sequence[str], start: int) -> List[str]:
    if not items:
        return []
    start = start % len(items)
    return list(items[start:]) + list(items[:start])


def sampled(items: Sequence[str], rnd: random.Random, count: int) -> List[str]:
    items = list(items)
    if count >= len(items):
        rnd.shuffle(items)
        return items
    return rnd.sample(items, count)


def pick_variation_pool(block_id: int, global_context: Dict[str, Any]) -> Dict[str, List[str]]:
    rnd = random.Random(RANDOM_SEED + block_id * 997)

    context_envs = clean_string_list(global_context.get("suggested_environments"))
    context_objects = clean_string_list(global_context.get("recurring_objects"))
    context_emotions = clean_string_list(global_context.get("dominant_emotions"))
    context_motifs = clean_string_list(global_context.get("visual_motifs"))

    envs = dedupe_preserve_order(context_envs + sampled(ENVIRONMENT_BANK, rnd, 8))
    objects = dedupe_preserve_order(context_objects + sampled(OBJECT_BANK, rnd, 8))
    emotions = dedupe_preserve_order(context_emotions + sampled(EMOTION_MICRO_BEATS, rnd, 7))
    motifs = dedupe_preserve_order(context_motifs + sampled(COMPOSITION_BANK, rnd, 5))

    return {
        "environments": envs[:10],
        "actions": sampled(ACTION_BANK, rnd, 10),
        "objects": objects[:10],
        "camera_framings": sampled(CAMERA_VARIATION_BANK, rnd, 8),
        "composition_ideas": motifs[:8],
        "body_language": sampled(BODY_LANGUAGE_BANK, rnd, 8),
        "emotion_micro_beats": emotions[:10],
    }


def compact_recent_scene(scene: Dict[str, Any]) -> Dict[str, str]:
    return {
        "scene_type": clean_text_fragment(scene.get("scene_type")),
        "environment": clean_text_fragment(scene.get("environment")),
        "action": clean_text_fragment(scene.get("action")),
        "emotion": clean_text_fragment(scene.get("emotion")),
        "body_pose": clean_text_fragment(scene.get("body_pose")),
        "camera_framing": clean_text_fragment(scene.get("camera_framing")),
        "object_or_details": "; ".join(clean_string_list(scene.get("key_details"))[:2]),
    }


def summarize_recent_scenes(recent_scenes: Sequence[Dict[str, Any]]) -> List[Dict[str, str]]:
    return [compact_recent_scene(scene) for scene in recent_scenes[-RECENT_SCENES_FOR_PROMPT:]]


def count_recent_values(recent_scenes: Sequence[Dict[str, Any]], field: str) -> Counter:
    values = []
    for scene in recent_scenes[-DIVERSITY_MEMORY_WINDOW:]:
        value = clean_text_fragment(scene.get(field)).lower()
        if value:
            values.append(value)
    return Counter(values)


def repeated_scene_types_to_avoid(recent_scenes: Sequence[Dict[str, Any]]) -> List[str]:
    counts = count_recent_values(recent_scenes, "scene_type")
    return [scene_type for scene_type, count in counts.items() if count >= MAX_SAME_SCENE_TYPE_IN_RECENT]


def default_detail_priority_for_type(scene_type: str) -> str:
    return "detailed" if scene_type in DETAILED_SCENE_TYPES else "balanced"


def normalize_detail_priority(value: Any, scene_type: str) -> str:
    text = clean_text_fragment(value).lower()
    if text not in {"detailed", "balanced"}:
        text = default_detail_priority_for_type(scene_type)
    return text


def fallback_from_pool(block_id: int, local_scene_index: int, pool: Dict[str, List[str]]) -> Dict[str, Any]:
    # Циклический fallback, чтобы даже при плохом ответе модели сцены не превращались в одинаковые портреты.
    base = dict(FALLBACK_SCENES[(block_id + local_scene_index) % len(FALLBACK_SCENES)])
    rnd = random.Random(RANDOM_SEED + block_id * 101 + local_scene_index * 17)
    if pool.get("environments"):
        base["environment"] = rnd.choice(pool["environments"])
    if pool.get("actions"):
        base["action"] = rnd.choice(pool["actions"])
    if pool.get("camera_framings"):
        base["camera_framing"] = rnd.choice(pool["camera_framings"])
    if pool.get("composition_ideas"):
        base["composition_notes"] = rnd.choice(pool["composition_ideas"])
    if pool.get("body_language"):
        base["body_pose"] = rnd.choice(pool["body_language"])
    if pool.get("emotion_micro_beats"):
        base["emotion"] = rnd.choice(pool["emotion_micro_beats"])
    return base


def choose_non_repeated(current: str, options: Sequence[str], recent_scenes: Sequence[Dict[str, Any]], field: str, max_repeat: int, salt: int) -> str:
    current_clean = clean_text_fragment(current)
    counts = count_recent_values(recent_scenes, field)
    if current_clean and counts[current_clean.lower()] < max_repeat and not is_generic_text(current_clean, GENERIC_ENVIRONMENT_PATTERNS if field == "environment" else []):
        return current_clean

    rnd = random.Random(RANDOM_SEED + salt)
    candidates = list(options)
    rnd.shuffle(candidates)
    for candidate in candidates:
        candidate_clean = clean_text_fragment(candidate)
        if candidate_clean and counts[candidate_clean.lower()] < max_repeat:
            return candidate_clean
    return current_clean or (candidates[0] if candidates else "specific grounded place")


def enforce_scene_type_diversity(scene: Dict[str, Any], recent_scenes: Sequence[Dict[str, Any]], block_id: int, local_scene_index: int) -> str:
    scene_type = clean_text_fragment(scene.get("scene_type"))
    if scene_type not in SCENE_TYPES:
        scene_type = "inner_portrait"

    avoid = set(repeated_scene_types_to_avoid(recent_scenes))
    if scene_type not in avoid:
        return scene_type

    all_types = list(SCENE_TYPES.keys())
    rnd = random.Random(RANDOM_SEED + block_id * 31 + local_scene_index * 7)
    rnd.shuffle(all_types)
    recent_counts = count_recent_values(recent_scenes, "scene_type")
    for candidate in all_types:
        if candidate not in avoid and recent_counts[candidate.lower()] < MAX_SAME_SCENE_TYPE_IN_RECENT:
            return candidate
    return scene_type


# ============================================================
# 10) SCENE NORMALIZATION / PROMPT BUILDING
# ============================================================
def build_default_detail_lists(scene_type: str, subject: str, environment: str, action: str, emotion: str) -> Dict[str, List[str]]:
    subject_fallback = subject or "the character"
    environment_fallback = environment or "a specific grounded place"
    action_fallback = action or "a specific physical action"
    emotion_fallback = emotion or "inner tension"

    key_details = [
        f"readable expression connected to {emotion_fallback}",
        f"clear physical evidence of {action_fallback}",
        f"small props or surfaces near {subject_fallback}",
    ]
    background_details = [
        f"lived-in details of {environment_fallback}",
        "foreground and background depth that supports the emotion",
    ]
    body_language_details = [
        "tension in shoulders neck hands or jaw",
        "subtle shift in posture or breathing",
    ]

    if scene_type == "mirror_moment":
        key_details = [
            "reflection clearly readable without duplicating the character",
            f"emotion of {emotion_fallback} visible in face and stillness",
            "small imperfections on glass mirror or reflective surface",
        ]
    elif scene_type == "window_moment":
        key_details = [
            "hands near the window frame curtain or glass",
            f"emotion of {emotion_fallback} visible in pause and gaze direction",
            "difference between interior space and outside light",
        ]
    elif scene_type == "close_detail_action":
        key_details = [
            "tiny hand movement facial tension or breath detail",
            f"direct physical sign of {emotion_fallback}",
            "small tactile contact with a nearby object or fabric",
        ]
    elif scene_type == "walking_reflection":
        key_details = [
            "measured walking pace or sudden stop",
            f"body movement shaped by {emotion_fallback}",
            "feet hands and torso rhythm clearly readable",
        ]
    elif scene_type == "domestic_ritual":
        key_details = [
            f"ordinary action made emotionally meaningful: {action_fallback}",
            "one domestic object handled with unusual care",
            f"small signs of pressure inside {environment_fallback}",
        ]
    elif scene_type == "threshold_moment":
        key_details = [
            "doorway corridor stair or exit clearly visible",
            "body paused between moving forward and staying back",
            f"choice or hesitation visible through {action_fallback}",
        ]
    elif scene_type == "public_isolation":
        key_details = [
            "empty public space around the character",
            "character physically alone without any other people",
            f"posture shaped by {emotion_fallback}",
        ]
    elif scene_type == "object_decision":
        key_details = [
            "object clearly central to the gesture",
            "hands show a decision rather than decoration",
            f"emotional consequence visible in {subject_fallback}",
        ]
    elif scene_type == "routine_break":
        key_details = [
            "interrupted routine visible through stopped movement",
            "ordinary object left unfinished or misplaced",
            f"pause reveals {emotion_fallback}",
        ]
    elif scene_type == "environmental_pressure":
        key_details = [
            "space visually feels narrow empty cluttered or oversized",
            "character placement shows psychological pressure",
            f"environment reinforces {emotion_fallback}",
        ]

    return {
        "key_details": key_details,
        "background_details": background_details,
        "body_language_details": body_language_details,
    }


def ensure_scene_specificity(
    scene: Dict[str, Any],
    recent_scenes: Optional[Sequence[Dict[str, Any]]] = None,
    variation_pool: Optional[Dict[str, List[str]]] = None,
    block_id: int = 0,
    local_scene_index: int = 0,
) -> Dict[str, Any]:
    recent_scenes = recent_scenes or []
    variation_pool = variation_pool or {}

    scene_type = enforce_scene_type_diversity(scene, recent_scenes, block_id, local_scene_index)

    subject = clean_text_fragment(scene.get("subject")) or "this character"
    if not subject.lower().startswith("this character"):
        subject = f"this character, {subject}"

    environment = clean_text_fragment(scene.get("environment"))
    if is_generic_text(environment, GENERIC_ENVIRONMENT_PATTERNS):
        environment = ""
    environment = choose_non_repeated(
        environment,
        variation_pool.get("environments", ENVIRONMENT_BANK),
        recent_scenes,
        "environment",
        MAX_SAME_ENVIRONMENT_IN_RECENT,
        block_id * 19 + local_scene_index,
    )

    action = clean_text_fragment(scene.get("action"))
    if is_generic_text(action, GENERIC_ACTION_PATTERNS):
        action = ""
    action = choose_non_repeated(
        action,
        variation_pool.get("actions", ACTION_BANK),
        recent_scenes,
        "action",
        1,
        block_id * 23 + local_scene_index,
    )

    emotion = clean_text_fragment(scene.get("emotion")) or choose_non_repeated(
        "",
        variation_pool.get("emotion_micro_beats", EMOTION_MICRO_BEATS),
        recent_scenes,
        "emotion",
        2,
        block_id * 29 + local_scene_index,
    )

    body_pose = clean_text_fragment(scene.get("body_pose")) or choose_non_repeated(
        "",
        variation_pool.get("body_language", BODY_LANGUAGE_BANK),
        recent_scenes,
        "body_pose",
        1,
        block_id * 37 + local_scene_index,
    )

    detail_priority = normalize_detail_priority(scene.get("detail_priority"), scene_type)

    camera_framing = clean_text_fragment(scene.get("camera_framing")) or DEFAULT_CAMERA_BY_TYPE.get(scene_type, "medium close-up")
    camera_framing = choose_non_repeated(
        camera_framing,
        variation_pool.get("camera_framings", CAMERA_VARIATION_BANK),
        recent_scenes,
        "camera_framing",
        MAX_SAME_CAMERA_IN_RECENT,
        block_id * 41 + local_scene_index,
    )

    composition_notes = clean_text_fragment(scene.get("composition_notes"))
    if not composition_notes:
        composition_notes = choose_non_repeated(
            "",
            variation_pool.get("composition_ideas", COMPOSITION_BANK),
            recent_scenes,
            "composition_notes",
            1,
            block_id * 43 + local_scene_index,
        )

    # Минималистичный подход: берём только то, что дала модель, без раздувания деталями из банков
    key_details = dedupe_preserve_order(clean_string_list(scene.get("key_details")))[:3]
    background_details = dedupe_preserve_order(clean_string_list(scene.get("background_details")))[:2]
    body_language_details = dedupe_preserve_order(clean_string_list(scene.get("body_language_details")))[:2]

    # Минимальные fallback-детали только если модель не дала ничего
    if not key_details:
        key_details = [f"visible sign of {emotion or 'inner tension'}"]
    if not body_language_details:
        body_language_details = ["tension visible in posture"]

    return {
        "scene_type": scene_type,
        "detail_priority": detail_priority,
        "subject": subject,
        "environment": environment or "specific grounded place",
        "action": action or "specific visible action",
        "emotion": emotion or "inner tension",
        "body_pose": body_pose or "specific readable body posture",
        "key_details": key_details,
        "background_details": background_details,
        "body_language_details": body_language_details,
        "camera_framing": camera_framing or DEFAULT_CAMERA_BY_TYPE.get(scene_type, "medium close-up"),
        "composition_notes": composition_notes,
        "reason": clean_text_fragment(scene.get("reason")),
    }


def join_list_as_clause(prefix: str, items: List[str], max_items: int = 4) -> str:
    cleaned = [clean_text_fragment(x) for x in items if clean_text_fragment(x)]
    if not cleaned:
        return ""
    return f"{prefix}: " + "; ".join(cleaned[:max_items])


def build_final_prompt(scene: Dict[str, Any]) -> str:
    # Минималистичный промпт: только самое важное для визуальной передачи смысла текста
    parts = [
        scene["subject"],
        scene["action"],
        scene["environment"],
        f"{scene['emotion']}",
        f"{scene['camera_framing']}",
        GLOBAL_NEGATIVE_TEMPLATE,
    ]

    prompt = ", ".join([p for p in parts if p])
    prompt = re.sub(r"\s+,", ",", prompt)
    prompt = re.sub(r"\s+", " ", prompt).strip()
    return prompt


# ============================================================
# 11) ANALYSIS PROMPTS
# ============================================================
def analyze_global_context(client: OpenAI, scenario_text: str) -> Dict[str, Any]:
    system_prompt = (
        "You analyze scripts for AI-generated psychological videos. "
        "The video uses one recurring character throughout. "
        "Return JSON only."
    )

    user_prompt = f"""
Analyze the scenario and return JSON with this exact structure:
{{
  "topic": "",
  "psychological_theme": "",
  "core_conflict": "",
  "emotional_arc": [""],
  "dominant_emotions": [""],
  "visual_motifs": [""],
  "recurring_objects": [""],
  "suggested_environments": [""],
  "character_profile": "",
  "scene_direction_notes": "",
  "key_ideas": ["list of 5-10 core ideas/messages from the text that scenes should visually communicate"]
}}

Requirements:
- Adapt the scenario to ONE recurring character in all scenes.
- Do not describe visual style, render style, lighting preset, artists, or genre aesthetics.
- Focus on psychology, emotion, behavior, body language, and symbolic physical action.
- emotional_arc: 4-8 short stages of inner development.
- dominant_emotions: 5-8 words or short phrases (keep it concise).
- visual_motifs: 3-5 motifs max, each tied to a specific idea from the text.
- recurring_objects: 3-5 concrete objects only.
- suggested_environments: specific real places, not just "room" or "interior". Max 6.
- key_ideas: the actual messages/arguments the speaker makes — these must be visualized in scenes.
- Write all fields in English.

SCENARIO:
{scenario_text}
""".strip()

    result = call_model_json(client, system_prompt, user_prompt)
    return {
        "topic": clean_text_fragment(result.get("topic")),
        "psychological_theme": clean_text_fragment(result.get("psychological_theme")),
        "core_conflict": clean_text_fragment(result.get("core_conflict")),
        "emotional_arc": clean_string_list(result.get("emotional_arc")),
        "dominant_emotions": clean_string_list(result.get("dominant_emotions")),
        "visual_motifs": clean_string_list(result.get("visual_motifs")),
        "recurring_objects": clean_string_list(result.get("recurring_objects")),
        "suggested_environments": clean_string_list(result.get("suggested_environments")),
        "character_profile": clean_text_fragment(result.get("character_profile")),
        "scene_direction_notes": clean_text_fragment(result.get("scene_direction_notes")),
        "key_ideas": clean_string_list(result.get("key_ideas")),
    }


def plan_scenes_for_block(
    client: OpenAI,
    block: SceneBlock,
    global_context: Dict[str, Any],
    recent_scenes: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    scene_type_reference = build_scene_type_reference()
    variation_pool = pick_variation_pool(block.block_id, global_context)
    recent_summary = summarize_recent_scenes(recent_scenes)
    avoid_scene_types = repeated_scene_types_to_avoid(recent_scenes)

    system_prompt = (
        "You create video generation scenes for a psychological video with ONE recurring character. "
        "Return JSON only. No prose outside JSON. "
        "No image style language, no artist names, no render language."
    )

    key_ideas_str = "\n".join(f"- {idea}" for idea in global_context.get("key_ideas", []))

    user_prompt = f"""
GLOBAL CONTEXT:
{json.dumps(global_context, ensure_ascii=False)}

KEY IDEAS FROM THE SCRIPT (scenes must visually embody these):
{key_ideas_str or "(none extracted)"}

AVAILABLE SCENE TYPES:
{scene_type_reference}

RECENT SCENES TO AVOID REPEATING:
{json.dumps(recent_summary, ensure_ascii=False)}

SCENE TYPES OVERUSED RECENTLY (avoid):
{json.dumps(avoid_scene_types, ensure_ascii=False)}

VARIATION POOL:
{json.dumps(variation_pool, ensure_ascii=False)}

BLOCK TEXT (spoken aloud in the video — scenes must visually convey this meaning):
{block.text}

ESTIMATED BLOCK DURATION: {block.est_seconds:.2f} seconds
REQUIRED NUMBER OF SCENES: {block.scene_budget}

Your PRIMARY task: each scene must visually communicate what is SAID in the block text.
Not a literal illustration, but a visual metaphor or physical action that carries the same meaning.
Ask yourself: if someone watches this scene without sound, will they feel/understand the idea of the text?

Return JSON:
{{
  "block_summary": "",
  "emotion_stage": "",
  "scenes": [
    {{
      "scene_type": "one_of_available_scene_types",
      "detail_priority": "detailed|balanced",
      "subject": "what exactly is visible, must begin with this character",
      "environment": "specific grounded place",
      "action": "one clear visible action that embodies the meaning of the block text",
      "emotion": "dominant inner state in this shot",
      "body_pose": "specific posture",
      "key_details": ["2 to 3 concrete visible details only"],
      "background_details": ["1 to 2 visible environment details"],
      "body_language_details": ["1 to 2 body language details"],
      "camera_framing": "specific framing",
      "composition_notes": "brief note",
      "reason": "one sentence: how this scene visually conveys the block text meaning"
    }}
  ]
}}

Rules:
- scenes array must contain EXACTLY {block.scene_budget} scenes.
- All fields in English.
- One recurring character only. No extra people.
- subject must begin with "this character".
- No style words: no cinematic, film grain, HDR, painting, anime, hyperrealistic, lens brand, artist names.
- No generic environments ("room", "interior"). No generic actions ("looking away", "standing quietly").
- Each scene must have a DIFFERENT physical situation from recent scenes.
- Keep details minimal: 2-3 key details maximum. No exhaustive lists.
- The action must directly embody or metaphorically represent the spoken text content.
""".strip()

    result = call_model_json(client, system_prompt, user_prompt)
    scenes = result.get("scenes", [])
    if not isinstance(scenes, list):
        scenes = []

    cleaned_scenes: List[Dict[str, Any]] = []
    rolling_recent: List[Dict[str, Any]] = list(recent_scenes[-DIVERSITY_MEMORY_WINDOW:])

    for local_scene_index, scene in enumerate(scenes, start=1):
        if not isinstance(scene, dict):
            continue
        normalized = ensure_scene_specificity(
            scene,
            recent_scenes=rolling_recent,
            variation_pool=variation_pool,
            block_id=block.block_id,
            local_scene_index=local_scene_index,
        )
        cleaned_scenes.append(normalized)
        rolling_recent.append(normalized)

    while len(cleaned_scenes) < block.scene_budget:
        local_scene_index = len(cleaned_scenes) + 1
        fallback = fallback_from_pool(block.block_id, local_scene_index, variation_pool)
        normalized = ensure_scene_specificity(
            fallback,
            recent_scenes=rolling_recent,
            variation_pool=variation_pool,
            block_id=block.block_id,
            local_scene_index=local_scene_index,
        )
        cleaned_scenes.append(normalized)
        rolling_recent.append(normalized)

    cleaned_scenes = cleaned_scenes[: block.scene_budget]

    return {
        "block_id": block.block_id,
        "block_text": block.text,
        "sentence_indexes": block.sentence_indexes,
        "est_seconds": block.est_seconds,
        "scene_budget": block.scene_budget,
        "block_summary": clean_text_fragment(result.get("block_summary")),
        "emotion_stage": clean_text_fragment(result.get("emotion_stage")),
        "variation_pool": variation_pool,
        "recent_scenes_used_for_avoidance": recent_summary,
        "scenes": cleaned_scenes,
    }


# ============================================================
# 12) STATS / QUALITY CHECK
# ============================================================
def normalize_for_stats(value: Any) -> str:
    return clean_text_fragment(value).lower()


def build_diversity_report(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    scene_type_counts = Counter(normalize_for_stats(r.get("scene_type")) for r in records)
    env_counts = Counter(normalize_for_stats(r.get("environment")) for r in records)
    action_counts = Counter(normalize_for_stats(r.get("action")) for r in records)
    camera_counts = Counter(normalize_for_stats(r.get("camera_framing")) for r in records)

    total = len(records)
    return {
        "scene_type_counts": dict(scene_type_counts.most_common()),
        "top_repeated_environments": dict(env_counts.most_common(10)),
        "top_repeated_actions": dict(action_counts.most_common(10)),
        "top_repeated_camera_framings": dict(camera_counts.most_common(10)),
        "unique_scene_types": len(scene_type_counts),
        "unique_environments": len(env_counts),
        "unique_actions": len(action_counts),
        "unique_camera_framings": len(camera_counts),
        "unique_environment_share": round(len(env_counts) / total, 4) if total else 0,
        "unique_action_share": round(len(action_counts) / total, 4) if total else 0,
    }


# ============================================================
# 13) MAIN
# ============================================================
def main() -> None:
    random.seed(RANDOM_SEED)
    ensure_base_dir()

    scenario_file = find_scenario_file()
    scenario_text = read_text(scenario_file)
    audio_file = find_audio_file()
    audio_seconds = get_audio_duration(audio_file, scenario_text)

    info(f"Рабочая папка: {BASE_DIR}")
    info(f"Найден сценарий: {scenario_file.name}")
    if audio_file:
        info(f"Найдено аудио: {audio_file.name}")
    else:
        info("Аудио не найдено, длина будет рассчитана по объёму текста.")

    units = estimate_sentence_durations(scenario_text, audio_seconds)
    blocks = build_blocks(units)

    info(f"Всего предложений: {len(units)}")
    info(f"Всего блоков: {len(blocks)}")
    total_scene_budget = sum(block.scene_budget for block in blocks)
    info(f"Планируемое число сцен: {total_scene_budget}")
    info(f"Модели-кандидаты: {', '.join(MODEL_CANDIDATES)}")
    info(f"Температура генерации: {GENERATION_TEMPERATURE}")

    client = get_client()

    info("Анализирую общий контекст сценария...")
    global_context = analyze_global_context(client, scenario_text)
    OUTPUT_CONTEXT.write_text(json.dumps(global_context, ensure_ascii=False, indent=2), encoding="utf-8")

    block_outputs: List[Dict[str, Any]] = []
    detailed_outputs: List[Dict[str, Any]] = []
    detailed_done = 0

    recent_memory: deque = deque(maxlen=DIVERSITY_MEMORY_WINDOW)

    for block in blocks:
        info(
            f"Block {block.block_id}/{len(blocks)} | sentences {block.sentence_indexes} | "
            f"~{block.est_seconds:.2f}s | scenes={block.scene_budget}"
        )

        block_plan = plan_scenes_for_block(
            client=client,
            block=block,
            global_context=global_context,
            recent_scenes=list(recent_memory),
        )

        block_record = {
            "block_id": block.block_id,
            "text": block.text,
            "sentence_indexes": block.sentence_indexes,
            "est_seconds": block.est_seconds,
            "scene_budget": block.scene_budget,
            "block_summary": block_plan.get("block_summary", ""),
            "emotion_stage": block_plan.get("emotion_stage", ""),
            "variation_pool": block_plan.get("variation_pool", {}),
            "recent_scenes_used_for_avoidance": block_plan.get("recent_scenes_used_for_avoidance", []),
            "scenes": [],
        }

        for local_scene_index, scene in enumerate(block_plan["scenes"], start=1):
            final_prompt = build_final_prompt(scene)
            is_detail = scene["detail_priority"] == "detailed" or scene["scene_type"] in DETAILED_SCENE_TYPES
            detailed_done += 1 if is_detail else 0

            scene_record = {
                "global_scene_index": len(detailed_outputs) + 1,
                "block_id": block.block_id,
                "scene_index_in_block": local_scene_index,
                "sentence_indexes": block.sentence_indexes,
                "block_est_seconds": block.est_seconds,
                "block_text": block.text,
                "scene_type": scene["scene_type"],
                "detail_priority": scene["detail_priority"],
                "is_detailed_scene": is_detail,
                "subject": scene["subject"],
                "environment": scene["environment"],
                "action": scene["action"],
                "emotion": scene["emotion"],
                "body_pose": scene["body_pose"],
                "key_details": scene["key_details"],
                "background_details": scene["background_details"],
                "body_language_details": scene["body_language_details"],
                "camera_framing": scene["camera_framing"],
                "composition_notes": scene["composition_notes"],
                "reason": scene.get("reason", ""),
                "final_prompt": final_prompt,
            }
            detailed_outputs.append(scene_record)
            block_record["scenes"].append(scene_record)
            recent_memory.append(scene_record)

        block_outputs.append(block_record)

    OUTPUT_BLOCKS.write_text(json.dumps(block_outputs, ensure_ascii=False, indent=2), encoding="utf-8")
    OUTPUT_JSON.write_text(json.dumps(detailed_outputs, ensure_ascii=False, indent=2), encoding="utf-8")
    OUTPUT_TXT.write_text("\n".join(item["final_prompt"] for item in detailed_outputs), encoding="utf-8")

    diversity_report = build_diversity_report(detailed_outputs)

    stats = {
        "total_sentences": len(units),
        "total_blocks": len(blocks),
        "total_scenes": len(detailed_outputs),
        "detailed_scenes": detailed_done,
        "detailed_scene_share": round(detailed_done / len(detailed_outputs), 4) if detailed_outputs else 0,
        "audio_seconds": round(audio_seconds, 2),
        "reference_character_mode": "single_character_reference_only",
        "style_in_prompt": False,
        "image_prompts_generated": False,
        "generation_temperature": GENERATION_TEMPERATURE,
        "presence_penalty": PRESENCE_PENALTY,
        "frequency_penalty": FREQUENCY_PENALTY,
        "diversity_memory_window": DIVERSITY_MEMORY_WINDOW,
        "diversity_report": diversity_report,
    }
    OUTPUT_STATS.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    info(f"Сохранено: {OUTPUT_TXT}")
    info(f"Сохранено: {OUTPUT_JSON}")
    info(f"Сохранено: {OUTPUT_BLOCKS}")
    info(f"Сохранено: {OUTPUT_CONTEXT}")
    info(f"Сохранено: {OUTPUT_STATS}")
    info("Готово. Проверь prompt_generation_stats.json: там есть отчёт по разнообразию сцен.")


if __name__ == "__main__":
    main()
