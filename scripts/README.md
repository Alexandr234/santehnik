# translate_and_voice.py

Локализует англоязычный сценарий на **русский, португальский и испанский** через
Chat API OpenAI, затем озвучивает каждый перевод через Voicer API
(`https://voiceapi.csv666.ru`).

Перевод — это **локализация**, а не подстрочник: идиомы, фразеологизмы и
культурный контекст адаптируются под целевой язык. Длинный текст автоматически
режется на чанки по границам абзацев.

## Установка

```bash
pip install requests
```

## Запуск

```bash
# всё сразу: перевод + озвучка (файл по умолчанию — scenario.txt на рабочем столе)
python3 scripts/translate_and_voice.py

# свой файл
python3 scripts/translate_and_voice.py "/путь/к/scenario.txt"

# только перевод
python3 scripts/translate_and_voice.py --skip-tts

# озвучить уже готовые переводы (scenario_ru.txt и т.д.)
python3 scripts/translate_and_voice.py --skip-translate

# конкретные языки
python3 scripts/translate_and_voice.py --langs ru,es

# свой голос или готовый шаблон из Telegram-бота
python3 scripts/translate_and_voice.py --voice-id 21m00Tcm4TlvDq8ikWAM
python3 scripts/translate_and_voice.py --template-uuid <uuid>
```

## Результаты

Рядом с входным файлом (или в `--out-dir`):

- `scenario_ru.txt`, `scenario_pt.txt`, `scenario_es.txt` — переводы;
- `scenario_ru.mp3`, `scenario_pt.mp3`, `scenario_es.mp3` — озвучка
  (или `.zip`, если результат приходит архивом чанков).

## Ключи

Ключ **Voicer (TTS)** вписан прямо в скрипт.

Ключ **OpenAI** в файле не хранится: GitHub Push Protection блокирует любой push
с ключом формата OpenAI, поэтому он берётся из переменной окружения. Задайте его
перед запуском:

```bash
export OPENAI_API_KEY="sk-proj-...ваш ключ..."
python3 scripts/translate_and_voice.py
```

Либо впишите ключ в **локальную** копию строки `OPENAI_API_KEY` в начале скрипта —
только не коммитьте этот файл обратно, иначе push будет отклонён. Обе переменные
окружения (`OPENAI_API_KEY`, `TTS_API_KEY`) имеют приоритет над значениями в коде.

## Примечания

- Стоимость озвучки считается по символам (мин. 500); скрипт покажет баланс и
  предупредит, если его может не хватить на все языки.
- Португальский по умолчанию — бразильский; испанский — нейтральный
  латиноамериканский. Меняется в словаре `LANGUAGES` в начале скрипта.
- Модель озвучки по умолчанию — `eleven_v3` (мультиязычная), поэтому один голос
  работает на всех трёх языках. Модель перевода по умолчанию — `gpt-4o`
  (`--model` для смены).
