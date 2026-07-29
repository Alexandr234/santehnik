# Оркестраторы пайплайнов

Два скрипта, каждый по очереди запускает этапы своего пайплайна и показывает
весь их вывод в одном окне:

- `orchestrator_birds_pipeline.py` — «ПТИЦЫ РЕАЛИСТИЧНЫЕ», 5 шагов;
- `orchestrator_children_pipeline.py` — «ДЕТИ» (папка `ПРОМПТЫ`), 6 шагов.

Устроены одинаково, отличаются только константами `BASE` и `STEPS` в начале
файла, так что всё написанное ниже относится к обоим.

## Запуск

```bash
python3 tools/orchestrator_birds_pipeline.py          # окно (Tkinter)
python3 tools/orchestrator_birds_pipeline.py --nogui  # то же самое, но в консоли

python3 tools/orchestrator_children_pipeline.py
```

Можно просто дважды кликнуть по файлу в Finder, если `.py` открывается
через Python Launcher.

## Порядок шагов — «ПТИЦЫ РЕАЛИСТИЧНЫЕ»

1. `ПРОМПТЫ/doc_prompt_pipeline_universal_ru_de_es_pl.py`
2. `ВИЗУАЛ/flow_visual_batch_generator_realistic.py`
3. `flow_repair_missing_videos_100percent.py`
4. `analyze_voiceover_cutplan.py`
5. `МОНТАЖ/video_creator_times_autovenv_no_subs_realistic.py`

## Порядок шагов — «ДЕТИ»

База: `/Users/aleksandrtomilov/Desktop/ПРОМПТЫ`

1. `master_prompt_pipeline_children_enabled_no_famous_APIKEY_FIXED_v4ДЕТИ.py`
2. `СКРИПТ ГЕНЕРАЦИЯ КАРТИНОК/media_gen_character_image_pipeline_flow.py`
3. `ВИДЕО/video_from_local_images_base64.py`
4. `video_repair_characters_100percent_FLOW.py`
5. `ПЕРЕВОД+ПЕРЕОЗВУЧКА/translate_and_voice.py`
6. `audio_video_sync_montage.py`

## Как это работает

Следующий шаг стартует только после того, как предыдущий завершился успешно
(код возврата 0). Если шаг упал — пайплайн останавливается; снимите галочку
«Стоп при ошибке», чтобы он шёл дальше несмотря на сбои.

## Что умеет окно

- живой вывод шага: обычный текст — светлым, `stderr` — красным;
- чекбоксы у каждого шага — можно прогнать только часть пайплайна;
- «Стоп» гасит текущий шаг вместе со всеми его дочерними процессами
  (ffmpeg и прочее), а не только сам Python;
- поле «Ввод в скрипт» — если шаг что-то спрашивает, ответ уходит ему в stdin;
- итоговая таблица: статус и время каждого шага плюс общее время;
- каждый прогон дублируется в файл `tools/logs/` — `pipeline_ГГГГММДД_ЧЧММСС.log`
  для «ПТИЦ» и `children_ГГГГММДД_ЧЧММСС.log` для «ДЕТЕЙ», плюс кнопка
  «Сохранить лог…».

## Настройка

Пути лежат в начале файла:

```python
BASE = Path("/Users/aleksandrtomilov/Desktop/ПТИЦЫ РЕАЛИСТИЧНЫЕ")
STEPS = [ ... ]
```

Поменяйте `BASE`, если папка проекта переедет, или отредактируйте `STEPS`,
чтобы добавить/убрать этап.

Каждый шаг запускается из своей папки (`cwd` = папка скрипта), поэтому
относительные пути внутри скриптов работают как при обычном запуске.
Интерпретатор выбирается автоматически: если рядом со скриптом или в корне
проекта есть `.venv/bin/python` (или `venv/bin/python`) — берётся он, иначе
тот же Python, которым запущен оркестратор.
