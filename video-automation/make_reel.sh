#!/usr/bin/env bash
# make_reel.sh — автоматическая сборка рекламного рилса для бота "Photo LIKE | ИИ ФОТО"
#
# Структура ролика (как в референсе IMG_1645.MOV):
#   [1] Хук      — оживлённое фото (видео) + текст-крючок поверх
#   [2] Демо     — постоянный скринкаст бота (одинаковый во всех роликах)
#   [3] Результат — итоговое фото с медленным зумом
#
# Использование:
#   ./make_reel.sh -a hook.mp4 -t hook_text.txt -d bot_demo.mp4 -p final.jpg -o out.mp4 [-m music.mp3] [опции]
#
# Обязательные аргументы:
#   -a  видео с оживлённым фото (любое разрешение, будет приведено к 1080x1920)
#   -t  файл с текстом хука (UTF-8, переносы строк = переносы в кадре)
#   -d  скринкаст бота (постоянная часть)
#   -p  итоговое фото (jpg/png)
#   -o  выходной файл
#
# Необязательные:
#   -m  музыка (mp3/m4a). Если не задана — ролик без звука
#   -1  длительность хука, сек (по умолчанию 2.0; если исходник короче — берётся весь)
#   -2  длительность демо, сек (по умолчанию 3.3)
#   -3  длительность финального фото, сек (по умолчанию 2.0)
#   -f  путь к шрифту ttf (по умолчанию DejaVuSans-Bold)
#   -s  размер шрифта (по умолчанию 58)
set -euo pipefail

FONT="/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONTSIZE=58
D1=2.0; D2=3.3; D3=2.0
MUSIC=""

while getopts "a:t:d:p:o:m:1:2:3:f:s:" opt; do
  case $opt in
    a) HOOK="$OPTARG";;
    t) TEXTFILE="$OPTARG";;
    d) DEMO="$OPTARG";;
    p) PHOTO="$OPTARG";;
    o) OUT="$OPTARG";;
    m) MUSIC="$OPTARG";;
    1) D1="$OPTARG";;
    2) D2="$OPTARG";;
    3) D3="$OPTARG";;
    f) FONT="$OPTARG";;
    s) FONTSIZE="$OPTARG";;
    *) exit 1;;
  esac
done

for v in HOOK TEXTFILE DEMO PHOTO OUT; do
  [ -n "${!v:-}" ] || { echo "Не задан обязательный аргумент: $v" >&2; exit 1; }
done

W=1080; H=1920; FPS=30
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

# Приведение к вертикали 1080x1920: масштаб с заполнением + центральный кроп
NORM="scale=${W}:${H}:force_original_aspect_ratio=increase,crop=${W}:${H},fps=${FPS},setsar=1,format=yuv420p"

# --- [1] Хук: оживлённое фото + текст ---
# Текст: белый с чёрной обводкой, по центру, в верхней трети кадра (как в референсе)
DRAW="drawtext=fontfile='${FONT}':textfile='${TEXTFILE}':fontsize=${FONTSIZE}:fontcolor=white:borderw=6:bordercolor=black@0.9:x=(w-text_w)/2:y=h*0.14:line_spacing=14"
# text_align=center появился в ffmpeg 7; в 6.x блок центрируется целиком
ffmpeg -y -v error -i "$HOOK" -t "$D1" \
  -vf "${NORM},${DRAW}" -an -c:v libx264 -preset fast -crf 18 "$TMP/seg1.mp4"

# --- [2] Демо бота (постоянная часть) ---
ffmpeg -y -v error -i "$DEMO" -t "$D2" \
  -vf "$NORM" -an -c:v libx264 -preset fast -crf 18 "$TMP/seg2.mp4"

# --- [3] Финальное фото с медленным зумом ---
FRAMES=$(python3 -c "print(int($D3*$FPS))")
ffmpeg -y -v error -loop 1 -i "$PHOTO" -t "$D3" \
  -vf "scale=$((W*2)):-2,zoompan=z='min(1+0.0012*on,1.12)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=${FRAMES}:s=${W}x${H}:fps=${FPS},setsar=1,format=yuv420p" \
  -frames:v "$FRAMES" -an -c:v libx264 -preset fast -crf 18 "$TMP/seg3.mp4"

# --- Склейка ---
printf "file '%s'\nfile '%s'\nfile '%s'\n" "$TMP/seg1.mp4" "$TMP/seg2.mp4" "$TMP/seg3.mp4" > "$TMP/list.txt"
ffmpeg -y -v error -f concat -safe 0 -i "$TMP/list.txt" -c copy "$TMP/joined.mp4"

# --- Музыка ---
if [ -n "$MUSIC" ]; then
  TOTAL=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$TMP/joined.mp4")
  FADE_ST=$(python3 -c "print(max(0, float('$TOTAL')-0.8))")
  ffmpeg -y -v error -i "$TMP/joined.mp4" -stream_loop -1 -i "$MUSIC" \
    -filter_complex "[1:a]atrim=0:${TOTAL},afade=t=out:st=${FADE_ST}:d=0.8,volume=1.0[a]" \
    -map 0:v -map "[a]" -c:v copy -c:a aac -b:a 192k -shortest "$OUT"
else
  cp "$TMP/joined.mp4" "$OUT"
fi

echo "Готово: $OUT ($(ffprobe -v error -show_entries format=duration -of csv=p=0 "$OUT") сек)"
