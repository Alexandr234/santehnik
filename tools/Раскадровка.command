#!/bin/bash
# ============================================================================
#  Раскадровка.command — двойной клик, ничего ставить не нужно.
#
#  Скринит каждое видео из папки каждые 2 секунды (30 кадров/минуту),
#  складывает кадры по порядку и собирает PDF (сетка с таймкодами) + ZIP.
#
#  Папка с видео по умолчанию:  ~/Desktop/ФАБРИКА ВИДЕО
#  Можно перетащить папку с видео прямо на этот файл.
#
#  ffmpeg скачивается автоматически один раз в папку рядом со скриптом (_bin),
#  без Homebrew и без пароля. PDF собирает встроенный python3 (только stdlib).
# ============================================================================

INTERVAL=2          # секунд между кадрами
WIDTH=720           # ширина кадра в пикселях (для компактности)
COLS=5              # колонок в PDF-сетке
ROWS=6              # строк в PDF-сетке (5x6 = 30 кадров = 1 минута на страницу)

set -o pipefail
SELF_DIR="$(cd "$(dirname "$0")" && pwd)"

# первым аргументом можно передать/перетащить папку с видео
INPUT="${1:-$HOME/Desktop/ФАБРИКА ВИДЕО}"

echo "============================================================"
echo "  Раскадровка видео"
echo "============================================================"

if [ ! -d "$INPUT" ]; then
  echo "❌ Папка не найдена: $INPUT"
  echo "   Положите видео в  ~/Desktop/ФАБРИКА ВИДЕО  или перетащите папку на этот файл."
  echo; read -n 1 -s -r -p "Нажмите любую клавишу для выхода..."; echo; exit 1
fi

# ---- ffmpeg: берём системный, иначе качаем статический разово ----------------
FFMPEG="$(command -v ffmpeg 2>/dev/null)"
if [ -z "$FFMPEG" ]; then
  BIN="$SELF_DIR/_bin"; mkdir -p "$BIN"
  FFMPEG="$BIN/ffmpeg"
  if [ ! -x "$FFMPEG" ]; then
    echo "⬇️  Первый запуск: скачиваю ffmpeg (~30 МБ, один раз)..."
    if curl -fL --retry 3 -o "$BIN/ffmpeg.zip" "https://evermeet.cx/ffmpeg/getrelease/ffmpeg/zip"; then
      /usr/bin/unzip -o -j "$BIN/ffmpeg.zip" -d "$BIN" >/dev/null
      rm -f "$BIN/ffmpeg.zip"
      chmod +x "$FFMPEG" 2>/dev/null
      xattr -dr com.apple.quarantine "$FFMPEG" 2>/dev/null
    fi
    if [ ! -x "$FFMPEG" ]; then
      echo "❌ Не удалось скачать ffmpeg. Проверьте интернет и запустите ещё раз."
      echo; read -n 1 -s -r -p "Нажмите любую клавишу для выхода..."; echo; exit 1
    fi
  fi
fi
echo "🔧 ffmpeg: $FFMPEG"

# ---- python3 для сборки PDF (скрипт вшит в этот файл) -------------------------
PYTHON="$(command -v python3 2>/dev/null)"
PDF_PY=""
if [ -z "$PYTHON" ]; then
  echo "⚠️  Не найден python3 — PDF собран не будет (кадры и ZIP всё равно сделаю)."
  echo "   Чтобы включить PDF, установите Command Line Tools:  xcode-select --install"
else
  PDF_PY="$(mktemp -t make_pdf).py"
  # распаковываем встроенный генератор PDF во временный файл
  sed -n '/^# <<<PDFPY$/,/^# PDFPY>>>$/p' "$0" | sed '1d;$d' > "$PDF_PY"
fi

OUT_ROOT="$INPUT/_РАСКАДРОВКА"
mkdir -p "$OUT_ROOT"

echo "📂 Видео из:   $INPUT"
echo "🖼  Интервал:  каждые ${INTERVAL} c  ->  $((60 / INTERVAL)) кадров/минуту"
echo "📦 Результат:  $OUT_ROOT"
echo

shopt -s nullglob nocaseglob
VIDEOS=()
for f in "$INPUT"/*.{mp4,mov,m4v,avi,mkv,webm,mpg,mpeg,flv,wmv}; do
  [ -f "$f" ] && VIDEOS+=("$f")
done
shopt -u nullglob nocaseglob

if [ ${#VIDEOS[@]} -eq 0 ]; then
  echo "❌ В папке нет видеофайлов."
  echo; read -n 1 -s -r -p "Нажмите любую клавишу для выхода..."; echo; exit 1
fi
echo "Найдено видео: ${#VIDEOS[@]}"

for VIDEO in "${VIDEOS[@]}"; do
  BASE="$(basename "$VIDEO")"
  STEM="${BASE%.*}"
  SAFE="$(echo "$STEM" | tr ' /:' '___')"
  VDIR="$OUT_ROOT/$SAFE"
  FRAMES="$VDIR/frames"
  mkdir -p "$FRAMES"
  rm -f "$FRAMES"/frame_*.jpg

  echo
  echo "🎬 $BASE"
  "$FFMPEG" -hide_banner -loglevel error -i "$VIDEO" \
     -vf "fps=1/${INTERVAL},scale=${WIDTH}:-2" -q:v 3 \
     "$FRAMES/frame_%05d.jpg" </dev/null

  COUNT=$(ls "$FRAMES"/frame_*.jpg 2>/dev/null | wc -l | tr -d ' ')
  if [ "$COUNT" -eq 0 ]; then
    echo "   ⚠️  кадры не извлечены — пропуск"
    continue
  fi
  echo "   кадров: $COUNT"

  # ZIP кадров
  ( cd "$VDIR" && /usr/bin/zip -q -r "${SAFE}_кадры.zip" frames )
  echo "   🗜  ZIP:  $VDIR/${SAFE}_кадры.zip"

  # PDF (если есть python3)
  if [ -n "$PDF_PY" ]; then
    "$PYTHON" "$PDF_PY" "$FRAMES" "$VDIR/${SAFE}_кадры.pdf" \
        "$INTERVAL" "$COLS" "$ROWS" \
      && echo "   📄 PDF:  $VDIR/${SAFE}_кадры.pdf"
  fi
done

[ -n "$PDF_PY" ] && rm -f "$PDF_PY"

echo
echo "✅ Готово! Всё лежит в:"
echo "   $OUT_ROOT"
echo
read -n 1 -s -r -p "Нажмите любую клавишу для выхода..."; echo
exit 0

# ============================================================================
#  Ниже — встроенный генератор PDF на чистой стандартной библиотеке Python.
#  Он извлекается во временный файл и запускается; редактировать не нужно.
# ============================================================================
# <<<PDFPY
# -*- coding: utf-8 -*-
import sys
from pathlib import Path


def jpeg_info(data):
    i = 2
    n = len(data)
    sof = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
           0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}
    while i + 9 < n:
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
            i += 2
            continue
        seglen = (data[i + 2] << 8) | data[i + 3]
        if marker in sof:
            h = (data[i + 5] << 8) | data[i + 6]
            w = (data[i + 7] << 8) | data[i + 8]
            comp = data[i + 9]
            return w, h, comp
        i += 2 + seglen
    raise ValueError("SOF marker not found")


def human_time(seconds):
    s = int(round(seconds))
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def esc(text):
    return text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")


def build(frames_dir, out_pdf, interval, cols, rows):
    frames = sorted(frames_dir.glob("frame_*.jpg"))
    if not frames:
        raise SystemExit("нет кадров")

    margin = 14.0
    gap = 8.0
    label_h = 11.0
    thumb_w = 150.0
    per_page = cols * rows

    w0, h0, _ = jpeg_info(frames[0].read_bytes())
    thumb_h = thumb_w * h0 / w0

    cell_w = thumb_w + gap
    cell_h = thumb_h + label_h + gap
    page_w = margin * 2 + cols * cell_w - gap
    page_h = margin * 2 + rows * cell_h - gap

    objects = []

    def add(body):
        objects.append(body)
        return len(objects)

    font_obj = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    page_obj_nums = []
    pages_parent_num = len(objects) + 1
    objects.append(None)

    for start in range(0, len(frames), per_page):
        chunk = frames[start:start + per_page]
        img_refs = []
        content = []
        for idx, fpath in enumerate(chunk):
            gidx = start + idx
            raw = fpath.read_bytes()
            try:
                iw, ih, comp = jpeg_info(raw)
            except Exception:
                continue
            cs = "/DeviceGray" if comp == 1 else ("/DeviceCMYK" if comp == 4 else "/DeviceRGB")
            stream = (
                f"<< /Type /XObject /Subtype /Image /Width {iw} /Height {ih} "
                f"/ColorSpace {cs} /BitsPerComponent 8 /Filter /DCTDecode "
                f"/Length {len(raw)} >>\nstream\n"
            ).encode("latin-1") + raw + b"\nendstream"
            obj_num = add(stream)
            name = f"Im{gidx}"
            img_refs.append((name, obj_num))
            r, c = divmod(idx, cols)
            x = margin + c * cell_w
            y_top = page_h - margin - r * cell_h
            y_img = y_top - label_h - thumb_h
            content.append(
                f"q {thumb_w:.2f} 0 0 {thumb_h:.2f} {x:.2f} {y_img:.2f} cm /{name} Do Q"
            )
            label = f"#{gidx + 1}  {human_time(gidx * interval)}"
            content.append(
                f"BT /F1 8 Tf {x:.2f} {y_top - label_h + 2:.2f} Td ({esc(label)}) Tj ET"
            )
        content_bytes = ("\n".join(content)).encode("latin-1")
        content_obj = add(
            f"<< /Length {len(content_bytes)} >>\nstream\n".encode("latin-1")
            + content_bytes + b"\nendstream"
        )
        xobj = " ".join(f"/{name} {num} 0 R" for name, num in img_refs)
        page_body = (
            f"<< /Type /Page /Parent {pages_parent_num} 0 R "
            f"/MediaBox [0 0 {page_w:.2f} {page_h:.2f}] "
            f"/Resources << /Font << /F1 {font_obj} 0 R >> "
            f"/XObject << {xobj} >> >> "
            f"/Contents {content_obj} 0 R >>"
        ).encode("latin-1")
        page_obj_nums.append(add(page_body))

    kids = " ".join(f"{n} 0 R" for n in page_obj_nums)
    objects[pages_parent_num - 1] = (
        f"<< /Type /Pages /Count {len(page_obj_nums)} /Kids [{kids}] >>"
    ).encode("latin-1")
    catalog_num = add(f"<< /Type /Catalog /Pages {pages_parent_num} 0 R >>".encode("latin-1"))

    out = bytearray()
    out += b"%PDF-1.5\n%\xE2\xE3\xCF\xD3\n"
    offsets = [0] * (len(objects) + 1)
    for num in range(1, len(objects) + 1):
        offsets[num] = len(out)
        out += f"{num} 0 obj\n".encode("latin-1") + objects[num - 1] + b"\nendobj\n"
    xref_pos = len(out)
    total = len(objects) + 1
    out += f"xref\n0 {total}\n".encode("latin-1")
    out += b"0000000000 65535 f \n"
    for num in range(1, total):
        out += f"{offsets[num]:010d} 00000 n \n".encode("latin-1")
    out += (
        f"trailer\n<< /Size {total} /Root {catalog_num} 0 R >>\n"
        f"startxref\n{xref_pos}\n%%EOF\n"
    ).encode("latin-1")
    out_pdf.write_bytes(out)


if __name__ == "__main__":
    build(Path(sys.argv[1]), Path(sys.argv[2]),
          float(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))
# PDFPY>>>
