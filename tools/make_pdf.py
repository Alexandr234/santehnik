#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_pdf.py — собирает PDF-контактный лист из JPEG-кадров.
Только стандартная библиотека Python (никаких pip-пакетов): JPEG вставляется
в PDF напрямую (фильтр DCTDecode), таймкоды пишутся встроенным шрифтом Helvetica.

Использование:
    python3 make_pdf.py <папка_с_кадрами> <выходной.pdf> <интервал_сек> <колонок> <строк>
"""
import sys
from pathlib import Path


def jpeg_info(data: bytes):
    """Возвращает (width, height, components) из JPEG по маркеру SOF."""
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


def human_time(seconds: float) -> str:
    s = int(round(seconds))
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def esc(text: str) -> str:
    return text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")


def build(frames_dir: Path, out_pdf: Path, interval: float, cols: int, rows: int):
    frames = sorted(frames_dir.glob("frame_*.jpg"))
    if not frames:
        raise SystemExit("нет кадров")

    # геометрия страницы (в пунктах, 72 pt = 1 дюйм)
    margin = 14.0
    gap = 8.0
    label_h = 11.0
    thumb_w = 150.0
    per_page = cols * rows

    # высота миниатюры по пропорциям первого кадра
    w0, h0, _ = jpeg_info(frames[0].read_bytes())
    thumb_h = thumb_w * h0 / w0

    cell_w = thumb_w + gap
    cell_h = thumb_h + label_h + gap
    page_w = margin * 2 + cols * cell_w - gap
    page_h = margin * 2 + rows * cell_h - gap

    objects = []  # список bytes-тел объектов; индекс+1 = номер объекта

    def add(body: bytes) -> int:
        objects.append(body)
        return len(objects)  # номер объекта

    font_obj = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    page_obj_nums = []
    pages_parent_num = None  # заполним позже, но ссылки на него нужны заранее

    # зарезервируем номер объекта Pages (создадим тело в конце)
    pages_parent_num = len(objects) + 1
    objects.append(None)  # плейсхолдер, тело допишем ниже

    for start in range(0, len(frames), per_page):
        chunk = frames[start:start + per_page]
        img_refs = []  # (name, obj_num)
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
            # картинка
            content.append(
                f"q {thumb_w:.2f} 0 0 {thumb_h:.2f} {x:.2f} {y_img:.2f} cm /{name} Do Q"
            )
            # подпись-таймкод над картинкой
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

    # тело Pages
    kids = " ".join(f"{n} 0 R" for n in page_obj_nums)
    objects[pages_parent_num - 1] = (
        f"<< /Type /Pages /Count {len(page_obj_nums)} /Kids [{kids}] >>"
    ).encode("latin-1")

    catalog_num = add(f"<< /Type /Catalog /Pages {pages_parent_num} 0 R >>".encode("latin-1"))

    # сериализация с таблицей xref
    out = bytearray()
    out += b"%PDF-1.5\n%\xE2\xE3\xCF\xD3\n"
    offsets = [0] * (len(objects) + 1)
    for num in range(1, len(objects) + 1):
        offsets[num] = len(out)
        body = objects[num - 1]
        out += f"{num} 0 obj\n".encode("latin-1") + body + b"\nendobj\n"

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
    if len(sys.argv) < 6:
        raise SystemExit("usage: make_pdf.py <frames_dir> <out.pdf> <interval> <cols> <rows>")
    build(Path(sys.argv[1]), Path(sys.argv[2]),
          float(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))
