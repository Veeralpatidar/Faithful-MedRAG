
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Emu, Inches, Pt

from . import config as C

log = logging.getLogger("fmrag.deck")

# Palette
NAVY = RGBColor(0x1F, 0x3A, 0x5F)
BLUE = RGBColor(0x2E, 0x5E, 0xAA)
RED = RGBColor(0xD1, 0x49, 0x5B)
GREEN = RGBColor(0x3B, 0x7A, 0x57)
AMBER = RGBColor(0xED, 0xAE, 0x49)
GREY = RGBColor(0x5A, 0x5A, 0x5A)
LIGHT = RGBColor(0xF2, 0xF5, 0xF9)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
BLACK = RGBColor(0x12, 0x12, 0x12)

W, H = Inches(13.333), Inches(7.5)        # 16:9

# --------------------------------------------------------------------------
# Auto-fit: choose the largest font that fills the box without overflowing.
# --------------------------------------------------------------------------
import math as _math

CONTENT_TOP = Inches(1.28)      # just under the title rule
CONTENT_BOT = Inches(6.88)      # just above the footer line


def _est_lines(text, pt, width_in, bullet_pad=0.0):
    """Estimate wrapped line count for one paragraph at a given point size."""
    if not text:
        return 1
    usable = max(width_in - bullet_pad, 0.5)
    chars_per_line = max(int(usable * 96 / (pt * 0.50)), 8)
    return max(1, _math.ceil(len(text) / chars_per_line))


def _fit_size(items, width_in, height_in, max_pt=17.0, min_pt=9.5,
              line_mult=1.30, gap_pt=6.0):
    """Largest point size at which `items` (list of (text, level)) fit."""
    pt = max_pt
    while pt > min_pt:
        total = 0.0
        for text, level in items:
            size = pt - 1.3 * level
            lines = _est_lines(text, size, width_in, bullet_pad=0.30 + 0.28 * level)
            total += lines * size * line_mult / 72.0 + gap_pt / 72.0
        if total <= height_in:
            return pt
        pt -= 0.5
    return min_pt



# --------------------------------------------------------------------------
# Primitives
# --------------------------------------------------------------------------
def new_deck() -> Presentation:
    p = Presentation()
    p.slide_width, p.slide_height = W, H
    return p


def _blank(prs: Presentation):
    return prs.slides.add_slide(prs.slide_layouts[6])


def _box(slide, x, y, w, h, text, size=14, bold=False, color=BLACK,
         align=PP_ALIGN.LEFT, font="Calibri", wrap=True, spacing=1.0):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = wrap
    tf.margin_left = tf.margin_right = Inches(0.04)
    tf.margin_top = tf.margin_bottom = 0
    lines = text.split("\n") if isinstance(text, str) else list(text)
    for i, line in enumerate(lines):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.alignment = align
        para.line_spacing = spacing
        run = para.add_run()
        run.text = line
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.color.rgb = color
        run.font.name = font
    return tb


def _rect(slide, x, y, w, h, fill=LIGHT, line=None):
    from pptx.enum.shapes import MSO_SHAPE

    sh = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h)
    sh.fill.solid()
    sh.fill.fore_color.rgb = fill
    if line is None:
        sh.line.fill.background()
    else:
        sh.line.color.rgb = line
        sh.line.width = Pt(1)
    sh.shadow.inherit = False
    sh.text_frame.text = ""
    return sh


def title_slide(prs, title, subtitle, meta_lines):
    s = _blank(prs)
    _rect(s, 0, 0, W, Inches(2.9), fill=NAVY)
    _box(s, Inches(0.8), Inches(0.75), Inches(11.8), Inches(1.0), title,
         size=40, bold=True, color=WHITE)
    _box(s, Inches(0.8), Inches(1.85), Inches(11.8), Inches(0.8), subtitle,
         size=16, color=RGBColor(0xD8, 0xE2, 0xF0))
    _box(s, Inches(0.8), Inches(3.4), Inches(11.8), Inches(3.0),
         meta_lines, size=15, spacing=1.5)
    return s


def section_slide(prs, number, title):
    s = _blank(prs)
    _rect(s, 0, Inches(2.15), W, Inches(3.2), fill=NAVY)
    _box(s, Inches(1.0), Inches(2.75), Inches(1.8), Inches(1.6), str(number),
         size=56, bold=True, color=AMBER)
    _box(s, Inches(2.9), Inches(3.00), Inches(9.6), Inches(1.0), title,
         size=34, bold=True, color=WHITE)
    return s


def content_slide(prs, title, bullets=None, footer=None, sub=None):
    s = _blank(prs)
    _box(s, Inches(0.55), Inches(0.35), Inches(12.2), Inches(0.7), title,
         size=27, bold=True, color=NAVY)
    _rect(s, Inches(0.55), Inches(1.02), Inches(12.2), Emu(14000), fill=AMBER)
    y = Inches(1.3)
    if sub:
        _box(s, Inches(0.6), y, Inches(12.1), Inches(0.45), sub, size=14,
             color=GREY)
        y = Inches(1.8)
    if bullets:
        _bullets(s, bullets, Inches(0.65), y, Inches(12.0),
                 height=Emu(max(Emu(CONTENT_BOT) - Emu(y), Emu(Inches(1.0)))))
    if footer:
        _box(s, Inches(0.55), Inches(6.92), Inches(12.2), Inches(0.42), footer,
             size=10, color=GREY)
    return s


def _bullets(slide, items, x, y, w, size=None, height=None):
    """
    Render bullets, auto-sizing the font to fill the available box.

    `height` defaults to everything between y and the footer line, so content
    fills the slide instead of leaving the lower third empty.
    """
    norm = []
    for item in items:
        if isinstance(item, tuple):
            norm.append((str(item[0]), int(item[1])))
        else:
            norm.append((str(item), 0))

    h = height if height is not None else max(Emu(CONTENT_BOT) - Emu(y), Emu(Inches(1.0)))
    w_in, h_in = Emu(w) / 914400, Emu(h) / 914400

    base = float(size) if size else _fit_size(norm, w_in, h_in)
    if size:                                   # caller gave a hint: shrink if needed
        base = min(base, _fit_size(norm, w_in, h_in, max_pt=base))

    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0.03)
    tf.margin_top = tf.margin_bottom = 0

    first = True
    for text, level in norm:
        para = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        para.level = min(level, 4)
        para.space_after = Pt(6 if base > 13 else 4)
        para.line_spacing = 1.16
        run = para.add_run()
        run.text = ("" if not text else ("\u25cf  " if level == 0 else "\u2013  ")) + text
        run.font.size = Pt(max(base - 1.3 * level, 9))
        run.font.color.rgb = BLACK if level == 0 else GREY
        run.font.name = "Calibri"
    return tb


def table_slide(prs, title, df: pd.DataFrame, source: str, note=None,
                max_rows=12, fmt="{:.4f}", col_width=None, highlight_row=None):
    """Render a DataFrame as a real table, with its source file named."""
    s = _blank(prs)
    _box(s, Inches(0.55), Inches(0.35), Inches(12.2), Inches(0.7), title,
         size=26, bold=True, color=NAVY)
    _rect(s, Inches(0.55), Inches(1.02), Inches(12.2), Emu(14000), fill=AMBER)

    if df is None or df.empty:
        pending_box(s, Inches(0.55), Inches(1.6), Inches(12.2), Inches(2.0),
                    "NOT YET EXECUTED",
                    "This table has no backing results file yet. "
                    "No placeholder numbers are shown.")
        return s

    d = df.head(max_rows).copy()
    nrows, ncols = len(d) + 1, len(d.columns)

    # Reserve room for the note BELOW the table (it used to sit above and
    # collide with the table on long notes).
    note_h = 0.0
    if note:
        est_lines = max(1, int(len(note) / 118) + 1)
        note_h = est_lines * 0.185 + 0.10

    top = Inches(1.32)
    bottom_limit = 6.82 - note_h
    avail_in = max(bottom_limit - 1.32, 1.2)
    row_in = min(0.40, avail_in / nrows)
    tbl_h = Inches(row_in * nrows)

    gt = s.shapes.add_table(nrows, ncols, Inches(0.55), top,
                            Inches(12.2), tbl_h).table

    body_pt = 11.0 if nrows <= 7 else (10.0 if nrows <= 10 else 9.0)
    head_pt = body_pt + 0.5

    for j, cname in enumerate(d.columns):
        c = gt.cell(0, j)
        c.text = str(cname).replace("_", " ")
        p = c.text_frame.paragraphs[0]
        p.runs[0].font.size = Pt(head_pt)
        p.runs[0].font.bold = True
        p.runs[0].font.color.rgb = WHITE
        c.fill.solid()
        c.fill.fore_color.rgb = NAVY

    for i in range(len(d)):
        for j, cname in enumerate(d.columns):
            v = d.iloc[i][cname]
            if isinstance(v, float):
                txt = "n/a" if pd.isna(v) else fmt.format(v)
            else:
                txt = str(v)
            if len(txt) > 46:
                txt = txt[:43] + "..."
            c = gt.cell(i + 1, j)
            c.text = txt
            p = c.text_frame.paragraphs[0]
            p.runs[0].font.size = Pt(body_pt)
            p.runs[0].font.color.rgb = BLACK
            c.fill.solid()
            hl = highlight_row is not None and i == highlight_row
            c.fill.fore_color.rgb = RGBColor(0xDE, 0xEC, 0xDB) if hl else (
                LIGHT if i % 2 == 0 else WHITE)

    if col_width:
        for j, wd in enumerate(col_width):
            if j < ncols:
                gt.columns[j].width = Inches(wd)

    if note:
        _box(s, Inches(0.6), Inches(1.32 + row_in * nrows + 0.12), Inches(12.1),
             Inches(note_h), note, size=11.5, color=GREY, spacing=1.12)

    _box(s, Inches(0.55), Inches(6.92), Inches(12.2), Inches(0.42),
         f"Source: {source}", size=10, color=GREY)
    return s


def image_slide(prs, title, img: Path, explanation: str, source: str = ""):
    """A graph slide. Every graph gets an explanation of what it demonstrates."""
    s = _blank(prs)
    _box(s, Inches(0.55), Inches(0.35), Inches(12.2), Inches(0.7), title,
         size=26, bold=True, color=NAVY)
    _rect(s, Inches(0.55), Inches(1.02), Inches(12.2), Emu(14000), fill=AMBER)

    if img and Path(img).exists():
        from PIL import Image

        iw, ih = Image.open(img).size
        max_w, max_h = Inches(8.0), Inches(5.1)
        scale = min(max_w / Emu(int(iw * 9525)), max_h / Emu(int(ih * 9525)))
        w_ = Emu(int(iw * 9525 * scale))
        h_ = Emu(int(ih * 9525 * scale))
        s.shapes.add_picture(str(img), Inches(0.6), Inches(1.35), w_, h_)
        _box(s, Inches(8.85), Inches(1.45), Inches(3.95), Inches(4.8),
             explanation, size=13, spacing=1.15)
    else:
        pending_box(s, Inches(0.55), Inches(1.6), Inches(12.2), Inches(2.2),
                    "GRAPH NOT YET GENERATED",
                    f"Expected at: {img}\nRun the corresponding experiment script.")
        _box(s, Inches(0.6), Inches(4.1), Inches(12.1), Inches(2.0),
             explanation, size=13)

    if source:
        _box(s, Inches(0.55), Inches(6.92), Inches(12.2), Inches(0.42),
             f"Source: {source}", size=10, color=GREY)
    return s


def pending_box(slide, x, y, w, h, heading, detail):
    _rect(slide, x, y, w, h, fill=RGBColor(0xFF, 0xF4, 0xE5), line=AMBER)
    _box(slide, x + Inches(0.3), y + Inches(0.22), w - Inches(0.6), Inches(0.45),
         heading, size=18, bold=True, color=RGBColor(0x8A, 0x55, 0x00))
    _box(slide, x + Inches(0.3), y + Inches(0.78), w - Inches(0.6), h - Inches(1.0),
         detail, size=13, color=RGBColor(0x5A, 0x40, 0x10))


def two_col_slide(prs, title, left_head, left_items, right_head, right_items,
                  footer=None):
    s = _blank(prs)
    _box(s, Inches(0.55), Inches(0.35), Inches(12.2), Inches(0.7), title,
         size=26, bold=True, color=NAVY)
    _rect(s, Inches(0.55), Inches(1.02), Inches(12.2), Emu(14000), fill=AMBER)

    panel_h = Inches(5.45) if footer else Inches(5.75)
    _rect(s, Inches(0.55), Inches(1.30), Inches(5.95), panel_h, fill=LIGHT)
    _box(s, Inches(0.82), Inches(1.50), Inches(5.5), Inches(0.5), left_head,
         size=17, bold=True, color=NAVY)
    inner_h = Emu(panel_h) - Emu(Inches(0.95))
    _bullets(s, left_items, Inches(0.82), Inches(2.10), Inches(5.42), height=Emu(inner_h))

    _rect(s, Inches(6.83), Inches(1.30), Inches(5.95), panel_h,
          fill=RGBColor(0xEC, 0xF4, 0xEE))
    _box(s, Inches(7.10), Inches(1.50), Inches(5.5), Inches(0.5), right_head,
         size=17, bold=True, color=GREEN)
    _bullets(s, right_items, Inches(7.10), Inches(2.10), Inches(5.42), height=Emu(inner_h))

    if footer:
        _box(s, Inches(0.55), Inches(6.92), Inches(12.2), Inches(0.42), footer,
             size=10, color=GREY)
    return s


def metric_cards(prs, title, cards: list[tuple[str, str, str]], footer=None,
                 sub=None):
    """Headline-number slide. cards = [(value, label, sublabel), ...]"""
    s = _blank(prs)
    _box(s, Inches(0.55), Inches(0.35), Inches(12.2), Inches(0.7), title,
         size=26, bold=True, color=NAVY)
    _rect(s, Inches(0.55), Inches(1.02), Inches(12.2), Emu(14000), fill=AMBER)
    if sub:
        _box(s, Inches(0.6), Inches(1.2), Inches(12.1), Inches(0.5), sub,
             size=14, color=GREY)

    n = max(len(cards), 1)
    total_w = Inches(12.2)
    gap = Inches(0.22)
    cw = Emu(int((Emu(total_w) - Emu(gap) * (n - 1)) / n))
    y = Inches(2.05) if sub else Inches(1.80)
    card_h = Inches(3.55) if footer else Inches(3.95)
    val_pt = 40 if n <= 4 else 32
    for i, (value, label, sublabel) in enumerate(cards):
        x = Emu(int(Emu(Inches(0.55)) + i * (Emu(cw) + Emu(gap))))
        _rect(s, x, y, cw, card_h, fill=LIGHT)
        v = str(value)
        _box(s, x, y + Inches(0.45), cw, Inches(1.2),
             v, size=val_pt if len(v) <= 7 else val_pt - 11,
             bold=True, color=BLUE, align=PP_ALIGN.CENTER)
        _box(s, x + Inches(0.14), y + Inches(1.80),
             Emu(Emu(cw) - Emu(Inches(0.28))), Inches(0.7), label,
             size=15, bold=True, align=PP_ALIGN.CENTER)
        _box(s, x + Inches(0.14), y + Inches(2.52),
             Emu(Emu(cw) - Emu(Inches(0.28))), Inches(0.9), sublabel,
             size=12, color=GREY, align=PP_ALIGN.CENTER)
    if footer:
        _box(s, Inches(0.55), Inches(6.92), Inches(12.2), Inches(0.42), footer,
             size=10, color=GREY)
    return s


# --------------------------------------------------------------------------
# Result loading -- returns None when a file is absent, never a fake number
# --------------------------------------------------------------------------
def load_result(phase_dir: Path, name: str) -> pd.DataFrame | None:
    p = phase_dir / "Results" / f"{name}.csv"
    if not p.exists():
        log.warning("results file missing: %s", p)
        return None
    try:
        return pd.read_csv(p)
    except Exception as exc:
        log.warning("could not read %s: %s", p, exc)
        return None


def graph_path(phase_dir: Path, name: str) -> Path:
    return phase_dir / "Graphs" / f"{name}.png"


def rel(p: Path) -> str:
    try:
        return str(Path(p).relative_to(C.ROOT))
    except Exception:
        return str(p)
