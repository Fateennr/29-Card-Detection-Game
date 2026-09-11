"""Procedurally rendered card faces for the 32-card *29* deck.

These exist so the pipeline is trainable *before* real card scans are available.
Faces are drawn geometrically rather than with a font, so nothing depends on
which glyphs the runtime happens to ship.

They are a stand-in, not a substitute: a classifier trained purely on these will
learn this synthetic look. Drop real scans into ``data/reference/<CODE>.png``
(e.g. ``JS.png``, ``10H.png``) and ``load_gallery`` prefers them automatically.
"""

from __future__ import annotations

import os

import numpy as np
from PIL import Image, ImageDraw

CARD_W, CARD_H = 140, 200
RED = (200, 30, 40)
BLACK = (25, 25, 30)

RANKS = ["7", "8", "9", "10", "J", "Q", "K", "A"]
SUITS = ["S", "H", "D", "C"]
SUIT_COLOR = {"S": BLACK, "C": BLACK, "H": RED, "D": RED}


def _diamond(draw, cx, cy, r, color):
    draw.polygon([(cx, cy - r), (cx + r * 0.72, cy), (cx, cy + r), (cx - r * 0.72, cy)], fill=color)


def _heart(draw, cx, cy, r, color):
    draw.ellipse([cx - r, cy - r * 0.95, cx, cy + r * 0.1], fill=color)
    draw.ellipse([cx, cy - r * 0.95, cx + r, cy + r * 0.1], fill=color)
    draw.polygon([(cx - r * 0.97, cy - r * 0.1), (cx + r * 0.97, cy - r * 0.1), (cx, cy + r)], fill=color)


def _spade(draw, cx, cy, r, color):
    draw.polygon([(cx, cy - r), (cx + r * 0.95, cy + r * 0.25), (cx - r * 0.95, cy + r * 0.25)], fill=color)
    draw.ellipse([cx - r, cy - r * 0.1, cx, cy + r * 0.7], fill=color)
    draw.ellipse([cx, cy - r * 0.1, cx + r, cy + r * 0.7], fill=color)
    draw.polygon([(cx - r * 0.3, cy + r), (cx + r * 0.3, cy + r), (cx, cy + r * 0.25)], fill=color)


def _club(draw, cx, cy, r, color):
    draw.ellipse([cx - r * 0.42, cy - r, cx + r * 0.42, cy - r * 0.16], fill=color)
    draw.ellipse([cx - r, cy - r * 0.3, cx - r * 0.16, cy + r * 0.54], fill=color)
    draw.ellipse([cx + r * 0.16, cy - r * 0.3, cx + r, cy + r * 0.54], fill=color)
    draw.polygon([(cx - r * 0.32, cy + r), (cx + r * 0.32, cy + r), (cx, cy + r * 0.2)], fill=color)


_PIP = {"S": _spade, "H": _heart, "D": _diamond, "C": _club}


def _draw_glyph(draw, suit, cx, cy, r, color):
    _PIP[suit](draw, cx, cy, r, color)


# Seven-segment style digits/letters, drawn as strokes so no font is required.
_SEGMENTS = {
    "7": [(0, 0, 1, 0), (1, 0, 1, 1), (1, 1, 1, 2)],
    "8": [(0, 0, 1, 0), (0, 0, 0, 1), (1, 0, 1, 1), (0, 1, 1, 1), (0, 1, 0, 2), (1, 1, 1, 2), (0, 2, 1, 2)],
    "9": [(0, 0, 1, 0), (0, 0, 0, 1), (1, 0, 1, 1), (0, 1, 1, 1), (1, 1, 1, 2), (0, 2, 1, 2)],
    "1": [(1, 0, 1, 1), (1, 1, 1, 2)],
    "0": [(0, 0, 1, 0), (0, 0, 0, 1), (1, 0, 1, 1), (0, 1, 0, 2), (1, 1, 1, 2), (0, 2, 1, 2)],
    "J": [(1, 0, 1, 1), (1, 1, 1, 2), (0, 2, 1, 2), (0, 1, 0, 2)],
    "Q": [(0, 0, 1, 0), (0, 0, 0, 1), (1, 0, 1, 1), (0, 1, 0, 2), (1, 1, 1, 2), (0, 2, 1, 2)],
    "K": [(0, 0, 0, 1), (0, 1, 0, 2), (0, 1, 1, 1), (1, 0, 0, 1), (0, 1, 1, 2)],
    "A": [(0, 0, 1, 0), (0, 0, 0, 1), (1, 0, 1, 1), (0, 1, 1, 1), (0, 1, 0, 2), (1, 1, 1, 2)],
}


def _draw_text(draw, text, x, y, w, h, color, width=3):
    """Render a rank string as segment strokes inside a (w,h) box at (x,y)."""
    per = w / (len(text) * 1.35)
    for i, ch in enumerate(text):
        ox = x + i * per * 1.35
        for (x0, y0, x1, y1) in _SEGMENTS.get(ch, []):
            draw.line(
                [(ox + x0 * per, y + y0 * h / 2), (ox + x1 * per, y + y1 * h / 2)],
                fill=color,
                width=width,
            )


# Pip layouts as (x, y) in unit card space, for the numeric ranks.
_LAYOUTS = {
    "7": [(.5, .18), (.28, .3), (.72, .3), (.28, .5), (.72, .5), (.28, .72), (.72, .72)],
    "8": [(.28, .2), (.72, .2), (.28, .4), (.72, .4), (.28, .6), (.72, .6), (.28, .8), (.72, .8)],
    "9": [(.28, .2), (.72, .2), (.28, .4), (.72, .4), (.5, .5), (.28, .6), (.72, .6), (.28, .8), (.72, .8)],
    "10": [(.28, .18), (.72, .18), (.28, .35), (.72, .35), (.5, .27), (.28, .65), (.72, .65), (.5, .73), (.28, .82), (.72, .82)],
    "A": [(.5, .5)],
}


def render_card(rank: str, suit: str) -> np.ndarray:
    """Render one card face as an RGB numpy array."""
    img = Image.new("RGB", (CARD_W, CARD_H), (250, 249, 245))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle([1, 1, CARD_W - 2, CARD_H - 2], radius=10, outline=(180, 180, 180), width=2)
    color = SUIT_COLOR[suit]

    # Corner indices, top-left and (rotated) bottom-right.
    for flip in (False, True):
        if flip:
            corner = Image.new("RGB", (34, 52), (250, 249, 245))
            cdraw = ImageDraw.Draw(corner)
            _draw_text(cdraw, rank, 3, 3, 24, 24, color, width=3)
            _draw_glyph(cdraw, suit, 17, 41, 8, color)
            img.paste(corner.rotate(180), (CARD_W - 38, CARD_H - 56))
        else:
            _draw_text(draw, rank, 7, 7, 24, 24, color, width=3)
            _draw_glyph(draw, suit, 21, 45, 8, color)

    if rank in _LAYOUTS:
        for (ux, uy) in _LAYOUTS[rank]:
            r = 20 if rank == "A" else 11
            _draw_glyph(draw, suit, ux * CARD_W, uy * CARD_H, r, color)
    else:
        # Court cards: a framed panel with a large central glyph.
        draw.rectangle([38, 42, CARD_W - 38, CARD_H - 42], outline=color, width=3)
        _draw_glyph(draw, suit, CARD_W / 2, CARD_H / 2, 26, color)
        _draw_text(draw, rank, CARD_W / 2 - 12, CARD_H / 2 - 62, 24, 24, color, width=3)

    return np.array(img)


def load_gallery(reference_dir: str = "data/reference") -> dict[str, np.ndarray]:
    """Return ``{card_code: RGB array}``, preferring real scans when present."""
    gallery: dict[str, np.ndarray] = {}
    real = 0
    for suit in SUITS:
        for rank in RANKS:
            codeval = f"{rank}{suit}"
            path = os.path.join(reference_dir, f"{codeval}.png")
            if os.path.exists(path):
                img = Image.open(path).convert("RGB").resize((CARD_W, CARD_H))
                gallery[codeval] = np.array(img)
                real += 1
            else:
                gallery[codeval] = render_card(rank, suit)
    print(f"gallery: {len(gallery)} cards ({real} real scans, {len(gallery) - real} procedural)")
    return gallery
