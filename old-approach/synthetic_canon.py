"""Compose synthetic training scenes directly in canonical table space.

Training and inference must share a geometry. Inference warps each real frame
onto the canonical table disc, so scenes are composited there too -- rather than
in raw frame space, which would force the detector to also absorb the hand-held
camera's drift and foreshortening.

Backgrounds are real empty-table crops harvested from the gameplay videos, so
the model sees the actual cane weave and lighting rather than a flat colour.
"""

from __future__ import annotations

import random

import cv2
import numpy as np

from p29.vision.registration import CANON_RADIUS, CANON_SIZE
from p29.vision.regions import CANON_CENTER

CARD_SCALE = 0.34  # card height as a fraction of the table radius


def _place(scene, card_rgb, cx, cy, angle, scale, brightness):
    """Alpha-composite a rotated card onto the scene; return its bbox."""
    h, w = card_rgb.shape[:2]
    target_h = max(8, int(CANON_RADIUS * CARD_SCALE * scale))
    target_w = max(6, int(target_h * w / h))
    card = cv2.resize(card_rgb, (target_w, target_h), interpolation=cv2.INTER_AREA)
    card = np.clip(card.astype(np.float32) * brightness, 0, 255).astype(np.uint8)

    mask = np.full((target_h, target_w), 255, dtype=np.uint8)
    diag = int(np.hypot(target_w, target_h)) + 4
    pad_card = np.zeros((diag, diag, 3), dtype=np.uint8)
    pad_mask = np.zeros((diag, diag), dtype=np.uint8)
    y0, x0 = (diag - target_h) // 2, (diag - target_w) // 2
    pad_card[y0:y0 + target_h, x0:x0 + target_w] = card
    pad_mask[y0:y0 + target_h, x0:x0 + target_w] = mask

    rot = cv2.getRotationMatrix2D((diag / 2, diag / 2), angle, 1.0)
    pad_card = cv2.warpAffine(pad_card, rot, (diag, diag))
    pad_mask = cv2.warpAffine(pad_mask, rot, (diag, diag))

    top, left = int(cy - diag / 2), int(cx - diag / 2)
    sy0, sx0 = max(0, top), max(0, left)
    sy1, sx1 = min(CANON_SIZE, top + diag), min(CANON_SIZE, left + diag)
    if sy1 <= sy0 or sx1 <= sx0:
        return None

    cy0, cx0 = sy0 - top, sx0 - left
    sub_card = pad_card[cy0:cy0 + (sy1 - sy0), cx0:cx0 + (sx1 - sx0)]
    sub_mask = pad_mask[cy0:cy0 + (sy1 - sy0), cx0:cx0 + (sx1 - sx0)]
    alpha = (sub_mask.astype(np.float32) / 255.0)[..., None]
    region = scene[sy0:sy1, sx0:sx1].astype(np.float32)
    scene[sy0:sy1, sx0:sx1] = (region * (1 - alpha) + sub_card.astype(np.float32) * alpha).astype(np.uint8)

    ys, xs = np.nonzero(sub_mask)
    if len(xs) == 0:
        return None
    return (sx0 + xs.min(), sy0 + ys.min(), sx0 + xs.max(), sy0 + ys.max())


def make_scene(gallery, backgrounds, rng: random.Random):
    """Build one canonical-space scene. Returns (image, [(code, bbox), ...])."""
    bg = backgrounds[rng.randrange(len(backgrounds))].copy()
    scene = cv2.resize(bg, (CANON_SIZE, CANON_SIZE))

    codes = list(gallery)
    rng.shuffle(codes)
    n_cards = rng.randint(1, 4)
    chosen = codes[:n_cards]

    # Lay the trick out as a cross, jittered, the way it actually falls on table.
    centre = (
        CANON_CENTER[0] + rng.uniform(-0.18, 0.18) * CANON_RADIUS,
        CANON_CENTER[1] + rng.uniform(0.05, 0.35) * CANON_RADIUS,
    )
    arm = rng.uniform(0.28, 0.42) * CANON_RADIUS
    slots = [(0, -arm), (arm, 0), (0, arm), (-arm, 0)]
    rng.shuffle(slots)

    global_light = rng.uniform(0.72, 1.18)
    annotations = []
    for codeval, (dx, dy) in zip(chosen, slots):
        cx = centre[0] + dx + rng.uniform(-14, 14)
        cy = centre[1] + dy + rng.uniform(-14, 14)
        box = _place(
            scene,
            gallery[codeval],
            cx,
            cy,
            angle=rng.uniform(-180, 180),
            scale=rng.uniform(0.85, 1.15),
            brightness=global_light * rng.uniform(0.92, 1.08),
        )
        if box is not None:
            annotations.append((codeval, box))

    if rng.random() < 0.5:
        k = rng.choice([3, 5])
        scene = cv2.GaussianBlur(scene, (k, k), 0)
    if rng.random() < 0.3:
        noise = rng.uniform(3, 11)
        scene = np.clip(scene.astype(np.float32) + np.random.normal(0, noise, scene.shape), 0, 255).astype(np.uint8)

    return scene, annotations


def _table_disc_mask():
    """The whole table top, not just the play area.

    Emptiness is scored over the entire table on purpose. Any real face-up card
    left in a background is an *unlabelled* card in a training image, which teaches
    the detector to ignore exactly what it is meant to find -- and that applies
    wherever on the table it sits, not only in the middle. The face-down hand piles
    are always present and so contribute a near-constant offset that ranking
    absorbs harmlessly.
    """
    yy, xx = np.mgrid[0:CANON_SIZE, 0:CANON_SIZE]
    return np.hypot(xx - CANON_CENTER[0], yy - CANON_CENTER[1]) < 0.98 * CANON_RADIUS


def harvest_backgrounds(video_paths, tracker_cls, max_per_video=60, sample_every=2.0):
    """Collect canonical table crops whose play area is as empty as possible.

    Emptiness is *ranked*, not thresholded. The cane weave's own highlights read as
    bright and desaturated, so even a bare table scores ~8% "card-like" pixels in
    the table -- any absolute cutoff either takes everything or nothing. Taking
    the lowest-scoring frames per video needs no tuned constant and adapts to
    whatever the lighting happens to be.

    Two passes, so only the selected frames are ever held in memory: score first,
    then re-read the winners using the table fit recorded alongside each score.
    """
    play = _table_disc_mask()
    backgrounds = []

    for path in video_paths:
        cap = cv2.VideoCapture(path)
        fps = max(cap.get(cv2.CAP_PROP_FPS), 1.0)
        step = max(1, int(fps * sample_every))

        tracker = tracker_cls()
        scored = []
        i = -1
        while True:  # sequential decode; seeking per sample is far slower
            if not cap.grab():
                break
            i += 1
            if i % step:
                continue
            ok, frame = cap.retrieve()
            if not ok:
                break
            fit = tracker.update(frame)
            if fit is None:
                continue
            hsv = cv2.cvtColor(fit.warp(frame), cv2.COLOR_BGR2HSV)
            cardish = (hsv[:, :, 2] > 150) & (hsv[:, :, 1] < 60)
            scored.append((float((cardish & play).sum() / play.sum()), i, fit))

        # Only the winners are re-read, so peak memory stays at max_per_video frames.
        scored.sort(key=lambda t: t[0])
        for _, i, fit in scored[:max_per_video]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, frame = cap.read()
            if ok:
                backgrounds.append(fit.warp(frame))
        cap.release()

    return backgrounds
