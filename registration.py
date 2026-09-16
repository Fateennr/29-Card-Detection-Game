"""Table registration: map a drifting hand-held view onto a canonical top-down table.

The gameplay footage is shot hand-held at an oblique angle, so the table wanders
around the frame between (and within) tricks. Every downstream notion of "where a
card is" -- which arm of the trick cross, the trump spot, the counter spot -- is
only meaningful in a frame of reference attached to the *table*, not to the image.

The round table projects to an ellipse. We fit that ellipse on the cane (rattan)
surface, then apply the affine map that turns the ellipse back into a circle of
radius ``CANON_RADIUS`` centred in a ``CANON_SIZE`` square. That removes the
translation, scale and foreshortening drift, which is what actually moves between
frames. It deliberately stops short of a full perspective rectification: recovering
a true homography from a single conic is ambiguous without a second cue, and the
affine approximation is both stable and sufficient for region assignment.

In-plane orientation is preserved (the ellipse's tilt is re-applied after the
circularising scale) so that "players sit along the top edge" stays true in
canonical space.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# Canonical table space: a CANON_SIZE x CANON_SIZE image with the table drawn as a
# centred circle of radius CANON_RADIUS.
CANON_SIZE = 512
CANON_RADIUS = 230.0

# Cane-surface colour gate in HSV. The table top is a warm tan/orange weave; the
# floor tiles around it are desaturated pink-grey and the rim is near-black.
_CANE_LOWER = np.array([5, 60, 80], dtype=np.uint8)
_CANE_UPPER = np.array([35, 255, 255], dtype=np.uint8)

# Plausibility gates on a candidate ellipse, as fractions of frame area / extent.
_MIN_AREA_FRAC = 0.04
_MAX_AREA_FRAC = 0.90
_MAX_AXIS_RATIO = 4.0


@dataclass(frozen=True)
class TableFit:
    """A fitted table ellipse plus the affine map into canonical table space."""

    cx: float
    cy: float
    major: float  # full length of the major axis, pixels
    minor: float  # full length of the minor axis, pixels
    angle: float  # major-axis tilt in degrees, OpenCV convention
    score: float  # fraction of the ellipse actually covered by cane pixels

    @property
    def center(self) -> tuple[float, float]:
        return self.cx, self.cy

    def to_canonical(self) -> np.ndarray:
        """Return the 2x3 affine mapping image pixels -> canonical table space."""
        a = max(self.major / 2.0, 1e-6)
        b = max(self.minor / 2.0, 1e-6)
        theta = np.deg2rad(self.angle)

        # Rotate the major axis onto x, squash both axes to CANON_RADIUS, then put
        # the original tilt back so the table's in-image orientation is preserved.
        c, s = np.cos(theta), np.sin(theta)
        rot_to_axis = np.array([[c, s], [-s, c]])
        scale = np.diag([CANON_RADIUS / a, CANON_RADIUS / b])
        rot_back = np.array([[c, -s], [s, c]])
        linear = rot_back @ scale @ rot_to_axis

        offset = np.array([CANON_SIZE / 2.0, CANON_SIZE / 2.0]) - linear @ np.array(
            [self.cx, self.cy]
        )
        return np.hstack([linear, offset.reshape(2, 1)])

    def warp(self, frame: np.ndarray) -> np.ndarray:
        """Warp a frame into canonical table space."""
        return cv2.warpAffine(
            frame,
            self.to_canonical(),
            (CANON_SIZE, CANON_SIZE),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )

    def points_to_canonical(self, pts: np.ndarray) -> np.ndarray:
        """Map an (N,2) array of image points into canonical table space."""
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        m = self.to_canonical()
        return pts @ m[:, :2].T + m[:, 2]

    def points_from_canonical(self, pts: np.ndarray) -> np.ndarray:
        """Map an (N,2) array of canonical points back into image pixels."""
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        m = self.to_canonical()
        inv = np.linalg.inv(m[:, :2])
        return (pts - m[:, 2]) @ inv.T


def _cane_mask(frame: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, _CANE_LOWER, _CANE_UPPER)
    # The weave is full of holes; close hard enough to make the top a solid blob.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask


def fit_table(frame: np.ndarray) -> TableFit | None:
    """Fit the table ellipse in a single frame, or None if no plausible table."""
    h, w = frame.shape[:2]
    mask = _cane_mask(frame)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    frame_area = float(h * w)
    best: TableFit | None = None
    for contour in contours:
        if len(contour) < 5:
            continue
        # Cards and hands sit *on* the table and punch holes in the cane mask, so
        # use the convex hull -- we want the outline of the top, not of the weave.
        hull = cv2.convexHull(contour)
        if cv2.contourArea(hull) < _MIN_AREA_FRAC * frame_area:
            continue
        if len(hull) < 5:
            continue

        (cx, cy), (axis_a, axis_b), angle = cv2.fitEllipse(hull)
        major, minor = max(axis_a, axis_b), min(axis_a, axis_b)
        if minor <= 1.0 or major / minor > _MAX_AXIS_RATIO:
            continue

        ellipse_area = np.pi * (major / 2.0) * (minor / 2.0)
        if not (_MIN_AREA_FRAC * frame_area <= ellipse_area <= _MAX_AREA_FRAC * frame_area):
            continue

        # How much of the fitted ellipse is really cane? Guards against latching
        # onto a wooden bench or a stretch of warm-toned floor.
        probe = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(probe, ((cx, cy), (axis_a, axis_b), angle), 255, -1)
        probe_area = float(np.count_nonzero(probe))
        if probe_area < 1.0:
            continue
        score = float(np.count_nonzero(cv2.bitwise_and(probe, mask))) / probe_area
        if score < 0.45:
            continue

        # fitEllipse reports the angle of axis_a; re-express it for the major axis.
        major_angle = angle if axis_a >= axis_b else angle + 90.0
        candidate = TableFit(cx, cy, major, minor, major_angle % 180.0, score)
        if best is None or ellipse_area > best.major * best.minor * np.pi / 4.0:
            best = candidate

    return best


class TableTracker:
    """Temporally smoothed table fit.

    Per-frame fits jitter and occasionally fail outright (a hand sweeping across
    the top, a bad exposure). This keeps an exponential moving average and holds
    the last good fit through short dropouts, so canonical coordinates stay
    continuous across a trick.
    """

    def __init__(self, alpha: float = 0.2, max_hold: int = 90) -> None:
        self.alpha = alpha
        self.max_hold = max_hold
        self._state: TableFit | None = None
        self._misses = 0

    @property
    def current(self) -> TableFit | None:
        return self._state

    def update(self, frame: np.ndarray) -> TableFit | None:
        fit = fit_table(frame)
        if fit is None:
            self._misses += 1
            if self._misses > self.max_hold:
                self._state = None
            return self._state

        self._misses = 0
        if self._state is None:
            self._state = fit
            return self._state

        a = self.alpha
        prev = self._state
        # Average the tilt on the doubled angle: the major axis is 180-periodic, so
        # naive averaging would tear when the fit crosses 0/180.
        prev_r, cur_r = np.deg2rad(prev.angle * 2), np.deg2rad(fit.angle * 2)
        blended = np.arctan2(
            (1 - a) * np.sin(prev_r) + a * np.sin(cur_r),
            (1 - a) * np.cos(prev_r) + a * np.cos(cur_r),
        )
        self._state = TableFit(
            cx=(1 - a) * prev.cx + a * fit.cx,
            cy=(1 - a) * prev.cy + a * fit.cy,
            major=(1 - a) * prev.major + a * fit.major,
            minor=(1 - a) * prev.minor + a * fit.minor,
            angle=float(np.rad2deg(blended) / 2.0) % 180.0,
            score=(1 - a) * prev.score + a * fit.score,
        )
        return self._state
