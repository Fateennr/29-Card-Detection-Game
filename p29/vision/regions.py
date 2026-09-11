"""Region semantics in canonical table space.

A card's *role* in this footage is positional, not visual: the same 3 of diamonds
is a trump indicator on the trump spot and would be meaningless in the trick area.
This module turns a canonical-space position into a role, and -- for trick cards --
into a player seat.

Two distinct mechanisms, on purpose:

* **Role** (trick / trump / counter / hand) is decided by *absolute* position in
  canonical space, since those areas are fixed properties of how the table is laid
  out. They are configurable because the trump and counter spots are conventions of
  this particular group, not of the game.

* **Seat** is decided by the *bearing of a card from the centroid of the current
  trick*, not by absolute position. The four played cards land in a cross, and the
  cross as a whole shifts around the table between tricks. Measuring each card
  relative to its own trick's centre is invariant to that wander, and to any
  residual rotation the registration step leaves behind.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from .registration import CANON_RADIUS, CANON_SIZE

CANON_CENTER = (CANON_SIZE / 2.0, CANON_SIZE / 2.0)


class Role(str, Enum):
    """What a detected card means, given where it sits."""

    TRICK = "trick"  # a played card in the current trick
    TRUMP = "trump"  # the trump indicator (suit matters, rank is noise)
    COUNTER = "counter"  # one of the four 6s used as a point counter
    HAND = "hand"  # a player's face-down pile along the table edge
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Disc:
    """A circular region in canonical space, as fractions of the table radius."""

    cx: float
    cy: float
    radius: float

    def contains(self, x: float, y: float) -> bool:
        return math.hypot(x - self.cx, y - self.cy) <= self.radius


@dataclass
class TableLayout:
    """Where each role lives in canonical table space.

    Defaults are derived from the canonical persistence maps of both gameplay
    videos: three hand piles hug the top edge, and the trick cross sits in the
    lower-middle of the table. The trump and counter spots are deliberately wide
    and overridable -- they were inferred from a handful of frames, so they are
    config, not fact.
    """

    # Cards beyond this fraction of the table radius, in the upper half, are piles.
    hand_band_radius: float = 0.62
    hand_band_max_angle: float = 115.0  # degrees from "up", either side

    trick_area: Disc = field(
        default_factory=lambda: Disc(
            cx=CANON_CENTER[0] - 0.04 * CANON_RADIUS,
            cy=CANON_CENTER[1] + 0.22 * CANON_RADIUS,
            radius=0.72 * CANON_RADIUS,
        )
    )
    trump_spot: Disc | None = None
    counter_spot: Disc | None = None

    def to_json(self, path: str) -> None:
        def enc(d: Disc | None):
            return None if d is None else {"cx": d.cx, "cy": d.cy, "radius": d.radius}

        payload = {
            "hand_band_radius": self.hand_band_radius,
            "hand_band_max_angle": self.hand_band_max_angle,
            "trick_area": enc(self.trick_area),
            "trump_spot": enc(self.trump_spot),
            "counter_spot": enc(self.counter_spot),
        }
        with open(path, "w") as fh:
            json.dump(payload, fh, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "TableLayout":
        with open(path) as fh:
            payload = json.load(fh)

        def dec(d):
            return None if d is None else Disc(d["cx"], d["cy"], d["radius"])

        return cls(
            hand_band_radius=payload["hand_band_radius"],
            hand_band_max_angle=payload["hand_band_max_angle"],
            trick_area=dec(payload["trick_area"]),
            trump_spot=dec(payload["trump_spot"]),
            counter_spot=dec(payload["counter_spot"]),
        )

    def classify(self, x: float, y: float) -> Role:
        """Assign a role to a canonical-space point (centre of a detected card)."""
        # Explicitly configured spots win: they are narrow and deliberate, whereas
        # the hand band and trick area are broad catch-alls that would swallow them.
        if self.trump_spot is not None and self.trump_spot.contains(x, y):
            return Role.TRUMP
        if self.counter_spot is not None and self.counter_spot.contains(x, y):
            return Role.COUNTER

        dx, dy = x - CANON_CENTER[0], y - CANON_CENTER[1]
        radius_frac = math.hypot(dx, dy) / CANON_RADIUS
        # Bearing measured from "up" (negative y), positive clockwise.
        bearing = math.degrees(math.atan2(dx, -dy))
        if radius_frac >= self.hand_band_radius and abs(bearing) <= self.hand_band_max_angle:
            return Role.HAND

        if self.trick_area.contains(x, y):
            return Role.TRICK
        if radius_frac > 1.05:
            return Role.UNKNOWN
        return Role.TRICK


# Seat order around the cross, clockwise starting from the arm nearest the camera.
# The fourth player sits off-frame on the camera side, so their card lands in the
# south arm; the three visible players fill west, north and east.
_SEAT_BEARINGS = {
    "south": 180.0,
    "west": 270.0,
    "north": 0.0,
    "east": 90.0,
}


def assign_seats(points: np.ndarray) -> list[str]:
    """Map trick-card positions to cross arms by bearing from the trick centroid.

    ``points`` is an (N,2) array of canonical-space card centres belonging to one
    trick (N <= 4). Returns the arm name per point. Each arm is used at most once:
    with fewer than four cards the centroid is pulled off-centre, so a greedy
    nearest-bearing assignment would happily put two cards on the same arm.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(pts) == 0:
        return []

    centroid = pts.mean(axis=0)
    bearings = np.degrees(np.arctan2(pts[:, 0] - centroid[0], -(pts[:, 1] - centroid[1])))

    names = list(_SEAT_BEARINGS)
    # Cost = angular distance from each card to each arm, then a small exhaustive
    # search for the lowest-cost one-to-one assignment (at most 4! = 24 options).
    cost = np.zeros((len(pts), len(names)))
    for i, b in enumerate(bearings):
        for j, name in enumerate(names):
            diff = abs((b - _SEAT_BEARINGS[name] + 180.0) % 360.0 - 180.0)
            cost[i, j] = diff

    from itertools import permutations

    best_total, best_choice = float("inf"), None
    for choice in permutations(range(len(names)), len(pts)):
        total = sum(cost[i, j] for i, j in enumerate(choice))
        if total < best_total:
            best_total, best_choice = total, choice

    assert best_choice is not None
    return [names[j] for j in best_choice]
