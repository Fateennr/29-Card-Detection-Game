"""Temporal layer: frames -> tricks -> games -> {player: card} per trick.

Single-frame detection is not the deliverable. What matters is, per game and per
trick, which card each player played. Getting there needs three things that only
exist across time:

1. **Trick segmentation.** Cards accumulate in the play area, then get swept up
   when the trick is taken. That sweep is the boundary. A trick is the span
   between two sweeps.

2. **Voting within a trick.** A card sits on the table for many frames, so it gets
   classified many times. The consensus over those frames is far stronger than any
   single frame -- which is the point, given a played card is only ~55x85 px.

3. **The uniqueness constraint.** Each of the 32 cards is played exactly once per
   game. That turns identification into a global assignment problem rather than 32
   independent guesses, and it repairs errors that no amount of per-frame
   confidence would: if the model likes 9H for two different slots, at most one can
   keep it, and the other is forced to its best remaining option.

Seating is only read off frames where the trick is at its fullest. With one or two
cards down, the centroid sits on top of the cards themselves and the bearing that
:func:`assign_seats` relies on is meaningless.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from .regions import assign_seats

# Seats, in the fixed cross arrangement, mapped to the engine's player numbers.
# Player 4 sits off-frame nearest the camera and plays into the south arm.
SEAT_TO_PLAYER: dict[str, int] = {"south": 4, "west": 1, "north": 2, "east": 3}


@dataclass(frozen=True)
class Detection:
    """One detected trick card in canonical table space."""

    code: str
    conf: float
    x: float
    y: float


@dataclass
class RawTrick:
    """A segmented trick, before uniqueness is enforced across the game."""

    start_frame: int
    end_frame: int
    peak_cards: int
    # seat -> {card code -> accumulated confidence}
    votes: dict[str, dict[str, float]] = field(default_factory=dict)

    def best_by_seat(self) -> dict[str, str]:
        """Greedy per-seat winner, ignoring the uniqueness constraint."""
        out = {}
        for seat, tally in self.votes.items():
            if tally:
                out[seat] = max(tally, key=tally.get)
        return out


class TrickSegmenter:
    """Turn a stream of per-frame detections into discrete tricks.

    ``min_empty_frames`` is how many consecutive card-free observations count as a
    sweep; it debounces the frequent single-frame dropout (a hand crossing the
    table, a bad exposure) that would otherwise split one trick into several.
    """

    def __init__(
        self,
        min_empty_frames: int = 3,
        min_peak_cards: int = 2,
        max_cards: int = 4,
        min_trick_frames: int = 0,
    ) -> None:
        self.min_empty_frames = min_empty_frames
        self.min_peak_cards = min_peak_cards
        self.max_cards = max_cards
        self.min_trick_frames = min_trick_frames
        # Segments dropped for being too short/weak, for diagnostics.
        self.rejected = 0
        self._observations: list[tuple[int, list[Detection]]] = []
        self._empty_streak = 0
        self._start_frame: int | None = None
        self.tricks: list[RawTrick] = []

    def feed(self, frame_index: int, detections: list[Detection]) -> None:
        detections = sorted(detections, key=lambda d: -d.conf)[: self.max_cards]

        if detections:
            self._empty_streak = 0
            if self._start_frame is None:
                self._start_frame = frame_index
            self._observations.append((frame_index, detections))
            return

        self._empty_streak += 1
        if self._observations and self._empty_streak >= self.min_empty_frames:
            self._close(frame_index)

    def finish(self) -> list[RawTrick]:
        """Close any trick still open at end of stream and return all tricks."""
        if self._observations:
            self._close(self._observations[-1][0])
        return self.tricks

    def _close(self, end_frame: int) -> None:
        observations = self._observations
        self._observations = []
        self._empty_streak = 0
        start = self._start_frame
        self._start_frame = None

        peak = max(len(d) for _, d in observations)
        if peak < self.min_peak_cards or start is None:
            self.rejected += 1
            return  # stray detections, not a real trick

        # A real trick lasts as long as it takes four players to play. Anything
        # briefer is detector flicker: on the real footage a naive emptiness test
        # fires an order of magnitude more often than there are actual tricks.
        if observations[-1][0] - start < self.min_trick_frames:
            self.rejected += 1
            return

        # Seat only from the fullest frames -- see the module docstring.
        fullest = [(idx, dets) for idx, dets in observations if len(dets) == peak]
        votes: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        for _, dets in fullest:
            seats = assign_seats(np.array([[d.x, d.y] for d in dets]))
            for det, seat in zip(dets, seats):
                votes[seat][det.code] += det.conf

        self.tricks.append(
            RawTrick(
                start_frame=start,
                end_frame=end_frame,
                peak_cards=peak,
                votes={s: dict(t) for s, t in votes.items()},
            )
        )


def split_games(
    tricks: list[RawTrick],
    tricks_per_game: int = 8,
    deal_gap_frames: int | None = None,
) -> list[list[RawTrick]]:
    """Group tricks into games.

    A game is ``tricks_per_game`` tricks. When ``deal_gap_frames`` is given, an
    unusually long card-free stretch (the deal) also closes a game, which recovers
    the grouping when a trick is missed or spuriously split.
    """
    games: list[list[RawTrick]] = []
    current: list[RawTrick] = []

    for i, trick in enumerate(tricks):
        if current and deal_gap_frames is not None:
            gap = trick.start_frame - current[-1].end_frame
            if gap >= deal_gap_frames:
                games.append(current)
                current = []
        current.append(trick)
        if len(current) == tricks_per_game:
            games.append(current)
            current = []

    if current:
        games.append(current)
    return games


def enforce_uniqueness(
    game_tricks: list[RawTrick], all_codes: list[str]
) -> list[dict[str, str]]:
    """Resolve a whole game at once, so each card is used at most once.

    Returns one ``{seat: card_code}`` mapping per trick. Solves a maximum-weight
    assignment over every (trick, seat) slot against the 32 cards, using the voted
    confidences as weights -- rather than taking each slot's argmax independently
    and hoping no card gets claimed twice.
    """
    slots: list[tuple[int, str]] = []
    for t_idx, trick in enumerate(game_tricks):
        for seat in trick.votes:
            slots.append((t_idx, seat))

    if not slots:
        return [{} for _ in game_tricks]

    code_index = {c: i for i, c in enumerate(all_codes)}
    score = np.zeros((len(slots), len(all_codes)))
    for row, (t_idx, seat) in enumerate(slots):
        for codeval, weight in game_tricks[t_idx].votes[seat].items():
            if codeval in code_index:
                score[row, code_index[codeval]] = weight

    try:
        from scipy.optimize import linear_sum_assignment

        rows, cols = linear_sum_assignment(score, maximize=True)
        pairs = list(zip(rows, cols))
    except ImportError:  # pragma: no cover - scipy ships with Colab
        pairs = _greedy_assignment(score)

    result: list[dict[str, str]] = [{} for _ in game_tricks]
    for row, col in pairs:
        # A slot with no vote mass at all would otherwise be handed an arbitrary
        # leftover card purely to complete the matching.
        if score[row, col] <= 0:
            continue
        t_idx, seat = slots[row]
        result[t_idx][seat] = all_codes[col]
    return result


def _greedy_assignment(score: np.ndarray) -> list[tuple[int, int]]:
    """Fallback for the assignment problem: repeatedly take the best free pair."""
    pairs = []
    used_rows: set[int] = set()
    used_cols: set[int] = set()
    order = np.dstack(np.unravel_index(np.argsort(score, axis=None)[::-1], score.shape))[0]
    for row, col in order:
        if row in used_rows or col in used_cols:
            continue
        used_rows.add(int(row))
        used_cols.add(int(col))
        pairs.append((int(row), int(col)))
    return pairs


def to_player_mapping(seat_mapping: dict[str, str]) -> dict[int, str]:
    """Convert ``{seat: code}`` to ``{player_number: code}``."""
    return {SEAT_TO_PLAYER[seat]: codeval for seat, codeval in seat_mapping.items()}
