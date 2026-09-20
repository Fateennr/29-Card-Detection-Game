"""Hand-level scoring for *29*: tally team points and resolve the bid.

Consumes the deterministic outputs of :mod:`rules` (trick winners, captured
points, marriage) and produces the final result of a hand: each team's score and
whether the bidding team made its contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .cards import Card, Suit
from .rules import (
    TEAM_A,
    TEAM_B,
    Trick,
    detect_marriage,
    order_from_leader,
    team_of,
    trick_winner,
)

# Marriage bonus points. Configurable; 4 is the common value in *29*.
DEFAULT_MARRIAGE_BONUS = 4


def _team_name(team: frozenset[int]) -> str:
    return "A" if team == TEAM_A else "B"


@dataclass
class TrumpInfo:
    """Describes the trump for a hand.

    * ``suit``               -- the trump suit, or None if never revealed on camera.
    * ``reveal_trick_index`` -- index into the hand's trick list at which trump
      becomes active (inclusive). None if trump was never revealed.
    """

    suit: Suit | None = None
    reveal_trick_index: int | None = None

    @property
    def is_active(self) -> bool:
        return self.suit is not None and self.reveal_trick_index is not None


@dataclass
class HandResult:
    team_points: dict[str, int]
    trick_winners: list[int]
    marriage_player: int | None
    marriage_team: str | None
    # Leader of each trick, in order. Derived from the previous trick's winner,
    # never observed from the video.
    trick_leaders: list[int] = field(default_factory=list)
    bidding_team: str | None = None
    bid: int | None = None
    contract_made: bool | None = None
    winning_team: str | None = None
    detail: list[dict] = field(default_factory=list)


def score_hand(
    tricks: list[Trick],
    trump: TrumpInfo,
    *,
    bidding_team: str | None = None,
    bid: int | None = None,
    marriage_bonus: int = DEFAULT_MARRIAGE_BONUS,
) -> HandResult:
    """Score one hand (one deal of 8 tricks) and resolve the bid if given.

    ``bidding_team`` is ``"A"`` or ``"B"`` (from user text input); ``bid`` is the
    contracted point target. If both are supplied, the bidding team must capture
    >= ``bid`` points (including any marriage bonus it earned) to make contract;
    otherwise the opposing team wins the hand.
    """
    points = {"A": 0, "B": 0}
    winners: list[int] = []
    detail: list[dict] = []

    for idx, trick in enumerate(tricks):
        # Trump is only active from the reveal trick onward.
        active_suit = (
            trump.suit
            if trump.is_active and idx >= trump.reveal_trick_index  # type: ignore[operator]
            else None
        )
        winner = trick_winner(trick, active_suit)
        winners.append(winner)
        team = _team_name(team_of(winner))
        points[team] += trick.points
        detail.append(
            {
                "trick": idx,
                "winner": winner,
                "team": team,
                "points": trick.points,
                "trump_active": active_suit.value if active_suit else None,
            }
        )

    # Marriage bonus.
    marriage_player = detect_marriage(tricks, trump.suit, trump.reveal_trick_index)
    marriage_team: str | None = None
    if marriage_player is not None:
        marriage_team = _team_name(team_of(marriage_player))
        points[marriage_team] += marriage_bonus

    result = HandResult(
        team_points=points,
        trick_winners=winners,
        marriage_player=marriage_player,
        marriage_team=marriage_team,
        bidding_team=bidding_team,
        bid=bid,
        detail=detail,
    )

    result.trick_leaders = [t.leader for t in tricks]

    # Resolve the contract.
    if bidding_team is not None and bid is not None:
        if bidding_team not in ("A", "B"):
            raise ValueError(f"bidding_team must be 'A' or 'B', got {bidding_team!r}")
        made = points[bidding_team] >= bid
        result.contract_made = made
        other = "B" if bidding_team == "A" else "A"
        result.winning_team = bidding_team if made else other

    return result


def build_tricks_from_plays(
    plays: list[dict[int, Card]],
    trump: TrumpInfo,
    first_leader: int,
) -> list[Trick]:
    """Turn per-trick ``{player: card}`` mappings into ordered :class:`Trick` objects.

    Vision recovers *who played what*, not the sequence. The sequence is derived:
    the winner of a trick leads the next, going round the table from there. So the
    whole chain follows from ``first_leader``.

    This is not bookkeeping -- the leader fixes the *led suit*, which decides the
    winner whenever no trump is played. Get a leader wrong and the winner can
    change, which changes the next leader, so an early error propagates.
    """
    tricks: list[Trick] = []
    leader = first_leader

    for index, mapping in enumerate(plays):
        trick = Trick.from_mapping(mapping, order=order_from_leader(leader))
        tricks.append(trick)

        active_suit = (
            trump.suit
            if trump.is_active and index >= trump.reveal_trick_index  # type: ignore[operator]
            else None
        )
        leader = trick_winner(trick, active_suit)

    return tricks


def score_hand_from_plays(
    plays: list[dict[int, Card]],
    trump: TrumpInfo,
    *,
    first_leader: int = 1,
    bidding_team: str | None = None,
    bid: int | None = None,
    marriage_bonus: int = DEFAULT_MARRIAGE_BONUS,
) -> HandResult:
    """Score a hand straight from ``{player: card}`` mappings, deriving play order.

    ``first_leader`` is the only ordering input the video cannot supply; in *29* it
    is the bid winner. Every later leader is computed from the trick outcomes.
    """
    tricks = build_tricks_from_plays(plays, trump, first_leader)
    return score_hand(
        tricks,
        trump,
        bidding_team=bidding_team,
        bid=bid,
        marriage_bonus=marriage_bonus,
    )
