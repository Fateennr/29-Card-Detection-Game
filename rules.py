"""Trick-taking rules for *29*: play model, trick-winner, and marriage detection.

These functions are the deterministic heart of the system. They take *already
identified* cards (rank+suit + which player played them) and produce trick
winners, captured points, and marriage events. They contain no CV logic, so they
can be unit-tested exhaustively against hand-computed games.

Player / team model
-------------------
Players are labelled 1..4 by seat. Players sitting opposite are partners:
    Team A = {1, 3}
    Team B = {2, 4}
"""

from __future__ import annotations

from dataclasses import dataclass

from .cards import Card, Rank, Suit

PLAYERS = (1, 2, 3, 4)
TEAM_A = frozenset({1, 3})
TEAM_B = frozenset({2, 4})

# Seating order around the table. Partners sit opposite, so going round the table
# alternates teams: 1 (A), 2 (B), 3 (A), 4 (B).
SEATING_ORDER = (1, 2, 3, 4)


def order_from_leader(leader: int) -> list[int]:
    """Return the play order for a trick led by ``leader``, going round the table.

    Play order is not observed from the video -- it is derived. The winner of a
    trick leads the next one, so once the first leader is known every subsequent
    order follows from the trick outcomes.
    """
    if leader not in SEATING_ORDER:
        raise ValueError(f"leader must be one of {SEATING_ORDER}, got {leader!r}")
    start = SEATING_ORDER.index(leader)
    return [SEATING_ORDER[(start + step) % len(SEATING_ORDER)] for step in range(4)]


def team_of(player: int) -> frozenset[int]:
    """Return the partner-set (team) that ``player`` belongs to."""
    if player in TEAM_A:
        return TEAM_A
    if player in TEAM_B:
        return TEAM_B
    raise ValueError(f"player must be one of {PLAYERS}, got {player!r}")


@dataclass(frozen=True)
class Play:
    """A single card played by a player. ``order`` is 0..3 within the trick."""

    player: int
    card: Card
    order: int


@dataclass(frozen=True)
class Trick:
    """The four plays of one trick, in play order (index 0 == leader)."""

    plays: tuple[Play, ...]

    def __post_init__(self) -> None:
        if len(self.plays) != 4:
            raise ValueError(f"a trick has exactly 4 plays, got {len(self.plays)}")
        players = {p.player for p in self.plays}
        if players != set(PLAYERS):
            raise ValueError(f"a trick must have all of players {PLAYERS}, got {players}")
        cards = [p.card for p in self.plays]
        if len(set(cards)) != 4:
            raise ValueError(f"duplicate card within a trick: {cards}")

    @property
    def leader(self) -> int:
        return self.ordered()[0].player

    @property
    def led_suit(self) -> Suit:
        return self.ordered()[0].card.suit

    def ordered(self) -> list[Play]:
        """Plays sorted by play order (leader first)."""
        return sorted(self.plays, key=lambda p: p.order)

    @property
    def points(self) -> int:
        """Total captured points available in this trick."""
        return sum(p.card.points for p in self.plays)

    @classmethod
    def from_mapping(
        cls, player_to_card: dict[int, Card], order: list[int] | None = None
    ) -> "Trick":
        """Build a Trick from ``{player: card}`` plus an optional play order.

        ``order`` is the list of players in the sequence they played
        (e.g. ``[2, 3, 4, 1]``). Defaults to ``[1, 2, 3, 4]``.
        """
        order = order or list(PLAYERS)
        if sorted(order) != list(PLAYERS):
            raise ValueError(f"order must be a permutation of {PLAYERS}, got {order}")
        plays = tuple(
            Play(player=pl, card=player_to_card[pl], order=i)
            for i, pl in enumerate(order)
        )
        return cls(plays)


def trick_winner(trick: Trick, trump_suit: Suit | None) -> int:
    """Return the player who wins ``trick``.

    Rules:
      * If any trump was played, the highest-strength trump wins.
      * Otherwise the highest-strength card of the *led* suit wins.
      * ``trump_suit=None`` means trump is not (yet) active -> led-suit only.

    Note on reveal timing: trump only becomes active once revealed. Callers that
    model a mid-game reveal should pass ``trump_suit=None`` for tricks played
    before trump is active and the actual suit from the reveal trick onward.
    """
    ordered = trick.ordered()
    if trump_suit is not None:
        trumps = [p for p in ordered if p.card.suit == trump_suit]
        if trumps:
            return max(trumps, key=lambda p: p.card.strength).player

    led = trick.led_suit
    followers = [p for p in ordered if p.card.suit == led]
    # followers is always non-empty (the leader itself follows the led suit).
    return max(followers, key=lambda p: p.card.strength).player


def detect_marriage(
    tricks: list[Trick],
    trump_suit: Suit | None,
    reveal_trick_index: int | None,
) -> int | None:
    """Return the player who scored a marriage, or ``None``.

    Marriage (per project spec): the King AND Queen of the *trump* suit are both
    played by the *same* player, *after* the trump reveal.

    ``reveal_trick_index`` is the index into ``tricks`` at which trump becomes
    active (inclusive). Plays in earlier tricks do not count toward marriage.
    ``None`` for either trump argument means no marriage is possible.
    """
    if trump_suit is None or reveal_trick_index is None:
        return None

    king = Card(Rank.KING, trump_suit)
    queen = Card(Rank.QUEEN, trump_suit)
    king_by: int | None = None
    queen_by: int | None = None

    for idx, trick in enumerate(tricks):
        if idx < reveal_trick_index:
            continue
        for play in trick.plays:
            if play.card == king:
                king_by = play.player
            elif play.card == queen:
                queen_by = play.player

    if king_by is not None and king_by == queen_by:
        return king_by
    return None
