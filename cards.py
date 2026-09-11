"""Core card model for the game *29*.

The deck is a fixed closed set of 32 cards: 8 ranks x 4 suits.

Two orderings matter and they are DIFFERENT:
  * point value   -- how many points the card is worth when captured
  * trick strength -- which card beats which within a suit

Standard 29:
    points:   J=3, 9=2, A=1, 10=1, K=0, Q=0, 8=0, 7=0   (28 total)
    strength (high->low): J, 9, A, 10, K, Q, 8, 7
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Suit(Enum):
    SPADES = "S"
    HEARTS = "H"
    DIAMONDS = "D"
    CLUBS = "C"

    @property
    def symbol(self) -> str:
        return {"S": "♠", "H": "♥", "D": "♦", "C": "♣"}[self.value]


class Rank(Enum):
    SEVEN = "7"
    EIGHT = "8"
    NINE = "9"
    TEN = "10"
    JACK = "J"
    QUEEN = "Q"
    KING = "K"
    ACE = "A"


# Point value captured when a card is won in a trick. Sums to 28 across the deck.
RANK_POINTS: dict[Rank, int] = {
    Rank.JACK: 3,
    Rank.NINE: 2,
    Rank.ACE: 1,
    Rank.TEN: 1,
    Rank.KING: 0,
    Rank.QUEEN: 0,
    Rank.EIGHT: 0,
    Rank.SEVEN: 0,
}

# Trick-taking strength, high -> low. Larger int beats smaller (within a suit).
_RANK_STRENGTH_ORDER: list[Rank] = [
    Rank.JACK,
    Rank.NINE,
    Rank.ACE,
    Rank.TEN,
    Rank.KING,
    Rank.QUEEN,
    Rank.EIGHT,
    Rank.SEVEN,
]
# Assign so that JACK is highest.
RANK_STRENGTH: dict[Rank, int] = {
    rank: len(_RANK_STRENGTH_ORDER) - 1 - i
    for i, rank in enumerate(_RANK_STRENGTH_ORDER)
}


@dataclass(frozen=True)
class Card:
    """An immutable, hashable card. Suitable as a dict key / set member."""

    rank: Rank
    suit: Suit

    @property
    def points(self) -> int:
        return RANK_POINTS[self.rank]

    @property
    def strength(self) -> int:
        """Rank strength for trick comparison (suit-agnostic; compare within a suit)."""
        return RANK_STRENGTH[self.rank]

    @property
    def code(self) -> str:
        """Compact code like ``'JS'`` (Jack of Spades) or ``'10H'``."""
        return f"{self.rank.value}{self.suit.value}"

    def __str__(self) -> str:  # pragma: no cover - display only
        return f"{self.rank.value}{self.suit.symbol}"

    @classmethod
    def from_code(cls, code: str) -> "Card":
        """Parse ``'JS'`` / ``'10H'`` / ``'7c'`` back into a Card (case-insensitive)."""
        code = code.strip().upper()
        suit_char = code[-1]
        rank_str = code[:-1]
        try:
            suit = Suit(suit_char)
        except ValueError as exc:
            raise ValueError(f"Unknown suit in card code {code!r}") from exc
        try:
            rank = Rank(rank_str)
        except ValueError as exc:
            raise ValueError(f"Unknown rank in card code {code!r}") from exc
        return cls(rank, suit)


def full_deck() -> list[Card]:
    """Return all 32 distinct cards of the fixed *29* deck."""
    return [Card(rank, suit) for suit in Suit for rank in Rank]


# Convenience: total points in the deck (invariant == 28).
DECK_TOTAL_POINTS = sum(c.points for c in full_deck())
