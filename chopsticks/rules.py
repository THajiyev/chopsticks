"""Rule variants for Chopsticks, bundled in a frozen dataclass, plus named presets."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Literal

Overflow = Literal["cutoff", "rollover"]


@dataclass(frozen=True)
class Rules:
    """A complete rule set.

    Attributes:
        overflow: ``"rollover"`` (default: the new value is the sum mod 5, so
            4 + 4 -> 3 and only an exact 5 kills the hand) or ``"cutoff"`` (a
            hand reaching 5 or more dies).
        splits_allowed: whether the split move exists at all.
        even_splits_only: a split must leave both hands equal.
        allow_revive: a dead hand can be brought back by a split. If False, a
            split is only legal when both hands are alive before it.
        allow_suicide_split: a split may leave one hand at 0.
        max_moves: number of plies after which the game is declared a draw.
    """

    overflow: Overflow = "rollover"
    splits_allowed: bool = True
    even_splits_only: bool = False
    allow_revive: bool = True
    allow_suicide_split: bool = True
    max_moves: int = 200

    def __post_init__(self) -> None:
        if self.overflow not in ("cutoff", "rollover"):
            raise ValueError(f"unknown overflow rule: {self.overflow!r}")
        if self.max_moves <= 0:
            raise ValueError("max_moves must be positive")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict of the rule fields."""
        return asdict(self)

    def with_(self, **changes: Any) -> "Rules":
        """Return a copy with some fields changed."""
        return replace(self, **changes)


PRESETS: dict[str, Rules] = {
    "standard": Rules(),
    "no_suicide": Rules(allow_suicide_split=False),
    "cutoff": Rules(overflow="cutoff"),
    "no_splits": Rules(splits_allowed=False),
    "no_revive": Rules(allow_revive=False),
}

PRESET_DESCRIPTIONS: dict[str, str] = {
    "standard": "Sums wrap mod 5 (4 + 4 = 3; exactly 5 is dead). Revives and suicide splits allowed.",
    "no_suicide": "Like standard, but a split may not leave a hand at 0.",
    "cutoff": "Like standard, but any hand reaching 5 or more dies (4 + 4 = dead).",
    "no_splits": "Attacks only; no splits at all.",
    "no_revive": "Dead hands stay dead; splits only when both hands are alive.",
}


def get_preset(name: str) -> Rules:
    """Look up a preset by name, raising ``KeyError`` with the valid names."""
    try:
        return PRESETS[name]
    except KeyError:
        raise KeyError(f"unknown preset {name!r}; choose from {sorted(PRESETS)}") from None
