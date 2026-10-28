"""Configuration terms for different managers."""

from __future__ import annotations

from mjlab.managers import RewardTermCfg


class MultiRewardCfg(dict[str, dict[str, RewardTermCfg | str | None]]):
    """Marker dictionary for grouped reward configurations.

    Instantiate this instead of a plain dictionary when reward groups should be
    consumed by :class:`MultiRewardManager`.
    """
