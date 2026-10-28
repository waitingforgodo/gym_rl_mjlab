from __future__ import annotations

from dataclasses import dataclass, field

from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnvCfg
from mjlab.viewer.viewer_config import ViewerConfig

from instinct_mj.envs.scene import InstinctScene
from instinct_mj.managers import MultiRewardManager
from instinct_mj.monitors import MonitorManager


@dataclass(kw_only=True)
class InstinctRlEnvCfg(ManagerBasedRlEnvCfg):
    """Configuration for a reinforcement learning environment with the manager-based workflow."""

    rewards: dict = field(default_factory=dict)
    """Reward terms or marked multi-reward groups."""

    viewer: ViewerConfig = field(default_factory=ViewerConfig)
    """Viewer Settings."""

    run_name: str = ""
    """Environment-specific suffix used for experiment log directories."""

    # monitor settings
    monitors: dict = field(default_factory=dict)
    """Monitor Settings.

  Please refer to the `instinct_mj.monitors.MonitorManager` class for more details.
  """

    scene_class_type: type = InstinctScene
    """The scene class the environment constructs. Override to swap in a custom scene."""

    multi_reward_manager_class_type: type = MultiRewardManager
    """The manager class used when ``rewards`` is a :class:`MultiRewardCfg` group.

  This only covers the grouped-reward path. Plain ``dict[str, RewardTermCfg]``
  rewards are built by mjlab's own ``load_managers()``, which hardcodes its
  ``RewardManager``, so that path is not overridable from here.
  """

    monitor_manager_class_type: type = MonitorManager
    """The manager class the environment constructs for ``monitors``."""
