"""Task registry for Instinct-RL based mjlab tasks."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Callable

from mjlab.envs import ManagerBasedRlEnvCfg

from instinct_mj.rl import InstinctRlOnPolicyRunnerCfg

DEFAULT_ENV_ENTRY_POINT = "instinct_mj.envs:InstinctRlEnv"
DEFAULT_VECENV_ENTRY_POINT = "instinct_mj.rl:InstinctRlVecEnvWrapper"
"""Entry points used when a task does not name its own classes.

Entry points are ``"module.path:ClassName"`` strings resolved at launch time, so
a downstream repository can point a task at its own environment, vec-env wrapper
or runner class without the registry importing it (or this package's default) up
front.
"""


@dataclass
class _TaskCfg:
    env_cfg_factory: Callable[[], ManagerBasedRlEnvCfg]
    play_env_cfg_factory: Callable[[], ManagerBasedRlEnvCfg]
    instinct_rl_cfg_factory: Callable[[], InstinctRlOnPolicyRunnerCfg]
    env_entry_point: str
    vecenv_entry_point: str
    runner_entry_point: str | None


_REGISTRY: dict[str, _TaskCfg] = {}


def _resolve_entry_point(entry_point: str) -> type:
    """Import ``"module.path:ClassName"`` and return the class object."""
    module_path, _, class_name = entry_point.partition(":")
    return getattr(importlib.import_module(module_path), class_name)


def register_instinct_task(
    task_id: str,
    env_cfg_factory: Callable[[], ManagerBasedRlEnvCfg],
    play_env_cfg_factory: Callable[[], ManagerBasedRlEnvCfg],
    instinct_rl_cfg_factory: Callable[[], InstinctRlOnPolicyRunnerCfg],
    env_entry_point: str = DEFAULT_ENV_ENTRY_POINT,
    vecenv_entry_point: str = DEFAULT_VECENV_ENTRY_POINT,
    runner_entry_point: str | None = None,
) -> None:
    if task_id in _REGISTRY:
        raise ValueError(f"Task '{task_id}' is already registered.")
    _REGISTRY[task_id] = _TaskCfg(
        env_cfg_factory,
        play_env_cfg_factory,
        instinct_rl_cfg_factory,
        env_entry_point,
        vecenv_entry_point,
        runner_entry_point,
    )


def list_tasks() -> list[str]:
    return sorted(_REGISTRY.keys())


def load_env_cfg(task_name: str, play: bool = False) -> ManagerBasedRlEnvCfg:
    task_cfg = _REGISTRY[task_name]
    return task_cfg.play_env_cfg_factory() if play else task_cfg.env_cfg_factory()


def load_instinct_rl_cfg(task_name: str) -> InstinctRlOnPolicyRunnerCfg:
    return _REGISTRY[task_name].instinct_rl_cfg_factory()


def load_env_cls(task_name: str) -> type:
    return _resolve_entry_point(_REGISTRY[task_name].env_entry_point)


def load_vecenv_cls(task_name: str) -> type:
    return _resolve_entry_point(_REGISTRY[task_name].vecenv_entry_point)


def load_runner_cls(task_name: str) -> type | None:
    entry_point = _REGISTRY[task_name].runner_entry_point
    if entry_point is None:
        return None
    return _resolve_entry_point(entry_point)
