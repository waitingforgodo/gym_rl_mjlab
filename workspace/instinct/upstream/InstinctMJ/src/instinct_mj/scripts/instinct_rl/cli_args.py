"""Instinct-RL CLI argument helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from instinct_mj.rl import InstinctRlOnPolicyRunnerCfg


@dataclass(frozen=True)
class InstinctRlCliConfig:
    """Arguments for an Instinct-RL agent, suitable for direct Tyro parsing."""

    # -- experiment arguments
    experiment_name: str | None = None
    """Name of the experiment folder where logs will be stored."""
    run_name: str | None = None
    """Run name suffix to the log directory."""
    # -- load arguments
    seed: int | None = None
    resume: bool | None = None
    """Whether to resume from a checkpoint."""
    load_run: str | None = None
    """Name of the run folder to resume from."""
    checkpoint: str | None = None
    """Checkpoint file to resume from."""
    # # -- logger arguments
    # logger: Literal["wandb", "tensorboard", "neptune"] | None = None
    # log_project_name: str | None = None


def parse_instinct_rl_cfg(task_name: str, args_cli: InstinctRlCliConfig) -> InstinctRlOnPolicyRunnerCfg:
    """Parse configuration for Instinct-RL agent based on inputs.

    Args:
        task_name: The name of the environment.
        args_cli: The command line arguments.

    Returns:
        The parsed configuration for Instinct-RL agent based on inputs.
    """
    from instinct_mj.tasks.registry import load_instinct_rl_cfg

    # load the default configuration from Instinct Mj registry
    instinctrl_cfg = load_instinct_rl_cfg(task_name)
    instinctrl_cfg = update_instinct_rl_cfg(instinctrl_cfg, args_cli)
    return instinctrl_cfg


def update_instinct_rl_cfg(agent_cfg: InstinctRlOnPolicyRunnerCfg, args_cli: InstinctRlCliConfig):
    """Update configuration for Instinct-RL agent based on inputs.

    Args:
        agent_cfg: The configuration for Instinct-RL agent.
        args_cli: The command line arguments.

    Returns:
        The updated configuration for Instinct-RL agent based on inputs.
    """
    # override the default configuration with CLI arguments
    if hasattr(args_cli, "seed") and args_cli.seed is not None:
        agent_cfg.seed = args_cli.seed
    if args_cli.resume is not None:
        agent_cfg.resume = args_cli.resume
    if args_cli.load_run is not None:
        agent_cfg.load_run = args_cli.load_run
    if args_cli.checkpoint is not None:
        agent_cfg.load_checkpoint = args_cli.checkpoint
    if args_cli.run_name is not None:
        agent_cfg.run_name = args_cli.run_name
    # if args_cli.logger is not None:
    #     agent_cfg.logger = args_cli.logger
    # # set the project name for wandb and neptune
    # if agent_cfg.logger in {"wandb", "neptune"} and args_cli.log_project_name:
    #     agent_cfg.wandb_project = args_cli.log_project_name
    #     agent_cfg.neptune_project = args_cli.log_project_name

    return agent_cfg
