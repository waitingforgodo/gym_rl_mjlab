"""Automatically play the last N runs of a specified task."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from typing import Annotated

import tyro


@dataclass(frozen=True)
class MultiPlayConfig:
    """Automatically play the last N runs of a specified task."""

    expdir: str = "g1_shadowing"
    task: str = "Instinct-Shadowing-Plane-PartBody-MultiReward-G1-Play-v0"
    n_runs: Annotated[int, tyro.conf.arg(aliases=("-n",))] = 0
    """Number of latest runs to play."""
    log_runs: Annotated[list[str], tyro.conf.arg(aliases=("-l",))] = field(default_factory=list)


def main(cfg: MultiPlayConfig, extra_args: list[str]) -> None:
    """Automatically play the last n runs of the specified task."""
    # specify directory for logging experiments
    expdir = os.path.join("logs", "instinct_rl", cfg.expdir)
    expdir = os.path.abspath(expdir)

    # get the last n runs
    if cfg.n_runs:
        all_runs = os.listdir(expdir)
        all_runs.sort(key=lambda x: datetime.strptime("_".join(x.split("_")[:2]), "%Y-%m-%d_%H-%M-%S"))
        runs_to_play = all_runs[-cfg.n_runs :]
        runs_to_play = ["_".join(x.split("_")[:2]) + ".*" for x in runs_to_play]
        print(f"No run specified, getting the latest {cfg.n_runs} runs")
    else:
        runs_to_play = cfg.log_runs

    for run_name in runs_to_play:
        print(f"Playing run: {run_name}")
        # play the run using the instinct-play CLI entry point
        subprocess.run(
            [
                "instinct-play",
                cfg.task,
                "--load-run",
                run_name,
            ]
            + extra_args
        )


def entry_point() -> None:
    cfg, extra_args = tyro.cli(MultiPlayConfig, return_unknown_args=True)
    main(cfg, extra_args)


if __name__ == "__main__":
    entry_point()
