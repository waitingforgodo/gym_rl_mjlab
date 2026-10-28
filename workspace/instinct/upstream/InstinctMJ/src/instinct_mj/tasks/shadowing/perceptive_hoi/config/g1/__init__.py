"""Register Instinct Mj perceptive HOI G1 tasks."""

from instinct_mj.tasks.registry import register_instinct_task


def g1_perceptive_hoi_shadowing_instinct_rl_cfg():
    from .rl_cfg import g1_perceptive_hoi_shadowing_instinct_rl_cfg as _cfg

    return _cfg()


def _perceptive_hoi_shadowing_env_cfg():
    from .perceptive_shadowing_cfg import G1PerceptiveHoiShadowingEnvCfg

    return G1PerceptiveHoiShadowingEnvCfg()


def _perceptive_hoi_shadowing_play_env_cfg():
    from .perceptive_shadowing_cfg import G1PerceptiveHoiShadowingEnvCfg_PLAY

    return G1PerceptiveHoiShadowingEnvCfg_PLAY()


register_instinct_task(
    task_id="Instinct-Perceptive-HOI-Shadowing-G1-v0",
    env_cfg_factory=_perceptive_hoi_shadowing_env_cfg,
    play_env_cfg_factory=_perceptive_hoi_shadowing_play_env_cfg,
    instinct_rl_cfg_factory=g1_perceptive_hoi_shadowing_instinct_rl_cfg,
)

register_instinct_task(
    task_id="Instinct-Perceptive-HOI-Shadowing-G1-Play-v0",
    env_cfg_factory=_perceptive_hoi_shadowing_play_env_cfg,
    play_env_cfg_factory=_perceptive_hoi_shadowing_play_env_cfg,
    instinct_rl_cfg_factory=g1_perceptive_hoi_shadowing_instinct_rl_cfg,
)
