# Copyright (c) 2022-2024, The Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Instinct-RL configuration for the G1 whole-body shadowing task."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from instinct_mj.rl import (
    InstinctRlActorCriticCfg,
    InstinctRlEncoderActorCriticCfg,
    InstinctRlMlpCfg,
    InstinctRlMoEActorCriticCfg,
    InstinctRlOnPolicyRunnerCfg,
    InstinctRlPpoAlgorithmCfg,
    InstinctRlTransformerHeadCfg,
)
from instinct_mj.tasks.config.rl_utils import default_policy_critic_normalizers


@dataclass(kw_only=True)
class MotionRefEncoderMlpCfg(InstinctRlMlpCfg):
    hidden_sizes: list = field(
        default_factory=lambda: [
            256,
            128,
        ]
    )
    nonlinearity: str = "CELU"
    takeout_input_components: bool = True
    output_size: int = 64

    component_names: list = field(
        default_factory=lambda: [
            # "time_to_target_ref",
            # "time_from_ref_update",
            # "pose_ref",
            # "position_ref",
            # "rotation_ref",
            # "position_ref_mask",
            # "rotation_ref_mask",
            # "joint_pos_ref",
            # "joint_vel_ref",
            # "joint_pos_err_ref",
            # "joint_pos_ref_mask",
            # "link_pos_ref",
            # "link_pos_err_ref",
            # "link_pos_ref_mask",
            # "link_rot_ref",
            # "link_rot_err_ref",
            # "link_rot_ref_mask",
        ]
    )


@dataclass(kw_only=True)
class MotionRefEncoderTransformerCfg(InstinctRlTransformerHeadCfg):
    activation: str = "gelu"
    nonlinearity: str = "CELU"
    output_size: int = 64
    num_heads: int = 2
    num_layers: int = 1
    d_model: int = 64
    dim_feedforward: int = 64
    output_selection: str = "maxpool"
    component_names: list = field(
        default_factory=lambda: [
            # "time_to_target_ref",
            # "time_from_ref_update",
            # "pose_ref",
            # "position_ref",
            # "rotation_ref",
            # "position_ref_mask",
            # "rotation_ref_mask",
            # "joint_pos_ref",
            # "joint_vel_ref",
            # "joint_pos_err_ref",
            # "joint_pos_ref_mask",
            # "link_pos_ref",
            # "link_pos_err_ref",
            # "link_pos_ref_mask",
            # "link_rot_ref",
            # "link_rot_err_ref",
            # "link_rot_ref_mask",
        ]
    )


@dataclass(kw_only=True)
class EncoderConfigs:
    motion_ref: object = field(default_factory=MotionRefEncoderMlpCfg)
    # motion_ref: object = field(default_factory=MotionRefEncoderTransformerCfg)


@dataclass(kw_only=True)
class PolicyCfgMixin:
    init_noise_std: float = 1.0
    actor_hidden_dims: list = field(default_factory=lambda: [512, 256, 128])
    critic_hidden_dims: list = field(default_factory=lambda: [512, 256, 128])
    activation: str = "elu"


@dataclass(kw_only=True)
class EncoderPolicyCfg(PolicyCfgMixin, InstinctRlEncoderActorCriticCfg):
    encoder_configs: object = field(default_factory=EncoderConfigs)
    critic_encoder_configs: object = field(default_factory=EncoderConfigs)


@dataclass(kw_only=True)
class MlpPolicyCfg(PolicyCfgMixin, InstinctRlActorCriticCfg):
    pass


@dataclass(kw_only=True)
class MoEPolicyCfg(PolicyCfgMixin, InstinctRlMoEActorCriticCfg):
    # Approximately 2.5M params
    actor_hidden_dims: list = field(default_factory=lambda: [256, 256, 128])
    critic_hidden_dims: list = field(default_factory=lambda: [256, 256, 128])
    num_moe_experts: int = 8
    moe_gate_hidden_dims: list = field(
        default_factory=lambda: [128, 64]
    )  # Naive MLP for gating: input -> 128 -> 64 -> num_experts


@dataclass(kw_only=True)
class EquivalentMlpPolicyCfg(PolicyCfgMixin, InstinctRlActorCriticCfg):
    # Approximately 2.5M params
    actor_hidden_dims: list = field(default_factory=lambda: [1024, 1024, 640])
    critic_hidden_dims: list = field(default_factory=lambda: [1024, 1024, 640])


def _shadowing_policy_cfg() -> InstinctRlActorCriticCfg:
    """Select the active policy. Swap the returned config to switch."""
    # return MlpPolicyCfg()
    return MoEPolicyCfg()
    # return EquivalentMlpPolicyCfg()


def _policy_run_name(policy: InstinctRlActorCriticCfg) -> str:
    return "".join(
        [
            "_MLPPolicy" if isinstance(policy, MlpPolicyCfg) else "",
            (
                "_MlpEncoder"
                if isinstance(policy, EncoderPolicyCfg)
                and isinstance(policy.encoder_configs.motion_ref, MotionRefEncoderMlpCfg)
                else ""
            ),
            "_MoEPolicy" if isinstance(policy, MoEPolicyCfg) else "",
            "_EquivalentMlpPolicy" if isinstance(policy, EquivalentMlpPolicyCfg) else "",
        ]
    )


def _shadowing_algorithm_cfg() -> InstinctRlPpoAlgorithmCfg:
    return InstinctRlPpoAlgorithmCfg(
        class_name="PPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


def g1_shadowing_ppo_runner_cfg() -> InstinctRlOnPolicyRunnerCfg:
    policy = _shadowing_policy_cfg()
    # ckpt_manipulator = "reinitialize_actor_critic_backbone"
    # ckpt_manipulator = "newStd"
    # ckpt_manipulator = "fit_smaller_mlp_input"
    # ckpt_manipulator_kwargs = {
    #     "weight_removal_slices": {
    #         "actor.0.weight": (585, 585 + 30), # remove 30 dimensions from the input
    #         "critic.0.weight": (15 + 585, 15 + 585 + 30), # remove 30 dimensions from the input
    #     },
    # }
    run_name = "".join(
        [
            _policy_run_name(policy),
            # "_newStd" if ckpt_manipulator == "newStd" else "",
            # ("_obsNorm" if not isinstance(normalizers, dict) else ""),
            # f"_entropyCoeff{algorithm.entropy_coef:.0e}" if algorithm.entropy_coef else "",
            f"_GPU{os.environ.get('CUDA_VISIBLE_DEVICES')}" if "CUDA_VISIBLE_DEVICES" in os.environ else "",
        ]
    )
    return InstinctRlOnPolicyRunnerCfg(
        policy=policy,
        algorithm=_shadowing_algorithm_cfg(),
        normalizers=default_policy_critic_normalizers(),
        num_steps_per_env=24,
        max_iterations=50000,
        save_interval=1000,
        log_interval=10,
        experiment_name="g1_shadowing",
        run_name=run_name,
        resume=False,
        load_run=".*",
        policy_observation_group="policy",
        critic_observation_group="critic",
    )


def g1_multi_reward_shadowing_ppo_runner_cfg() -> InstinctRlOnPolicyRunnerCfg:
    cfg = g1_shadowing_ppo_runner_cfg()
    cfg.algorithm.advantage_mixing_weights = (0.7, 0.3)
    cfg.run_name += "_Adv622"
    return cfg


def G1ShadowingPPORunnerCfg() -> InstinctRlOnPolicyRunnerCfg:
    """Return the whole-body shadowing PPO runner config."""

    return g1_shadowing_ppo_runner_cfg()


def G1MultiRewardShadowingPPORunnerCfg() -> InstinctRlOnPolicyRunnerCfg:
    """Return the multi-reward PPO runner config."""

    return g1_multi_reward_shadowing_ppo_runner_cfg()


def g1_shadowing_instinct_rl_cfg() -> InstinctRlOnPolicyRunnerCfg:
    return g1_shadowing_ppo_runner_cfg()
