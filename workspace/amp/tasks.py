"""独立的 G1 AMP 舞蹈任务；不修改已安装的 mjlab 或原 tracking 配置。"""
from dataclasses import dataclass, field

from mjlab.envs import mdp
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import RslRlPpoAlgorithmCfg
from mjlab.tasks.registry import list_tasks, register_mjlab_task
from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg
from mjlab.tasks.tracking.config.g1.rl_cfg import unitree_g1_tracking_ppo_runner_cfg

from .environment import amp_features, capture_terminal_features, reset_from_reference

TASK_ID = 'Workspace-AMP-Dance-Flat-Unitree-G1'


@dataclass
class AmpPpoCfg(RslRlPpoAlgorithmCfg):
    """PPO 的显式 AMP 扩展配置；路径在 build_configs 中验证并填写。"""
    class_name: str = 'workspace.amp.algorithm:AmpPPO'
    amp: dict = field(default_factory=lambda: dict(
        motion_file='', hidden_dims=[256, 128], learning_rate=0.0001,
        batch_size=1024, updates_per_iteration=4, replay_capacity=50000,
        style_reward_scale=2.0, task_reward_weight=1.0, gradient_penalty=5.0,
        logit_regularization=0.05, weight_decay=0.0001, max_grad_norm=1.0,
        control_dt=0.02))


def make_amp_env_cfg(play=False):
    """沿用已验证的 G1/平地/PD，替换为无参考目标输入的风格学习任务。"""
    cfg = unitree_g1_flat_tracking_env_cfg(play=play)
    # 不保留 tracking 的 MotionCommand，防止参考轨迹结尾把机器人传送复位。
    cfg.commands = {}
    cfg.curriculum = {}
    cfg.metrics = {}
    cfg.episode_length_s = 10.0
    terms = {
        'base_lin_vel': ObservationTermCfg(func=mdp.base_lin_vel),
        'base_ang_vel': ObservationTermCfg(func=mdp.base_ang_vel),
        'projected_gravity': ObservationTermCfg(func=mdp.projected_gravity),
        'joint_pos': ObservationTermCfg(func=mdp.joint_pos_rel),
        'joint_vel': ObservationTermCfg(func=mdp.joint_vel_rel),
        'last_action': ObservationTermCfg(func=mdp.last_action),
    }
    cfg.observations = {
        'actor': ObservationGroupCfg(terms=terms, concatenate_terms=True, enable_corruption=False),
        'critic': ObservationGroupCfg(terms=dict(terms), concatenate_terms=True, enable_corruption=False),
        'amp': ObservationGroupCfg(terms={'state': ObservationTermCfg(func=amp_features)},
                                   concatenate_terms=True, enable_corruption=False),
    }
    # 无逐帧参考误差奖励：舞蹈风格由独立判别器学习。以下仅是辅助物理约束。
    cfg.rewards = {
        'alive': RewardTermCfg(func=mdp.is_alive, weight=0.1),
        'failure': RewardTermCfg(func=mdp.is_terminated, weight=-25.0),
        'action_rate': RewardTermCfg(func=mdp.action_rate_l2, weight=-0.01),
        'joint_limits': RewardTermCfg(func=mdp.joint_pos_limits, weight=-1.0),
    }
    cfg.terminations = {
        'time_out': TerminationTermCfg(func=mdp.time_out, time_out=True),
        'fallen': TerminationTermCfg(func=mdp.root_height_below_minimum, params={'minimum_height': 0.45}),
        'orientation': TerminationTermCfg(func=mdp.bad_orientation, params={'limit_angle': 1.2}),
    }
    # 保留 startup 随机化，复位只使用本任务的 RSI；首版不加随机推搡。
    cfg.events = {k: v for k, v in cfg.events.items() if v.mode == 'startup'}
    cfg.events['amp_reset'] = EventTermCfg(func=reset_from_reference, mode='reset',
                                          params={'motion_file': '', 'start': play})
    cfg.events['amp_terminal'] = EventTermCfg(func=capture_terminal_features, mode='step')
    return cfg


def register():
    """可重复调用的注册函数，既供真实初始化，也供 CPU 配置测试使用。"""
    if TASK_ID in list_tasks():
        return
    agent = unitree_g1_tracking_ppo_runner_cfg()
    agent.algorithm = AmpPpoCfg()
    agent.actor.distribution_cfg['std_type'] = 'log'
    agent.actor.distribution_cfg['init_std'] = 0.5
    agent.clip_actions = 3.0
    agent.experiment_name = 'g1_amp_dance'
    register_mjlab_task(TASK_ID, make_amp_env_cfg(), make_amp_env_cfg(play=True), agent)
