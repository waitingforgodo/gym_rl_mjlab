"""G1 新实验的环境配方：保留固定 mjlab 物理和观测，显式改变学习任务。

velocity_robust_v1 结合 gym_rl 的失败/能耗约束与分阶段速度课程。
tracking_robust_v1 保留 BeyondMimic 的失败片段自适应采样。
这些是有来源、可验证的候选，不代表已通过对照实验证明的全局最优。
本文件哈希随模型保存，修改配方实现后会拒绝无记录地续训旧实验。
"""


def apply_recipe(cfg, agent, name=None, play=False):
    """只处理显式选择的新配方；旧 profile 无 recipe 字段时不改变环境。"""
    if name is None:
        return
    if name == 'velocity_robust_v1':
        if 'twist' not in cfg.commands or 'motion' in cfg.commands:
            raise ValueError('velocity_robust_v1 requires the velocity task')
        _velocity(cfg, agent, play)
    elif name == 'tracking_robust_v1':
        if 'motion' not in cfg.commands:
            raise ValueError('tracking_robust_v1 requires a motion tracking task')
        # 默认即为官方 adaptive，仍显式声明，避免以后的配置漂移。
        # play 从动作开头播放；训练根据失败片段增加采样，并保留均匀探索。
        cfg.commands['motion'].sampling_mode = 'start' if play else 'adaptive'
    else:
        raise ValueError(f'Unknown training recipe: {name}')


def _velocity(cfg, agent, play):
    """分阶段增加命令难度，并加入可单独观察的失败和下肢代价。

    不改 PD、动作尺度、关节顺序、99/111 维观测或 50 Hz 控制频率。
    代价系数以 gym_rl/robot_lab 的 G1 flat 配置为起点，MuJoCo 与 PhysX
    仍有差异，必须通过同协议评估决定是否保留，不能只比较总奖励。
    """
    from mjlab.envs import mdp
    from mjlab.managers.reward_manager import RewardTermCfg
    from mjlab.managers.scene_entity_config import SceneEntityCfg

    # 同一名称的 penalty 只惩罚 terminated，time_out 不会受到失败惩罚。
    # 当前 dt=0.02，因此 -200 在失败那一步实际贡献约 -4，不是每步 -200。
    cfg.rewards['failure_penalty'] = RewardTermCfg(func=mdp.is_terminated, weight=-200.)
    # 注意：mjlab 的力矩函数索引 actuator_force，必须选择 actuator_names；
    # 关节加速度函数索引 joint_acc，才使用 joint_names，不能混用两套下标。
    cfg.rewards['leg_torque_l2'] = RewardTermCfg(
        func=mdp.joint_torques_l2, weight=-2.e-6,
        params={'asset_cfg': SceneEntityCfg('robot', actuator_names=r'.*(hip|knee).*')})
    cfg.rewards['leg_acc_l2'] = RewardTermCfg(
        func=mdp.joint_acc_l2, weight=-1.e-7,
        params={'asset_cfg': SceneEntityCfg('robot', joint_names=(r'.*hip.*', r'.*knee.*'))})

    # 阶段按每个环境的公共控制步推进，不乘 num_envs。PPO 每轮采集 24
    # 步时，1500 次更新对应 36000 个控制步；续训由官方 runner 恢复此计数。
    stages = [
        (0, (-0.5, 1.0), (-0.25, 0.25), (-0.5, 0.5)),
        (1500, (-1.0, 1.5), (-0.5, 0.5), (-0.7, 0.7)),
        (4000, (-1.5, 2.0), (-1.0, 1.0), (-0.7, 0.7)),
    ]
    command = cfg.commands['twist']
    _, x, y, yaw = stages[-1] if play else stages[0]
    command.ranges.lin_vel_x = x
    command.ranges.lin_vel_y = y
    command.ranges.ang_vel_z = yaw
    if not play:
        # 复用 mjlab 官方课程函数；阶段值写入 env.yaml，没有进程外隐式状态。
        cfg.curriculum['command_vel'].params['velocity_stages'] = [
            dict(step=update * agent.num_steps_per_env, lin_vel_x=x, lin_vel_y=y, ang_vel_z=yaw)
            for update, x, y, yaw in stages]
