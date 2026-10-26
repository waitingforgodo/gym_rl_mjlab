"""mjlab 管理器回调：参考状态复位、AMP 观测以及复位前的终止帧捕获。"""
import torch

from .features import KEY_BODIES, build_features
from .reference import AmpReference


def reference_for_env(env, motion_file):
    """同一环境缓存一份专家数据；不在每步读取磁盘。"""
    if not hasattr(env, '_amp_reference'):
        env._amp_reference = AmpReference(motion_file, env.device)
    return env._amp_reference


def amp_features(env):
    """读取真实机器人状态；只用于判别器，不进入 actor/critic 的观测组。"""
    robot = env.scene['robot']
    if not hasattr(env, '_amp_key_indices'):
        env._amp_key_indices = [robot.body_names.index(name) for name in KEY_BODIES]
    d = robot.data
    return build_features(d.joint_pos, d.joint_vel,
                          d.root_link_pos_w-env.scene.env_origins, d.root_link_quat_w,
                          d.root_link_lin_vel_w, d.root_link_ang_vel_w,
                          d.body_link_pos_w[:, env._amp_key_indices]-env.scene.env_origins[:, None])


def capture_terminal_features(env, env_ids=None):
    """在 mjlab 自动复位之前捕获真实下一帧，包括跌倒和超时帧。

    mjlab 的 step 事件发生在物理积分后、自动复位前。这里额外 forward
    刷新刚体位姿，确保它与最新 qpos/qvel 对齐；随后克隆以免被复位覆盖。
    直接使用 step 返回的观测会把跌倒前状态和复位后姿态错误地拼成动作。
    """
    env.sim.forward()
    env.extras['amp_terminal_features'] = amp_features(env).detach().clone()


def reset_from_reference(env, env_ids, motion_file, start=False):
    """参考状态初始化（RSI）：训练从随机参考帧出发，回放从第 0 帧出发。

    该回调只负责初态，不在每步强制设置机器人位置。之后完全由策略和
    MuJoCo 物理产生动作，没有参考跟踪奖励，也没有参考进度输入。
    """
    ref = reference_for_env(env, motion_file)
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    ids = (torch.zeros(len(env_ids), device=env.device, dtype=torch.long) if start
           else torch.randint(ref.frames-1, (len(env_ids),), device=env.device))
    a, root = ref.arrays, ref.root_index
    position = a['body_pos_w'][ids, root].clone()+env.scene.env_origins[env_ids]
    position[:, 2] += 0.02  # 轻微抬升，减少落地初态的数值穿透。
    root_state = torch.cat((position, a['body_quat_w'][ids, root],
                            a['body_lin_vel_w'][ids, root], a['body_ang_vel_w'][ids, root]), -1)
    robot = env.scene['robot']
    limits = robot.data.soft_joint_pos_limits[env_ids]
    q = a['joint_pos'][ids].clamp(limits[..., 0], limits[..., 1])
    robot.write_joint_state_to_sim(q, a['joint_vel'][ids], env_ids=env_ids)
    robot.write_root_state_to_sim(root_state, env_ids=env_ids)
