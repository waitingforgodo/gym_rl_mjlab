"""评估协议与统计定义：只依赖标准库，便于在没有 GPU 的电脑上测试。

训练配置不在这里修改。固定命令只作用于 load_for_inference 创建的 play
配置副本，训练 profile 的哈希、旧模型和训练中的环境都不受影响。
"""
import math


def configure_fixed_velocity(cfg, values):
    """把 play 环境设为恒定的机体系 (vx, vy, wz)，单位为 m/s、m/s、rad/s。

    不能只把 ranges 改为相等的上下限：官方命令生成器还会随机插入站立、
    前进专用、世界系和航向控制环境，这些分支都可能再次改写速度目标。
    """
    if len(values) != 3 or not all(math.isfinite(v) for v in values):
        raise ValueError('Velocity command must be three finite values: vx vy yaw_rate')
    if 'twist' not in cfg.commands:
        raise ValueError('--velocity-cmd applies only to velocity tasks, not motion tracking')
    command = cfg.commands['twist']
    for name, value in zip(('lin_vel_x', 'lin_vel_y', 'ang_vel_z'), values):
        setattr(command.ranges, name, (float(value), float(value)))
    command.heading_command = False
    # 官方类要求 heading_command=False 时 ranges.heading 也必须为空。
    command.ranges.heading = None
    command.rel_heading_envs = command.rel_standing_envs = 0.
    command.rel_world_envs = command.rel_forward_envs = 0.
    command.init_velocity_prob = 0.  # 从静止开始，包含起步过程。


def velocity_protocol(cfg, fixed=None):
    """把命令采样条件写入评估 JSON，保证两次评估可以核对是否可比。"""
    if 'twist' not in cfg.commands:
        return {'mode': 'motion_tracking'}
    command = cfg.commands['twist']
    return dict(mode='fixed_body_velocity' if fixed is not None else 'official_play_random',
                fixed_vx_vy_yaw_rate=list(fixed) if fixed is not None else None,
                lin_vel_x=command.ranges.lin_vel_x, lin_vel_y=command.ranges.lin_vel_y,
                ang_vel_z=command.ranges.ang_vel_z, heading_command=command.heading_command,
                standing_fraction=command.rel_standing_envs, heading_fraction=command.rel_heading_envs,
                world_frame_fraction=command.rel_world_envs, forward_only_fraction=command.rel_forward_envs,
                resampling_time_range=command.resampling_time_range)


def survival_statistics(failure_counts, seconds):
    """区分‘摔倒事件次数’与‘发生过摔倒的环境数’，不把 5/16 当失败概率。

    该比例只描述这次有限环境数、有限时长的样本。它不是长期可靠性的保证，
    也不是每一个复位后的 episode 都成功的比例。
    """
    if not failure_counts or seconds <= 0 or any(c < 0 for c in failure_counts):
        raise ValueError('Expected nonnegative failure counts and positive exposure time')
    failed = sum(c > 0 for c in failure_counts)
    total = sum(failure_counts)
    count = len(failure_counts)
    return dict(per_env_failure_counts=failure_counts, environments_with_failures=failed,
                environments_without_failures=count-failed,
                no_failure_fraction_over_horizon=(count-failed)/count,
                failure_events_per_environment_minute=total/(count*seconds/60),
                total_exposure_environment_seconds=count*seconds)


def is_direct_metric(name):
    """Episode_* 是复位日志且可能被缓存，不能当逐步固定时长指标再平均。

    尤其 play 的 episode_length_s=1e9，Episode_Reward 会被该分母压得极小。
    保留 Metrics/ 日志供参考；真正的失败数、时长和 RMSE 另行直接累计。
    """
    return not name.startswith(('Episode_Reward/', 'Episode_Termination/', 'Episode_Metrics/'))
