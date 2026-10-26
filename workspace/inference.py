"""策略回放和定量评估的实际实现。

入口：scripts/reinforcement_learning/rsl_rl/play.py、evaluate.py。
这里仅执行 actor 前向推理；不更新网络参数或观测归一化统计。
评估保留失败终止和自动复位，并分别统计失败事件与失败环境数。"""
import math

from .contract import validate_resume
from .runtime import initialize, build_configs, checkpoint_context, make_env_runner


def load_for_inference(root, args, checkpoint, num_envs=1):
    """创建推理环境，只加载 actor 并切换 eval 模式，冻结训练归一化统计。

    固定速度的覆盖在续训契约验证之后应用，仅属于本次评估协议；不把
    评估用的 0.5m/s 限制写回训练 profile 或旧运行目录。
    """
    initialize(root)
    # 从 checkpoint 同目录清单恢复 profile/动作，play 配置关闭部分训练扰动。
    saved, profile, motion = checkpoint_context(root, checkpoint, args.profile, args.motion, args.legacy_velocity)
    spec, cfg, agent, manifest = build_configs(root, profile, num_envs, motion, play=True,
                                               seed=saved.get('seed', 42) if saved else 42)
    if saved:
        validate_resume(saved, manifest)
    from .evaluation import configure_fixed_velocity, velocity_protocol
    fixed = getattr(args, 'velocity_cmd', None)
    if fixed is not None:
        configure_fixed_velocity(cfg, fixed)
    manifest['evaluation_conditions'] = dict(
        command=velocity_protocol(cfg, fixed), seed=manifest['seed'], control_dt=manifest['control_dt'],
        episode_length_s=cfg.episode_length_s, pushes_enabled='push_robot' in cfg.events,
        actor_noise_enabled=cfg.observations['actor'].enable_corruption)
    env, runner = make_env_runner(cfg, agent, spec['task'], args.device)
    try:
        # 推理只需要 actor，get_inference_policy 会进入 eval 模式。
        runner.load(str(checkpoint), load_cfg={'actor': True}, strict=True, map_location=args.device)
        policy = runner.get_inference_policy(device=args.device)
        return env, runner, policy, manifest
    except BaseException:
        env.close()
        raise


def view(env, policy, viewer='viser', port=8080):
    """以无梯度模式回放；Viser 仅监听本机回环地址，供 Windows 浏览器访问。"""
    import torch
    from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer
    server = None
    try:
        with torch.inference_mode():
            if viewer == 'native':
                NativeMujocoViewer(env, policy).run()
            else:
                import viser
                server = viser.ViserServer(host='127.0.0.1', port=port)
                print(f'Open in Windows browser: http://localhost:{port}', flush=True)
                ViserPlayViewer(env, policy, viser_server=server).run()
    finally:
        if server:
            server.stop()


def evaluate(env, policy, seconds=60):
    """固定仿真时长内直接累计终止、存活时间与任务误差。

    失败后环境仍自动复位，所以‘事件数’可能大于‘失败环境数’。已完成
    episode 的均值不包含仍在运行的样本。RMSE 在施加动作前按当前命令
    与当前机器人状态采样，包含起步、命令切换和失败后重新起步的误差。
    """
    import torch
    from .evaluation import is_direct_metric, survival_statistics
    base = env.unwrapped
    obs = env.get_observations()
    # seconds 按仿真时间解释；渲染速度、GPU 吞吐不会改变采样总时长。
    steps = math.ceil(seconds / base.step_dt)
    lengths = torch.zeros(env.num_envs, device=base.device)
    failures_per_env = torch.zeros(env.num_envs, dtype=torch.long, device=base.device)
    first_failure = torch.full((env.num_envs,), float('nan'), device=base.device)
    ended_lengths, timeout_count, failure_count, reward_sum = [], 0, 0, 0.
    causes, metrics, task_errors = {}, {}, {}
    with torch.inference_mode():
        for step in range(steps):
            if 'twist' in base.command_manager.active_terms:
                command = base.command_manager.get_command('twist')
                robot = base.scene['robot']
                xy_error = command[:, :2] - robot.data.root_link_lin_vel_b[:, :2]
                yaw_error = command[:, 2] - robot.data.root_link_ang_vel_b[:, 2]
                task_errors.setdefault('velocity_xy_squared_norm', []).append(float(xy_error.square().sum(-1).mean()))
                task_errors.setdefault('yaw_rate_squared_error', []).append(float(yaw_error.square().mean()))
            if 'motion' in base.command_manager.active_terms:
                motion = base.command_manager.get_term('motion')
                error = motion.joint_pos - motion.robot_joint_pos
                task_errors.setdefault('joint_angle_squared_mean', []).append(float(error.square().mean()))
                error = motion.body_pos_relative_w - motion.robot_body_pos_w
                task_errors.setdefault('aligned_body_position_norm_mean', []).append(float(error.norm(dim=-1).mean()))
            actions = policy(obs)
            if not torch.isfinite(actions).all():
                raise FloatingPointError('Policy produced non-finite actions')
            # 用底层五元组保留 terminated 与 truncated；同步复现 wrapper 裁剪。
            if env.clip_actions is not None:
                actions = torch.clamp(actions, -env.clip_actions, env.clip_actions)
            obs_dict, rewards, terminated, truncated, extras = base.step(actions)
            # 自动复位后继续评估；terminated 是失败事件，truncated 是时间截断。
            # 二者分开累计，才能区分“运行满时长”和“摔倒后反复起身”。
            from tensordict import TensorDict
            obs = TensorDict(obs_dict, batch_size=[base.num_envs])
            lengths += 1
            done = terminated | truncated
            ended_lengths.extend((lengths[done] * base.step_dt).cpu().tolist())
            lengths[done] = 0
            failure_count += int(terminated.sum())
            first_failure[(failures_per_env == 0) & terminated] = (step+1)*base.step_dt
            failures_per_env += terminated.to(torch.long)
            timeout_count += int(truncated.sum())
            reward_sum += float(rewards.mean())
            # 跳过不能按当前固定时长解释的复位日志，避免输出接近 1e-8 的伪诊断。
            for name, value in extras.get('log', {}).items():
                if not is_direct_metric(name):
                    continue
                if isinstance(value, (float, int)) or (torch.is_tensor(value) and value.numel() == 1):
                    metrics.setdefault(name, []).append(float(value))
            manager = base.termination_manager
            for name in manager.active_terms:
                value = manager.get_term(name)
                causes[name] = causes.get(name, 0) + int(value.sum())
    import statistics
    return dict(sim_seconds_per_env=steps*base.step_dt, num_envs=env.num_envs,
                **survival_statistics(failures_per_env.cpu().tolist(), steps*base.step_dt),
                failure_terminations=failure_count, timeouts=timeout_count, termination_causes=causes,
                first_failure_time_seconds=[float(v) if math.isfinite(v) else None for v in first_failure.cpu().tolist()],
                completed_episode_seconds_mean=statistics.mean(ended_lengths) if ended_lengths else None,
                completed_episode_seconds_scope='Completed episodes only; unfinished survivors are excluded from this mean.',
                unfinished_episode_seconds=lengths.mul(base.step_dt).cpu().tolist(),
                reward_per_step_mean=reward_sum/steps,
                logged_metrics_snapshot_mean={k: statistics.mean(v) for k, v in metrics.items()},
                logged_metrics_note='Reference snapshots only; Episode_* reset logs excluded. Use direct termination counts and sampled RMSE.',
                sampled_errors={k: statistics.mean(v) for k, v in task_errors.items()},
                velocity_xy_rmse_m_per_s=math.sqrt(statistics.mean(task_errors['velocity_xy_squared_norm'])) if 'velocity_xy_squared_norm' in task_errors else None,
                yaw_rate_rmse_rad_per_s=math.sqrt(statistics.mean(task_errors['yaw_rate_squared_error'])) if 'yaw_rate_squared_error' in task_errors else None,
                joint_angle_rmse_rad=math.sqrt(statistics.mean(task_errors['joint_angle_squared_mean'])) if 'joint_angle_squared_mean' in task_errors else None,
                interpretation='Read termination causes and task errors; a finished command alone is not stable walking.')
