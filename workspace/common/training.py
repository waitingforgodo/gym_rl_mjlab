"""训练和续训的实际实现。

入口：scripts/reinforcement_learning/rsl_rl/train.py、resume.py。
本文件按“配置 → 清单 → 环境 → 恢复 → PPO → 清理”排列。
PPO 采样/GAE/优化循环由固定版本 RSL-RL 执行，源码位置见 show-sources。"""
from dataclasses import asdict

from .contract import validate_resume, write_json
from .configuration import inherited_run_options
from .runtime import initialize, build_configs, checkpoint_context, make_env_runner, new_run


def train(root, args, checkpoint=None):
    """训练或续训一次；iterations 表示本次新增更新数，始终创建新目录。

    resume 同时恢复 actor、critic、优化器及官方保存的环境课程计数。
    不恢复完整物理状态/RNG 轨迹，因此属于继续学习而非逐位确定性重放。
    """
    from .telemetry import ResourceSampler
    # 1. 先验证版本/设置缓存，再导入任务；此时还未创建仿真或神经网络。
    initialize(root)
    saved = None
    profile, motion = args.profile or 'velocity_flat', args.motion
    if checkpoint:
        # 续训优先恢复原任务和动作文件，不依赖本次命令的默认 profile。
        saved, profile, motion = checkpoint_context(root, checkpoint, args.profile, motion, args.legacy_velocity)
    # 2. 合并 profile、官方环境配置、PPO 配置和命令行覆盖。
    # 未显式指定时，续训继承原来的并行数与种子；tracking 不会再被平地
    # 脚本的 4096 默认值覆盖。从零训练则取 profile 环境数和默认种子 42。
    num_envs, seed = inherited_run_options(args, saved)
    spec, cfg, agent, manifest = build_configs(
        root, profile, num_envs, motion, seed=seed,
        profile_spec=saved.get('profile_spec') if saved else None,
        profile_sha256=saved.get('profile_sha256') if saved else None)
    if spec.get('learning_method') == 'supermimic':
        raise ValueError('SuperMimic uses its own world-model/BPTT trainer. Run bash scripts/19_train_supermimic.sh instead of the PPO entry point.')
    if saved:
        # 允许调整并行环境数；观测任务、动作数据、版本、控制周期等必须一致。
        validate_resume(saved, manifest)
    iterations = args.iterations or spec['iterations']
    agent.max_iterations = iterations
    # 3. 在本次独立目录保存清单和完整配置。模型、曲线也写到这个目录。
    from .profiles import method_of
    method = method_of(spec)
    run = new_run(root, f'{profile}_{args.name}', getattr(args, 'benchmark_run', False), method)
    manifest['method'] = method
    manifest.update(source_checkpoint=str(checkpoint) if checkpoint else None, additional_iterations=iterations,
                    legacy_contract_unverified=checkpoint is not None and saved is None,
                    elapsed_time_clock='perf_counter (process-local RSL-RL module binding)')
    write_json(run / 'run_manifest.json', manifest)
    from mjlab.utils.os import dump_yaml
    dump_yaml(run / 'params' / 'env.yaml', asdict(cfg))
    dump_yaml(run / 'params' / 'agent.yaml', asdict(agent))
    # runner 会修改传入字典，参数快照必须在构造 runner 之前写出。
    print(f'RUN_DIR={run}', flush=True)
    sampler = ResourceSampler()
    sampler.start()
    env = runner = None
    status = 'failed'
    try:
        # 4. ManagerBasedRlEnv 执行 MuJoCo/Warp 物理和各 manager。
        # RslRlVecEnvWrapper 适配接口；runner 创建 actor、critic、PPO 和缓存。
        env, runner = make_env_runner(cfg, agent, spec['task'], args.device, run)
        if checkpoint:
            # 5. 默认 load_cfg=None 恢复训练状态，不能只载入 actor 来假装续训。
            runner.load(str(checkpoint), strict=True, map_location=args.device)
            # RSL-RL 保存的是最后完成的编号，learn 又从当前编号开始，故加一。
            runner.current_learning_iteration += 1
            if manifest.get('recipe'):
                # 环境构造时公共步计数从 0 开始，load 才恢复课程计数。
                # 重新 reset，让本次第一次采样就使用恢复后的课程阶段。
                env.reset()
        import rsl_rl.runners.on_policy_runner as runner_module
        from .timing import monotonic_runner_clock
        with monotonic_runner_clock(runner_module):
            # 6. 每轮每环境采集 num_steps_per_env 步（当前官方配置为 24）。
            # runner 内部执行：actor 采样动作 → env.step → 保存转移，
            # 然后 critic bootstrap → GAE/return → 多个 minibatch 的 PPO 更新。
            # 随机初始 episode 长度可错开各环境的超时，避免所有环境同时复位。
            runner.learn(num_learning_iterations=iterations, init_at_random_ep_len=True)
        status = 'passed'
    finally:
        # 7. 正常结束、训练报错或 Ctrl+C 都进入清理。passed 只表示调用成功，
        # 不表示策略已经收敛或机器人不会摔倒；质量由独立评估程序判断。
        # 该版本没有自动关闭普通 TensorBoard writer。主动 flush/close，
        # 保证短测结束后读取 event 文件时不会丢最后几轮。
        if runner is not None and runner.logger.writer is not None:
            writer = runner.logger.writer
            if hasattr(writer, 'flush'):
                writer.flush()
            if hasattr(writer, 'close'):
                writer.close()
        if env is not None:
            env.close()
        resource = sampler.finish()
        resource.update(status=status, num_envs=cfg.scene.num_envs)
        write_json(run / 'resources.json', resource)
    return run
