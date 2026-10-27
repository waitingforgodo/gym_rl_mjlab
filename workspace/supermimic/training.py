"""SuperMimic 数据采集与交替优化主循环（不借用 PPO runner）。"""
from dataclasses import asdict
from datetime import datetime

from .adapter import SuperMimicAdapter
from .config import SuperMimicConfig
from .learner import SuperMimicLearner
from .replay import SequenceReplay
from .reset_sampler import ReturnResetSampler
from . import source_sha256


def _save(path, learner, config, manifest, replay, collection_iteration, data_credit):
    """checkpoint 包含三张网络和优化器；回放仅存元数据，续训会重新 warmup。"""
    import torch
    torch.save(dict(schema=1, algorithm='supermimic', config=config.to_dict(),
                    manifest=manifest, learner=learner.state_dict(),
                    collection_iteration=collection_iteration,
                    data_credit=data_credit, replay=replay.state_dict()), path)


def train(root, args):
    """每个控制步采集真实转移，按 UTD=0.4 交替拟合 WM 和 BPTT 策略。"""
    import torch
    from torch.utils.tensorboard import SummaryWriter
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.utils.os import dump_yaml
    from workspace.contract import write_json
    from workspace.runtime import build_configs, initialize
    from workspace.telemetry import ResourceSampler

    initialize(root)
    profile_name = args.profile
    profile, env_cfg, _, manifest = build_configs(
        root, profile_name, args.envs, args.motion, seed=args.seed)
    config = SuperMimicConfig.from_profile(
        profile, num_envs=args.envs, iterations=args.iterations)
    env_cfg.auto_reset = False
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    run = root/'outputs'/'supermimic'/'runs'/f'{stamp}_{args.name}_4096'
    (run/'params').mkdir(parents=True)
    manifest.update(algorithm='supermimic_independent_reproduction',
                    paper='arXiv:2610.28480v1', supermimic=config.to_dict(),
                    supermimic_source_sha256=source_sha256(root),
                    implementation_limits=['official source unavailable',
                                           'unpublished world/goal layouts use 71-D generalized states'])
    write_json(run/'run_manifest.json', manifest)
    dump_yaml(run/'params'/'env.yaml', asdict(env_cfg))
    write_json(run/'params'/'supermimic.json', config.to_dict())
    print(f'RUN_DIR={run}', flush=True)

    sampler, env, writer = ResourceSampler(), None, SummaryWriter(str(run))
    sampler.start()
    status = 'failed'
    try:
        env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device)
        obs_dict, _ = env.reset(seed=args.seed)
        adapter = SuperMimicAdapter(env, config, noisy=True)
        reset_sampler = ReturnResetSampler(env, config.gamma)
        observation = adapter.observation()
        adapter.initialize_history(observation)
        state_dim = adapter.state(obs_dict).shape[1]
        learner = SuperMimicLearner(config, state_dim, args.device)
        replay = SequenceReplay(config.replay_steps_per_env, env.num_envs)
        start = 0
        if args.checkpoint:
            saved = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
            if saved.get('algorithm') != 'supermimic':
                raise ValueError('Checkpoint is not a SuperMimic checkpoint')
            if saved['manifest'].get('supermimic_source_sha256') != source_sha256(root):
                raise ValueError('SuperMimic source differs from the checkpoint contract')
            from workspace.contract import validate_resume
            validate_resume(saved['manifest'], manifest)
            learner.load_state_dict(saved['learner'], training=True)
            start = int(saved.get('collection_iteration', 0))
            data_credit = float(saved.get('data_credit', 0.0))
        else:
            data_credit = 0.0
        minimum = max(config.world_window+config.future_frames-1,
                      learner.sequence_length)

        def collect_step():
            """采一帧所有环境；先保存 terminal state，再手动复位 done 环境。"""
            nonlocal obs_dict, observation
            state = adapter.state(obs_dict).detach()
            context = adapter.context().detach()
            goal = adapter.goal().detach()
            with torch.no_grad():
                action = learner.policy(adapter.history, adapter.future_goal())
                noisy_action = (action + torch.randn_like(action)*config.action_noise_std).clamp(-1, 1)
            next_dict, reward, terminated, truncated, _ = env.step(noisy_action)
            # auto_reset=False：这里读到的是真实终止帧，而不是复位后的初态。
            next_state = adapter.state(next_dict).detach()
            next_observation = adapter.observation().detach()
            clean_next_observation = adapter.observation(noisy=False).detach()
            boundary = terminated | truncated
            reset_sampler.record(reward, boundary)
            replay.append(state=state, next_state=next_state, observation=observation,
                          next_observation=next_observation,
                          clean_next_observation=clean_next_observation,
                          goal=goal, action=noisy_action,
                          reward=reward, terminated=terminated, boundary=boundary,
                          context=context)
            reset_ids = boundary.nonzero(as_tuple=False).squeeze(-1)
            if len(reset_ids):
                obs_dict, _ = env.reset(env_ids=reset_ids)
                next_observation = adapter.observation().detach()
            else:
                obs_dict = next_dict
            observation = next_observation
            adapter.advance_history(observation, reset_ids)
            return reward, terminated

        # checkpoint 不携带数 GB replay；从零训练和续训都先积累可形成 H+K
        # 连续窗口的数据。之后 UTD=0.4 即每次更新平均采 2.5 个真实控制步。
        while len(replay) < minimum:
            collect_step()
        for iteration in range(start, config.iterations):
            data_credit += 1.0/config.update_to_data_ratio
            collect_count = int(data_credit)
            data_credit -= collect_count
            rewards, terminations = [], []
            for _ in range(collect_count):
                reward, terminated = collect_step()
                rewards.append(reward.mean())
                terminations.append(terminated.float().mean())
            metrics = learner.fit_world(replay)
            for _ in range(config.policy_updates_per_world_update):
                metrics.update(learner.improve_policy(replay))
            for key, value in metrics.items():
                writer.add_scalar(f'train/{key}', value, iteration)
            writer.add_scalar('rollout/reward_mean', float(torch.stack(rewards).mean()), iteration)
            writer.add_scalar('rollout/termination_rate',
                              float(torch.stack(terminations).mean()), iteration)
            if (iteration+1) % config.save_interval == 0:
                _save(run/f'supermimic_{iteration+1}.pt', learner, config, manifest, replay,
                      iteration+1, data_credit)
                writer.flush()
        _save(run/f'supermimic_{config.iterations}.pt', learner, config, manifest, replay,
              config.iterations, data_credit)
        status = 'passed'
        return run
    finally:
        writer.flush(); writer.close()
        if env is not None:
            env.close()
        resource = sampler.finish()
        resource.update(status=status, num_envs=config.num_envs)
        write_json(run/'resources.json', resource)
