"""上游策略回放、整段 benchmark 与当前 mjlab 的奖励接口适配。"""
import argparse
from datetime import datetime
import json
from pathlib import Path
from .paths import ROOT
from .configuration import positive, merge_config, data_files, digest
from .runtime import initialize
from .checkpoints import resolve_checkpoint, load_checkpoint, check_data_contract


def inference_parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--checkpoint', default='latest', help='New upstream checkpoint, run directory, or latest')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--envs', type=positive, default=1)
    return p


def inference_context(args):
    path = resolve_checkpoint(args.checkpoint)
    initialize()
    from supermimic.config import TrainConfig
    from supermimic.eval.benchmark import load_policy_from_checkpoint
    saved = load_checkpoint(path)
    cfg = merge_config(TrainConfig(), saved['config'])
    del saved
    if cfg.data.library is not None:
        raise ValueError('Replay/benchmark requires original NPZ files; packed-library-only inference is not supported by upstream benchmark')
    files = data_files(cfg)
    check_data_contract(path, files)
    policy, _ = load_policy_from_checkpoint(path, args.device)
    from .safety import require_finite
    policy.register_forward_hook(lambda module, inputs, output: require_finite(output, 'inference actions'))
    print(f'CHECKPOINT={path}', flush=True)
    return path, policy, cfg, files


def play(args):
    _, policy, cfg, files = inference_context(args)
    import torch
    from supermimic.envs import SuperMimicEnvCfgArgs, make_env
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer
    # 上游 play.py 的确定性环境设置；通过同一 make_env 构造，保持奖励/终止定义。
    env = make_env(SuperMimicEnvCfgArgs(
        npz_files=[str(p) for p in files], num_envs=args.envs, episode_length_s=1.e5,
        action_noise_std=0., obs_noise=False, reset_noise=False, domain_randomization=False,
        sampling_mode='start', record_transitions=False, anchor_obs=cfg.env.anchor_obs,
        debug_vis=not args.headless, future_frames=cfg.env.future_frames), args.device)
    server = None
    try:
        if args.headless:
            obs, _ = env.reset()
            total, failed = 0., 0
            with torch.no_grad():
                for _ in range(args.steps):
                    obs, reward, term, _, _ = env.step(policy(obs['policy'], obs['goal']))
                    if not torch.isfinite(reward).all() or not all(torch.isfinite(v).all() for v in obs.values()):
                        raise FloatingPointError('Nonfinite simulation rollout')
                    total += float(reward.mean())
                    failed += int(term.sum())
            print(json.dumps(dict(steps=args.steps, mean_step_reward=total/args.steps, termination_events=failed)))
        else:
            wrapped = RslRlVecEnvWrapper(env)
            @torch.no_grad()
            def actor(obs):
                return policy(obs['policy'], obs['goal'])
            if args.viewer == 'native':
                NativeMujocoViewer(wrapped, actor).run()
            else:
                import viser
                server = viser.ViserServer(host='0.0.0.0', port=args.port)
                print(f'网页回放：http://localhost:{server.get_port()}', flush=True)
                ViserPlayViewer(wrapped, actor, viser_server=server).run()
    finally:
        env.close()
        if server:
            server.stop()


def evaluate(args):
    path, policy, cfg, files = inference_context(args)
    from supermimic.eval import benchmark
    from unittest.mock import patch
    original_make_env = benchmark.make_env

    def compatible_make_env(*a, **kw):
        env = original_make_env(*a, **kw)
        # mjlab 1.6 的 _step_reward 是加权后的「每秒奖励率」，上游新接口
        # term_rewards 是每控制步奖励。必须乘 step_dt，不能直接改属性名。
        # 仅代理本评测实例；不修改安装包，也不改全局 RewardManager 类。
        if not hasattr(env.reward_manager, 'term_rewards'):
            env.reward_manager = RewardManagerView(env.reward_manager, env.step_dt)
        return env

    with patch.object(benchmark, 'make_env', compatible_make_env):
        rows = benchmark.run_benchmark(policy, [str(p) for p in files], device=args.device,
                                       num_envs=args.envs, future_frames=cfg.env.future_frames,
                                       anchor_obs=cfg.env.anchor_obs)
    output = args.output or ROOT / 'outputs/supermimic/evaluations' / f'{datetime.now():%Y%m%d_%H%M%S_%f}.json'
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    benchmark.save_results(rows, output)
    report = json.loads(output.read_text())
    report.update(checkpoint=str(path), protocol='upstream Appendix F deterministic whole clip',
                  data=[dict(path=str(p), sha256=digest(p)) for p in files])
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f'EVALUATION={output}\n{json.dumps(benchmark.summarize(rows), ensure_ascii=False)}', flush=True)


class RewardManagerView:
    """把 mjlab 1.6 奖励率映射为新版逐步奖励，其他行为原样转发。"""
    def __init__(self, manager, step_dt):
        self.manager, self.step_dt = manager, step_dt

    def __getattr__(self, key):
        return getattr(self.manager, key)

    @property
    def term_rewards(self):
        return self.manager._step_reward * (self.step_dt if self.manager._scale_by_dt else 1.)
