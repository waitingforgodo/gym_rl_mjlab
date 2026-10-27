"""只加载导出 JIT 的 SuperMimic 仿真部署回归。"""
import json


def deploy(root, args):
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from workspace.contract import sha256
    from workspace.runtime import build_configs, initialize
    from workspace.inference import view
    from .adapter import SuperMimicAdapter
    from .config import SuperMimicConfig
    from .inference import LivePolicy
    from . import source_sha256
    bundle = args.bundle.resolve()
    manifest = json.loads((bundle/'manifest.json').read_text())
    if manifest.get('supermimic_source_sha256') != source_sha256(root):
        raise ValueError('SuperMimic source differs from the exported bundle contract')
    hashes = json.loads((bundle/'hashes.json').read_text())
    for name, expected in hashes.items():
        if sha256(bundle/name) != expected:
            raise ValueError(f'Bundle integrity differs: {name}')
    initialize(root)
    profile = manifest['profile_spec']
    config = SuperMimicConfig.from_profile(profile, num_envs=1)
    _, env_cfg, _, _ = build_configs(
        root, manifest['profile'], 1, bundle/'motion.npz', play=True,
        seed=manifest['seed'], profile_spec=profile,
        profile_sha256=manifest['profile_sha256'])
    env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device)
    env.reset(seed=manifest['seed'])
    adapter = SuperMimicAdapter(env, config, noisy=False)
    adapter.initialize_history()
    model = torch.jit.load(str(bundle/'policy_jit.pt'), map_location=args.device).eval()
    policy = LivePolicy(adapter, model)
    try:
        if args.headless:
            steps = int(args.seconds/env.step_dt)
            with torch.inference_mode():
                for _ in range(steps):
                    env.step(policy(None))
            return {'status': 'passed', 'steps': steps, 'simulation_seconds': steps*env.step_dt}
        view(env, policy, args.viewer, args.port)
    finally:
        env.close()
