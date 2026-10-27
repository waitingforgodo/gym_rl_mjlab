"""SuperMimic checkpoint 的 mjlab 回放。"""

from .adapter import SuperMimicAdapter
from .config import SuperMimicConfig
from .networks import SuperMimicPolicy
from . import source_sha256


def load_checkpoint(root, checkpoint, device, motion=None):
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from workspace.runtime import build_configs, initialize
    initialize(root)
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    manifest = saved['manifest']
    expected_source = manifest.get('supermimic_source_sha256')
    if expected_source and expected_source != source_sha256(root):
        raise ValueError('SuperMimic source differs from the checkpoint contract')
    profile = manifest['profile_spec']
    config = SuperMimicConfig.from_profile(profile, num_envs=1,
                                            iterations=saved['config']['iterations'])
    _, env_cfg, _, _ = build_configs(
        root, manifest['profile'], 1, motion or manifest['motion_file'], play=True,
        seed=manifest['seed'], profile_spec=profile,
        profile_sha256=manifest['profile_sha256'])
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    env.reset(seed=manifest['seed'])
    adapter = SuperMimicAdapter(env, config, noisy=False)
    adapter.initialize_history(adapter.observation())
    # 回放/导出只构造 actor；不分配世界模型、critic 或三个 optimizer。
    policy = SuperMimicPolicy(
        config.observation_dim, config.goal_frame_dim, config.action_dim,
        config.history_length, config.future_frames, config.latent_dim,
        config.motion_encoder_hidden, config.decoder_hidden).to(device)
    policy.load_state_dict(saved['learner']['policy'])
    policy.eval()
    return env, adapter, policy, manifest


class LivePolicy:
    """Viser viewer 所需 callable；观测直接从 adapter 构造，忽略 PPO obs。"""
    def __init__(self, adapter, policy):
        self.adapter, self.policy = adapter, policy

    def __call__(self, _):
        import torch
        with torch.inference_mode():
            observation = self.adapter.observation()
            reset_ids = (self.adapter.env.episode_length_buf == 0).nonzero(
                as_tuple=False).squeeze(-1)
            self.adapter.advance_history(observation, reset_ids)
            return self.policy(self.adapter.history, self.adapter.future_goal())

    def reset(self):
        self.adapter.initialize_history()


def play(root, args):
    from workspace.inference import view
    env, adapter, policy, _ = load_checkpoint(root, args.checkpoint, args.device, args.motion)
    try:
        view(env, LivePolicy(adapter, policy), args.viewer, args.port)
    finally:
        env.close()
