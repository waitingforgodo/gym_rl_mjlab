"""SuperMimic 论文参数的强类型表示和前置检查。"""
from dataclasses import asdict, dataclass, field
import math


@dataclass
class SuperMimicConfig:
    """论文表 4 的单 GPU设置；microbatch 仅控制显存，不改变总 batch。"""

    num_envs: int = 4096
    iterations: int = 320_000
    action_dim: int = 29
    observation_dim: int = 93
    goal_frame_dim: int = 71
    context_dim: int = 33
    history_length: int = 10
    future_frames: int = 10
    latent_dim: int = 32
    world_hidden: tuple[int, ...] = (2048, 2048, 2048)
    motion_encoder_hidden: tuple[int, ...] = (2048, 1024, 512, 256)
    decoder_hidden: tuple[int, ...] = (2048, 1024, 512, 256)
    critic_hidden: tuple[int, ...] = (256, 128)
    head_hidden: tuple[int, ...] = (256, 128)
    world_window: int = 8
    imagination_horizon: int = 32
    world_batch_size: int = 2048
    world_batches_per_update: int = 10
    imagination_batch_size: int = 4096
    imagination_microbatch: int = 256
    replay_steps_per_env: int = 1000
    update_to_data_ratio: float = 0.4
    policy_updates_per_world_update: int = 1
    learning_rate: float = 2e-4
    adam_betas: tuple[float, float] = (0.9, 0.95)
    gamma: float = 0.99
    gae_lambda: float = 0.95
    action_noise_std: float = 0.1
    max_grad_norm: float = 1.0
    policy_element_clip: float = 0.1
    critic_epochs: int = 5
    critic_minibatches: int = 4
    lr_decay_iterations: int = 250_000
    lr_final_fraction: float = 0.03
    save_interval: int = 5000
    # 论文未公布 learned reward/observation/termination head 的宽度；这是本复现
    # 明确记录的实现选择，不伪装成论文原值。
    learned_heads: bool = True
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_profile(cls, profile: dict, *, num_envs=None, iterations=None):
        """从 JSON 的 supermimic 字段构造，并允许命令行只覆盖规模/预算。"""
        values = dict(profile.get('supermimic', {}))
        for key in ('world_hidden', 'motion_encoder_hidden', 'decoder_hidden',
                    'critic_hidden', 'head_hidden', 'adam_betas'):
            if key in values:
                values[key] = tuple(values[key])
        values['num_envs'] = num_envs or profile['num_envs']
        values['iterations'] = iterations or profile['iterations']
        result = cls(**values)
        result.validate()
        return result

    def validate(self):
        """在创建仿真/大缓冲前拒绝维度、批量和概率错误。"""
        positive_ints = (
            'num_envs', 'iterations', 'action_dim', 'observation_dim', 'goal_frame_dim',
            'history_length', 'future_frames', 'latent_dim', 'world_window',
            'imagination_horizon', 'world_batch_size', 'world_batches_per_update',
            'imagination_batch_size', 'imagination_microbatch', 'replay_steps_per_env',
            'critic_epochs', 'critic_minibatches', 'lr_decay_iterations', 'save_interval')
        for name in positive_ints:
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f'{name} must be a positive integer')
        if self.context_dim < 0:
            raise ValueError('context_dim must be nonnegative')
        if self.imagination_batch_size % self.imagination_microbatch:
            raise ValueError('imagination_batch_size must be divisible by imagination_microbatch')
        if self.imagination_batch_size % self.critic_minibatches:
            raise ValueError('imagination_batch_size must be divisible by critic_minibatches')
        for name in ('learning_rate', 'gamma', 'gae_lambda', 'update_to_data_ratio',
                     'lr_final_fraction', 'max_grad_norm', 'policy_element_clip'):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if not 0 < self.gamma < 1 or not 0 < self.gae_lambda <= 1:
            raise ValueError('gamma/gae_lambda are outside valid ranges')
        for name in ('world_hidden', 'motion_encoder_hidden', 'decoder_hidden',
                     'critic_hidden', 'head_hidden'):
            if not getattr(self, name) or any(type(v) is not int or v < 1 for v in getattr(self, name)):
                raise ValueError(f'{name} must contain positive widths')
        return self

    def to_dict(self):
        """写入 checkpoint/部署清单的 JSON 友好副本。"""
        return asdict(self)
