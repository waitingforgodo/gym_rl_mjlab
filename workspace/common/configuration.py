"""训练参数的严格覆盖、采样批量检查和可复现摘要；不导入仿真或 GPU。

JSON 只负责声明数值，中文设计依据放在 profile 的 _notes 和训练方案文档。
覆盖失败必须在创建环境前报错，避免拼错字段后悄悄按默认值训练数小时。
"""
from dataclasses import asdict
import hashlib
import json
import math


def apply_agent_overrides(agent, overrides):
    """支持 dataclass 属性和字典路径，例如 actor.distribution_cfg.std_type。

    旧 profile 的 algorithm.entropy_coef 用法保持兼容。网络维数在 JSON
    中是列表，而官方配置用 tuple，覆盖时保留原容器类型。
    """
    for dotted, value in overrides.items():
        target = agent
        parts = dotted.split('.')
        for part in parts[:-1]:
            if isinstance(target, dict):
                if part not in target:
                    raise ValueError(f'Unknown agent parameter: {dotted}')
                target = target[part]
            elif hasattr(target, part) and not part.startswith('_'):
                target = getattr(target, part)
            else:
                raise ValueError(f'Unknown agent parameter: {dotted}')
        key = parts[-1]
        exists = key in target if isinstance(target, dict) else hasattr(target, key)
        if not exists or key.startswith('_'):
            raise ValueError(f'Unknown agent parameter: {dotted}')
        old = target[key] if isinstance(target, dict) else getattr(target, key)
        if isinstance(old, bool) and not isinstance(value, bool):
            raise ValueError(f'{dotted} must be a boolean')
        if isinstance(old, int) and not isinstance(old, bool) and type(value) is not int:
            raise ValueError(f'{dotted} must be an integer')
        if isinstance(old, float):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f'{dotted} must be a finite number')
            value = float(value)
        if isinstance(old, str) and not isinstance(value, str):
            raise ValueError(f'{dotted} must be a string')
        if isinstance(old, tuple):
            if not isinstance(value, (tuple, list)):
                raise ValueError(f'{dotted} must be a list')
            value = tuple(value)
        if isinstance(target, dict):
            target[key] = value
        else:
            setattr(target, key, value)


def validate_ppo_config(agent, num_envs):
    """检查批量、概率和优化参数，拒绝会丢采样或生成非有限优化的配置。"""
    algorithm = agent.algorithm
    for name, value in (('num_envs', num_envs), ('num_steps_per_env', agent.num_steps_per_env),
                        ('num_learning_epochs', algorithm.num_learning_epochs),
                        ('num_mini_batches', algorithm.num_mini_batches), ('save_interval', agent.save_interval)):
        if type(value) is not int or value < 1:
            raise ValueError(f'{name} must be a positive integer')
    batch = num_envs * agent.num_steps_per_env
    if batch % algorithm.num_mini_batches:
        raise ValueError(f'Rollout batch {batch} must be divisible by num_mini_batches={algorithm.num_mini_batches}')
    if algorithm.schedule not in ('adaptive', 'fixed'):
        raise ValueError('algorithm.schedule must be adaptive or fixed')
    if algorithm.optimizer not in ('adam', 'adamw', 'sgd', 'rmsprop'):
        raise ValueError('algorithm.optimizer is not supported by the pinned RSL-RL version')
    for name in ('learning_rate', 'desired_kl', 'max_grad_norm'):
        value = getattr(algorithm, name)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'algorithm.{name} must be finite and positive')
    for name in ('gamma', 'lam', 'clip_param'):
        value = getattr(algorithm, name)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value <= 1:
            raise ValueError(f'algorithm.{name} must be in (0, 1]')
    if not math.isfinite(algorithm.entropy_coef) or algorithm.entropy_coef < 0:
        raise ValueError('algorithm.entropy_coef must be finite and nonnegative')
    for name in ('actor', 'critic'):
        model = getattr(agent, name)
        if not model.hidden_dims or any(type(n) is not int or n < 1 for n in model.hidden_dims):
            raise ValueError(f'{name}.hidden_dims must contain positive integers')
    distribution = agent.actor.distribution_cfg
    if distribution and distribution.get('class_name') == 'GaussianDistribution':
        if distribution.get('std_type', 'scalar') not in ('scalar', 'log'):
            raise ValueError('Gaussian std_type must be scalar or log')
        initial_std = distribution.get('init_std', 1.)
        if not math.isfinite(initial_std) or initial_std <= 0:
            raise ValueError('Gaussian init_std must be finite and positive')


def effective_agent_sha256(agent):
    """只对影响学习和输入输出的实际参数取哈希。

    不包含本次更新预算、实验名和随机种子：续训可以增加预算、换名称或
    显式设置种子；网络、PPO、观测组、动作裁剪和采样长度则必须可核验。
    """
    cfg = asdict(agent)
    keys = ('actor', 'critic', 'algorithm', 'obs_groups', 'clip_actions', 'num_steps_per_env')
    payload = json.dumps({key: cfg[key] for key in keys}, sort_keys=True, allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def inherited_run_options(args, saved=None):
    """续训默认继承原实验的并行数和种子；从零训练仍采用 profile/42。"""
    saved = saved or {}
    envs = args.envs if args.envs is not None else saved.get('num_envs')
    seed = args.seed if args.seed is not None else saved.get('seed', 42)
    return envs, seed
