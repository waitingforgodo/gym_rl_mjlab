"""公共运行时：版本初始化、任务配置、checkpoint 上下文和环境构造。

实际业务已拆到 training.py、inference.py、exporting.py、deployment.py。
原 runtime.* 函数保留延迟转发，兼容现有调用；新程序直接导入对应模块。
"""
import importlib.metadata
import json
import os
import re
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from .contract import sha256
from .configuration import apply_agent_overrides, effective_agent_sha256, validate_ppo_config
from .recipes import apply_recipe


def initialize(root):
    """核对固定版本并设置缓存/线程，然后导入任务完成 registry 注册。

    Warp 的缓存位置必须在创建环境/编译 kernel 前设置。这里不执行 pip，
    发现版本不匹配直接报错，避免在用户正在使用的环境里隐式升级。
    """
    if importlib.metadata.version('mjlab') != '1.6.0':
        raise RuntimeError('This workspace targets the existing mjlab 1.6.0 runtime; do not upgrade it in place.')
    if importlib.metadata.version('rsl-rl-lib') != '5.4.2':
        raise RuntimeError('Expected the mjlab 1.6.0 dependency rsl-rl-lib 5.4.2; this bridge uses its model API.')
    import warp as wp
    import torch
    wp.config.kernel_cache_dir = str(root / '.cache' / 'warp')
    torch.set_num_threads(int(os.environ.get('G1_CPU_THREADS', '2')))
    from mjlab.utils.torch import configure_torch_backends
    configure_torch_backends()
    import mjlab.tasks  # noqa: F401
    return torch


def build_configs(root, profile_name, num_envs=None, motion=None, play=False, seed=42):
    """从 profile 与官方 registry 组合配置，返回配置及可核验的训练清单。

    registry 的 load_* 返回副本，故修改 num_envs 等字段不会污染其他任务。
    profile 只存少量实验选择；完整最终配置在训练时另存 params/*.yaml。
    play=True 会使用官方推理环境，可能与训练的命令范围和随机化不同。
    """
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
    profile_file = root / 'profiles' / f'{profile_name}.json'
    profile = json.loads(profile_file.read_text(encoding='utf-8'))
    cfg, agent = load_env_cfg(profile['task'], play=play), load_rl_cfg(profile['task'])
    cfg.scene.num_envs = num_envs or (1 if play else profile['num_envs'])
    cfg.seed = agent.seed = seed
    agent.logger, agent.upload_model, agent.save_interval = 'tensorboard', False, 100
    # 支持 actor.distribution_cfg.std_type 等字典路径，并在初始化 GPU 前
    # 检查类型、采样批量整除和优化参数，避免无效配置长时间运行。
    apply_agent_overrides(agent, profile.get('agent_overrides', {}))
    validate_ppo_config(agent, cfg.scene.num_envs)
    recipe = profile.get('recipe')
    apply_recipe(cfg, agent, recipe, play=play)
    manifest = dict(schema=1, profile=profile_name, profile_sha256=sha256(profile_file), task=profile['task'],
                    agent_overrides=profile.get('agent_overrides', {}), motion_file=None, motion_sha256=None,
                    seed=seed, num_envs=cfg.scene.num_envs, control_dt=cfg.sim.mujoco.timestep*cfg.decimation,
                    mjlab_version=importlib.metadata.version('mjlab'), rsl_rl_version=importlib.metadata.version('rsl-rl-lib'),
                    recipe=recipe, recipe_sha256=sha256(root / 'workspace/recipes.py') if recipe else None,
                    effective_agent_sha256=effective_agent_sha256(agent),
                    rollout_steps=agent.num_steps_per_env,
                    samples_per_update=cfg.scene.num_envs * agent.num_steps_per_env,
                    minibatch_size=cfg.scene.num_envs * agent.num_steps_per_env // agent.algorithm.num_mini_batches)
    if 'motion' in cfg.commands:
        # MotionLoader 按数组下标取数，不会自动按名字重排，也不会按 fps 调速。
        # 所以在创建 GPU 环境之前检查本地 NPZ 的顺序、形状和采样频率。
        if not motion:
            raise ValueError('Tracking needs --motion motions/processed/dance_102.npz')
        from mjlab.entity import Entity
        from .motion import inspect
        path = Path(motion).expanduser().resolve()
        robot = Entity(cfg.scene.entities['robot'])
        inspect(path, robot.joint_names, robot.body_names, 1 / manifest['control_dt'])
        cfg.commands['motion'].motion_file = str(path)
        manifest.update(motion_file=str(path), motion_sha256=sha256(path))
    elif motion:
        raise ValueError('Velocity tracking does not consume a reference motion')
    return profile, cfg, agent, manifest


def checkpoint_context(root, checkpoint, profile=None, motion=None, legacy_velocity=False):
    """读取模型对应 profile/动作，旧模型则要求显式接受官方 velocity 假设。

    维度一致不代表观测语义、奖励与物理参数一致；旧目录缺少本清单时，
    只能做严格参数形状加载，不能声称已证明完整配置一致。
    """
    saved_file = checkpoint.parent / 'run_manifest.json'
    if saved_file.exists():
        saved = json.loads(saved_file.read_text())
        return saved, profile or saved['profile'], motion or saved.get('motion_file')
    if not legacy_velocity or profile not in (None, 'velocity_flat'):
        raise ValueError('No workspace manifest. For your old g1_continue velocity model explicitly use --legacy-velocity.')
    print('Legacy velocity import: using the pinned official flat G1 contract; model shapes will be checked strictly.', flush=True)
    print('Old configuration provenance cannot be fully checked. Original run remains read-only.', flush=True)
    return None, 'velocity_flat', None


def make_env_runner(cfg, agent, task, device, log_dir=None):
    """创建 GPU 仿真、RSL 环境包装器和任务 runner，并负责初始化失败清理。

    wrapper 把环境五元组接口转换为 PPO 四元组接口，并处理动作裁剪。
    tracking 使用专属 runner，导出 ONNX 时才能保留时间索引与参考轨迹。
    """
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_runner_cls
    env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device=device), clip_actions=agent.clip_actions)
    try:
        cls = load_runner_cls(task) or MjlabOnPolicyRunner
        kwargs = {'registry_name': None} if 'motion' in cfg.commands else {}
        runner = cls(env, asdict(agent), str(log_dir) if log_dir else None, device=device, **kwargs)
        return env, runner
    except BaseException:
        env.close()
        raise


def new_run(root, label, benchmark=False):
    """创建唯一运行目录；正式训练与性能测试严格分开，避免 latest 选错。"""
    label = re.sub(r'[^a-zA-Z0-9_-]', '_', label)
    parent = root / 'outputs' / ('benchmarks/runs' if benchmark else 'runs')
    run = parent / f'{datetime.now():%Y%m%d_%H%M%S_%f}_{label}'
    run.mkdir(parents=True, exist_ok=False)
    return run


def train(root, args, checkpoint=None):
    """兼容旧 runtime.train 调用；实际实现见 training.py。"""
    from .training import train as implementation
    return implementation(root, args, checkpoint)


def load_for_inference(root, args, checkpoint, num_envs=1):
    """兼容旧 runtime.load_for_inference 调用；实际实现见 inference.py。"""
    from .inference import load_for_inference as implementation
    return implementation(root, args, checkpoint, num_envs)


def view(env, policy, viewer='viser', port=8080):
    """兼容旧 runtime.view 调用；实际实现见 inference.py。"""
    from .inference import view as implementation
    return implementation(env, policy, viewer, port)


def evaluate(env, policy, seconds=60):
    """兼容旧 runtime.evaluate 调用；实际实现见 inference.py。"""
    from .inference import evaluate as implementation
    return implementation(env, policy, seconds)


def export_bundle(root, args, checkpoint):
    """兼容旧 runtime.export_bundle 调用；实际实现见 exporting.py。"""
    from .exporting import export_bundle as implementation
    return implementation(root, args, checkpoint)


def deploy_sim(root, args):
    """兼容旧 runtime.deploy_sim 调用；实际实现见 deployment.py。"""
    from .deployment import deploy_sim as implementation
    return implementation(root, args)
