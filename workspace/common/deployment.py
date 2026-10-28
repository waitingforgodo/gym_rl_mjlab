"""仅加载导出策略的 mjlab 仿真部署实现。

入口：scripts/deploy/deploy_sim.py。
依次校验文件 → 恢复 play 配置 → 创建环境 → 加载 JIT → 推理施加动作。
这一过程不创建 PPO runner，也不读取训练 checkpoint。"""
import json
from pathlib import Path

from .contract import sha256, validate_resume, write_json
from .inference import evaluate, view
from .runtime import initialize, build_configs


def deploy_sim(root, args):
    """校验包哈希后只加载导出的 JIT，使用同一后端组装观测并做闭环回归。

    不加载训练 checkpoint，不连接真机；该步骤验证的是导出后接口一致性，
    不是第二物理引擎 sim2sim，也不等同于真机动力学/保护机制验收。
    """
    import torch
    # 1. bundle 是 export 程序打印的目录，不是 model_N.pt 或单独 ONNX 文件。
    bundle = Path(args.bundle).expanduser().resolve()
    manifest = json.loads((bundle / 'manifest.json').read_text())
    hashes = json.loads((bundle / 'hashes.json').read_text())
    for name, expected in hashes.items():
        if sha256(bundle / name) != expected:
            raise ValueError(f'Bundle integrity differs: {name}')
    initialize(root)
    # 2. 跟踪任务从包内读参考轨迹；速度任务没有 motion.npz。
    motion = str(bundle / 'motion.npz') if manifest.get('motion_file') else None
    _, cfg, agent, current = build_configs(
        root, manifest['profile'], 1, motion, play=True, seed=manifest['seed'],
        profile_spec=manifest.get('profile_spec'),
        profile_sha256=manifest.get('profile_sha256'))
    validate_resume(manifest, current)
    # 这里只构造环境，不构造 PPO runner，部署测试不依赖训练期优化器。
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device=args.device), clip_actions=agent.clip_actions)
    try:
        # 3. 只加载可推理的 TorchScript；不构造 PPO、critic 或优化器。
        model = torch.jit.load(str(bundle / 'policy_jit.pt'), map_location=args.device).eval()
        def policy(obs):
            """按导出清单顺序拼接观测组，归一化由 JIT 模型内部完成。"""
            return model(torch.cat([obs[g] for g in manifest['actor_observation_groups']], dim=-1))
        if args.headless:
            # 4. 共用定量评估器：组装观测 → JIT 动作 → 裁剪 → 环境物理子步。
            result = evaluate(env, policy, args.seconds)
            write_json(bundle / 'simulation_result.json', result)
            return result
        # 交互模式下同样用这个 JIT policy，viewer 负责推理/仿真循环。
        view(env, policy, args.viewer, args.port)
    finally:
        env.close()
