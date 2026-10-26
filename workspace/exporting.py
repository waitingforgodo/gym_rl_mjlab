"""策略导出和部署接口清单的实际实现。

入口：scripts/reinforcement_learning/rsl_rl/export.py。
导出的是 actor 和已学习的归一化器；训练 checkpoint 仍用于续训。
TorchScript 做真实观测数值对比，ONNX 做结构检查，二者不能混称。"""
import math
from datetime import datetime

from .contract import SDK_JOINTS, joint_permutation, sha256, write_json
from .inference import load_for_inference


def export_bundle(root, args, checkpoint):
    """导出带接口清单的仿真部署包，先做 JIT 数值一致性再做 ONNX 结构检查。

    归一化器随网络导出。SDK 映射按实际名字构造，PD 是自然关节顺序，
    action scale/offset 是动作目标顺序，两者不能默认当作同一数组顺序。
    """
    import torch
    import onnx
    from mjlab.rl.exporter_utils import get_base_metadata, attach_metadata_to_onnx
    env, runner, policy, manifest = load_for_inference(root, args, checkpoint)
    destination = root / 'outputs' / 'deploy' / f'{checkpoint.parent.name}_{checkpoint.stem}_{datetime.now():%H%M%S_%f}'
    destination.mkdir(parents=True, exist_ok=False)
    try:
        obs = env.get_observations()
        # 从真实仿真采集 32 个输入/输出样本，避免仅用零输入掩盖导出问题。
        actor = runner.alg.get_policy()
        # 以实际 actor 声明的组顺序拼接，不能假设所有任务都只有同一观测向量。
        observation_groups = list(actor.obs_groups)
        samples, expected = [], []
        with torch.inference_mode():
            for _ in range(32):
                action = policy(obs)
                samples.append(torch.cat([obs[g] for g in observation_groups], dim=-1).cpu())
                expected.append(action.cpu())
                obs, _, _, _ = env.step(action)
        # JIT 中包含观测归一化器。外部不能再套一层相同归一化。
        runner.export_policy_to_jit(str(destination), 'policy_jit.pt')
        jit = torch.jit.load(str(destination / 'policy_jit.pt'), map_location='cpu').eval()
        with torch.inference_mode():
            predicted = jit(torch.cat(samples))
        target = torch.cat(expected)
        max_error = float((predicted-target).abs().max())
        if not torch.allclose(predicted, target, atol=1e-4, rtol=1e-4):
            raise RuntimeError(f'TorchScript parity failed: {max_error}')
        # tracking 的专属 runner 还会导出 ONNX 时间索引/参考轨迹接口。
        runner.export_policy_to_onnx(str(destination), 'policy.onnx')
        base = env.unwrapped
        metadata = get_base_metadata(base, checkpoint.parent.name)
        attach_metadata_to_onnx(str(destination / 'policy.onnx'), metadata)
        onnx_model = onnx.load(str(destination / 'policy.onnx'))
        onnx.checker.check_model(onnx_model)
        action_term = base.action_manager.get_term('joint_pos')
        def first(value):
            """提取第 0 个环境的参数；标量与每关节 Tensor 都转换成 JSON 值。"""
            if torch.is_tensor(value):
                return value[0].detach().cpu().tolist() if value.ndim > 1 else value.detach().cpu().tolist()
            return value
        robot = base.scene['robot']
        joint_names = list(robot.joint_names)
        action_names = list(action_term.target_names)
        observation_terms = []
        cursor = 0
        # start:stop 是拼接后向量中的半开区间；保留函数来源供独立控制器核对。
        for group in observation_groups:
            for name, dims in zip(base.observation_manager.active_terms[group], base.observation_manager.group_obs_term_dim[group]):
                size = math.prod(dims)
                term_cfg = base.observation_manager.get_term_cfg(group, name)
                observation_terms.append(dict(group=group, name=name, start=cursor, stop=cursor+size,
                                              shape=dims, function=f'{term_cfg.func.__module__}.{term_cfg.func.__qualname__}'))
                cursor += size
        # 三套顺序必须分别记录：策略动作顺序、模型自然关节顺序、SDK 电机顺序。
        # action_to_sdk_index[i] 是第 i 个动作关节在 SDK 数组中的目标下标。
        manifest.update(checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint),
                        base_metadata=metadata, actor_observation_groups=observation_groups,
                        observation_terms=observation_terms, observation_dim=cursor,
                        observation_pipeline='compute -> noise (off during play) -> clip -> scale -> delay -> history; JIT includes learned normalization',
                        action_joint_names=action_names, action_scale=first(action_term.scale), action_offset=first(action_term.offset),
                        processed_target_clip=action_term.cfg.clip, wrapper_clip_actions=runner.cfg['clip_actions'],
                        action_to_sdk_index=joint_permutation(list(SDK_JOINTS), action_names),
                        natural_to_sdk_index=joint_permutation(list(SDK_JOINTS), joint_names),
                        sdk_joint_names=SDK_JOINTS, jit_max_abs_error=max_error,
                        onnx_check='structural only; ONNX Runtime execution has not been validated',
                        onnx_inputs=[i.name for i in onnx_model.graph.input], onnx_outputs=[i.name for i in onnx_model.graph.output])
        if manifest['motion_file']:
            # 部署包自带 NPZ；移动整个包后不再依赖训练时的动作绝对路径。
            import shutil
            shutil.copyfile(manifest['motion_file'], destination / 'motion.npz')
            manifest['motion_file'] = str(destination / 'motion.npz')
            if 'motion' in base.command_manager.active_terms:
                manifest['tracking_anchor_body'] = base.command_manager.get_term('motion').cfg.anchor_body_name
            # AMP 的参考数据仅供本后端的参考初态复位；JIT/ONNX 只有 actor，
            # 不导出判别器，不需要 time_step 或逐帧参考目标作为策略输入。
        write_json(destination / 'manifest.json', manifest)
        write_json(destination / 'hardware_template.json', dict(enabled=False, robot='Unitree G1 29 DoF',
                    source='manifest.json', network_interface=None, dds_domain=None,
                    control_dt=base.step_dt,
                    requires_base_linear_velocity_estimate=any(t['name'] in ('base_lin_vel', 'imu_lin_vel') for t in observation_terms),
                    requires_motion_anchor_position=any(t['name'] == 'motion_anchor_pos_b' for t in observation_terms),
                    policy_state_machine=['damping', 'stand', 'policy', 'damping'],
                    missing=['SDK communication adapter', 'observation assembly parity on hardware',
                             'state estimator and sensor latency validation', 'hardware-specific PD/torque/joint limits',
                             'watchdog and emergency stop integration'],
                    deployment_status='simulation only; configuration is not hardware-certified'))
        # 所有产物落盘后最后写哈希索引。部署记录 simulation_result.json 后写，
        # 不属于静态输入清单，所以不会影响部署包重复运行。
        write_json(destination / 'hashes.json', {p.name: sha256(p) for p in destination.iterdir() if p.is_file()})
        print(f'DEPLOY_BUNDLE={destination}', flush=True)
        return destination
    finally:
        env.close()
