"""文件、关节顺序和续训契约：只依赖标准库，不创建 GPU 环境。

同一台 G1 在 CSV、MuJoCo 和 SDK 中可以有不同数组顺序，必须通过名字
建立映射。模型选择也按数字编号，不能把 model_99 排在 model_100 后面。
"""
import hashlib
import json
import math
import re
from pathlib import Path

# 名字来自参考脚本明确列出的 CSV 顺序；适用本空间支持的 29-DoF G1。
# 若换为不同关节数/不同硬件版本，必须重新建立对应关系，不能截断数组。
SDK_JOINTS = (
    'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint',
    'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint',
    'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint',
    'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint',
    'waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint',
    'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint',
    'left_elbow_joint', 'left_wrist_roll_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint',
    'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint',
    'right_elbow_joint', 'right_wrist_roll_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint',
)


def sha256(path):
    """分块计算文件哈希；用于动作数据、profile、checkpoint 和导出包溯源。"""
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value):
    """把 Path/NumPy/Tensor 等转为 JSON 值；非有限浮点数显式转成文字。

    这样不会输出 JSON 标准不允许的裸 NaN/Infinity。出现非有限指标仍应
    追查源头；此函数只保证文件可读，不把异常数值伪装成有效零值。
    """
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if hasattr(value, 'tolist'):
        return json_safe(value.tolist())
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, payload):
    """创建父目录并保存 UTF-8 JSON；所有输出都由调用方限定在工作空间。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def latest_checkpoint(run_dir):
    """在一个运行目录中选择最大数字 model_N.pt，排除 policy.pt 等导出物。"""
    run_dir = Path(run_dir).expanduser().resolve()
    candidates = []
    for path in run_dir.glob('model_*.pt'):
        match = re.fullmatch(r'model_(\d+)\.pt', path.name)
        if match and path.is_file():
            candidates.append((int(match[1]), path))
    if not candidates:
        raise FileNotFoundError(f'No model_<iteration>.pt in {run_dir}')
    return max(candidates, key=lambda item: item[0])[1]


def resolve_checkpoint(value, run_root):
    """支持 latest、运行目录和具体模型路径；latest 不代表验证分数最好。

    run_root 只传正式 outputs/runs，benchmark 的新时间戳不会抢走 latest。
    仍应在混合多个任务时显式指定运行目录，确保回放的是预期实验。
    """
    if value == 'latest':
        runs = [p for p in Path(run_root).glob('*') if p.is_dir() and list(p.glob('model_*.pt'))]
        if not runs:
            raise FileNotFoundError(f'No trained runs in {run_root}')
        return latest_checkpoint(max(runs, key=lambda p: p.name))
    path = Path(value).expanduser().resolve()
    if path.is_dir():
        return latest_checkpoint(path)
    if not path.is_file() or not re.fullmatch(r'model_\d+\.pt', path.name):
        raise FileNotFoundError(f'Expected own training checkpoint or run directory: {path}')
    return path


def joint_permutation(source, target):
    """返回 target 每个名字在 source 数组中的索引，要求无重复且集合相同。

    target_values[i] = source_values[permutation[i]]。不能只凭两个数组长度
    相等就认为可用；缺一个腕关节也会使整套动作错位。
    """
    if len(set(source)) != len(source) or len(set(target)) != len(target):
        raise ValueError('Duplicate joint names')
    if set(source) != set(target):
        raise ValueError(f'Joint sets differ: source-only={set(source)-set(target)}, target-only={set(target)-set(source)}')
    return [source.index(name) for name in target]


def validate_resume(saved, current):
    """拒绝无记录地更换任务、profile、动作、版本或控制频率。

    环境数允许通过命令行覆盖；其变化会改变 batch 和训练行为，但不改变
    模型输入输出维度。profile 文件本身的任何编辑仍会改变哈希并触发检查。
    """
    for key in ('task', 'profile_sha256', 'agent_overrides', 'motion_sha256', 'mjlab_version', 'rsl_rl_version', 'control_dt',
                'recipe', 'recipe_sha256', 'amp_source_sha256'):
        if saved.get(key) != current.get(key):
            raise ValueError(f'Resume contract differs: {key}. Start a separate experiment instead.')
    # 旧模型没有配方字段，继续兼容；新配方还核对生效后的学习参数，避免
    # 同一 profile 下换网络、采样长度或 PPO 参数而没有被发现。
    if (saved.get('recipe') or saved.get('amp_source_sha256')) and saved.get('effective_agent_sha256') != current.get('effective_agent_sha256'):
        raise ValueError('Resume contract differs: effective_agent_sha256. Start a separate experiment instead.')


def choose_parallelism(rows, headroom_mib=1024, ram_headroom_gib=2):
    """只在计时有效、成功且资源有余量的候选中选吞吐最合适的一档。

    5% 以内视为近似速度，优先较小并行数。该启发式只根据短测结果选择，
    不等于长期温度、桌面响应性或最终策略质量的最优解。
    """
    viable = [r for r in rows if r.get('status') == 'passed'
              and r.get('timing_valid') is True
              and r.get('median_fps', 0) > 0
              and (r.get('min_free_vram_mib') or 0) >= headroom_mib
              and (r.get('min_available_ram_gib') or 0) >= ram_headroom_gib]
    if not viable:
        return None
    best = max(r['median_fps'] for r in viable)
    # 达到最佳中位吞吐 95% 就足够，避免为极小加速占用更多共享资源。
    return min((r for r in viable if r['median_fps'] >= .95 * best), key=lambda r: r['num_envs'])['num_envs']
