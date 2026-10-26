"""把 robot_lab 的 AMP NPZ 转为当前 mjlab G1 的标准专家轨迹。

原数据的 14 刚体和 Isaac 布局不直接沿用：只读取已命名的 29 关节与
pelvis 根姿态，随后复用本项目 CPU 正运动学重建全部刚体和速度。
"""
from pathlib import Path

import numpy as np

from ..contract import SDK_JOINTS, joint_permutation, sha256, write_json
from ..motion import normalize_quat, prepare, inspect


def source_to_csv_arrays(source):
    """无 pickle 加载专家数据，按名字映射并输出 xyz/xyzw/SDK 关节列。"""
    with np.load(source, allow_pickle=False) as data:
        required = ('fps', 'dof_names', 'body_names', 'dof_positions', 'body_positions', 'body_rotations')
        for key in required:
            if key not in data:
                raise ValueError(f'AMP source is missing {key}')
        fps = float(data['fps'])
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError('AMP source fps must be positive')
        joints, bodies = data['dof_names'].tolist(), data['body_names'].tolist()
        if len(set(bodies)) != len(bodies) or 'pelvis' not in bodies:
            raise ValueError('AMP source requires unique body names including pelvis')
        mapping = joint_permutation(joints, list(SDK_JOINTS))
        angles = data['dof_positions']
        positions, rotations = data['body_positions'], data['body_rotations']
        frames = len(angles)
        if frames < 3 or angles.shape != (frames, 29) or positions.shape != (frames, len(bodies), 3) or rotations.shape != (frames, len(bodies), 4):
            raise ValueError('AMP source has incompatible frame/joint/body shapes')
        root = bodies.index('pelvis')
        quat = normalize_quat(rotations[:, root])
        csv = np.concatenate((positions[:, root], quat[:, [1, 2, 3, 0]], angles[:, mapping]), -1)
        if not np.isfinite(csv).all():
            raise ValueError('AMP source contains non-finite root/joint data')
    return csv, fps


def prepare_amp_motion(root, source=None):
    """转换一次；已有结果校验来源后复用，绝不静默覆盖不同专家数据。"""
    root = Path(root)
    if source is None:
        matches = sorted((root / 'gym_rl').rglob('g1_dance1_subject2_30.npz'))
        if not matches:
            raise FileNotFoundError('Local AMP dance source not found; pass --source /path/to/motion.npz')
        source = matches[0]
    source = Path(source).expanduser().resolve()
    source_hash = sha256(source)
    target = root / 'motions/processed/amp_dance.npz'
    if target.exists():
        import json
        details = json.loads(target.with_suffix('.json').read_text())
        if details.get('amp_source_sha256') != source_hash:
            raise FileExistsError('Existing amp_dance uses a different source; preserve it and use a separate dataset name')
        result = inspect(target)
        if details['sha256'] != result['sha256']:
            raise ValueError('Existing AMP motion integrity differs')
        return details
    csv, fps = source_to_csv_arrays(source)
    raw = root / 'motions/raw/amp_dance.csv'
    raw.parent.mkdir(parents=True, exist_ok=True)
    if raw.exists():
        existing = np.loadtxt(raw, delimiter=',')
        if existing.shape != csv.shape or not np.allclose(existing, csv, atol=1e-10, rtol=1e-10):
            raise FileExistsError(f'Preserve existing raw AMP conversion: {raw}')
    else:
        np.savetxt(raw, csv, delimiter=',', fmt='%.17g')
    details = prepare(root, 'amp_dance', input_fps=fps)
    details.update(amp_source=str(source), amp_source_sha256=source_hash,
                   source_format='robot_lab_amp_npz_wxyz',
                   conversion='Named 29 joints + pelvis pose -> CPU G1 FK -> 50 Hz full-body NPZ')
    write_json(target.with_suffix('.json'), details)
    return details
