"""动作迁移：参考 CSV → 当前 G1 的控制频率、关节/刚体顺序和世界系速度。

CSV 每行 36 列：根 xyz、根 xyzw、SDK 顺序的 29 关节角。输出用当前
mjlab 模型做 CPU 正运动学，不能沿用 Isaac Lab 的刚体下标或旧 NPZ 顺序。
"""
import json
import shutil
from pathlib import Path

import numpy as np

from .contract import SDK_JOINTS, joint_permutation, sha256, write_json

REFERENCE_FILES = {
    'dance_102': 'G1_Take_102.bvh_60hz.csv',
    'gangnam_style': 'G1_gangnam_style_V01.bvh_60hz.csv',
}
REQUIRED = ('joint_pos', 'joint_vel', 'body_pos_w', 'body_quat_w',
            'body_lin_vel_w', 'body_ang_vel_w')


def scalar_float(value, name='value'):
    """Read an NPZ scalar while accepting the common one-element encoding."""
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f'{name} must contain exactly one value')
    result = float(array.reshape(-1)[0])
    if not np.isfinite(result):
        raise ValueError(f'{name} must be finite')
    return result


def normalize_quat(q):
    """归一化 wxyz 四元数，拒绝零长度与 NaN；支持任意批维度。"""
    q = np.asarray(q, dtype=np.float64).copy()
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(norm < 1e-8) or not np.isfinite(q).all():
        raise ValueError('Invalid quaternion')
    return q / norm


def continuous_quat(q):
    """沿时间轴统一相邻四元数符号，避免 q 与 -q 切换产生虚假转动。"""
    q = normalize_quat(q)
    for frame in range(1, len(q)):
        sign = np.where(np.sum(q[frame - 1] * q[frame], axis=-1) < 0, -1., 1.)
        q[frame] *= np.asarray(sign)[..., None]
    return q


def quat_multiply(a, b):
    """Hamilton 乘法，最后一维均为 wxyz；调用方负责先归一化。"""
    w1, x1, y1, z1 = np.moveaxis(a, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(b, -1, 0)
    return np.stack((w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2,
                     w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2), axis=-1)


def angular_velocity_world(q, fps):
    """利用 q_next * inverse(q_prev) 的旋转向量估算世界系角速度。

    左右乘顺序决定坐标系，不能交换。端点用一侧差分，中间平均两侧增量。
    小角度用 angle/sin(angle/2)→2 的极限，避免除零。
    """
    q = continuous_quat(q)
    inverse = q[:-1].copy()
    inverse[..., 1:] *= -1
    delta = quat_multiply(q[1:], inverse)  # q_next * inverse(q_prev): WORLD frame.
    delta *= np.where(delta[..., :1] < 0, -1., 1.)
    sine = np.linalg.norm(delta[..., 1:], axis=-1, keepdims=True)
    angle = 2 * np.arctan2(sine, np.clip(delta[..., :1], -1, 1))
    increments = delta[..., 1:] * np.divide(angle, sine, out=np.full_like(sine, 2.), where=sine > 1e-10) * fps
    result = np.empty(q.shape[:-1] + (3,), dtype=np.float64)
    result[0], result[-1] = increments[0], increments[-1]
    result[1:-1] = (increments[:-1] + increments[1:]) * .5
    return result


def resample_csv(data, input_fps, output_fps):
    """按真实秒数重采样：位置/关节线性插值，根朝向最短弧 SLERP。

    欧氏位置可线性插值；四元数直接逐元素插值后归一化仅用于非常接近的
    朝向，以避免 sin(theta) 接近零时数值不稳定。输出不超过源动作时长。
    """
    data = np.asarray(data, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] != 36 or len(data) < 3 or not np.isfinite(data).all():
        raise ValueError('CSV must contain at least 3 finite rows and exactly 36 columns (xyz, xyzw, 29 joints)')
    if input_fps <= 0 or output_fps <= 0:
        raise ValueError('Frame rates must be positive')
    old_t = np.arange(len(data)) / input_fps
    new_t = np.arange(int(np.floor(old_t[-1] * output_fps)) + 1) / output_fps
    if len(new_t) < 3:
        raise ValueError('Resampled motion is too short')
    root = np.stack([np.interp(new_t, old_t, data[:, k]) for k in range(3)], axis=-1)
    joints = np.stack([np.interp(new_t, old_t, data[:, k]) for k in range(7, 36)], axis=-1)
    q = continuous_quat(data[:, [6, 3, 4, 5]])  # xyzw -> wxyz.
    left = np.minimum(np.searchsorted(old_t, new_t, side='right') - 1, len(data) - 2)
    fraction = ((new_t - old_t[left]) * input_fps)[:, None]
    qa, qb = q[left], q[left + 1]
    dot = np.clip(np.sum(qa * qb, axis=-1, keepdims=True), -1, 1)
    angle = np.arccos(dot)
    sine = np.sin(angle)
    safe_sine = np.where(np.abs(sine) < 1e-7, 1., sine)
    spherical = (np.sin((1-fraction)*angle)*qa + np.sin(fraction*angle)*qb) / safe_sine
    linear = (1-fraction)*qa + fraction*qb
    quat = normalize_quat(np.where(np.abs(sine) < 1e-7, linear, spherical))
    return root, quat, joints


def import_reference(root, source):
    """只读搜索并复制指定两份 CSV，写来源路径、哈希与列顺序清单。

    同名重复文件按路径排序选第一份，实际选择以 reference_sources.json
    为准。已有目标与来源内容不一致时拒绝覆盖，不执行资料中的任何代码。
    """
    source = Path(source).expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    entries = {}
    for label, filename in REFERENCE_FILES.items():
        matches = sorted(source.rglob(filename))
        if not matches:
            raise FileNotFoundError(f'{filename} under {source}')
        original = matches[0]
        destination = root / 'motions' / 'raw' / f'{label}.csv'
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and sha256(destination) != sha256(original):
            raise FileExistsError(f'Existing motion differs; preserve it: {destination}')
        if not destination.exists():
            shutil.copyfile(original, destination)
        entries[label] = {'original': str(original), 'file': str(destination), 'sha256': sha256(destination),
                          'input_fps': 60, 'root_quaternion': 'xyzw', 'joint_names': SDK_JOINTS}
    write_json(root / 'motions' / 'reference_sources.json', entries)
    return entries


def prepare(root, label, input_fps=60):
    """CPU 正运动学转换，输出全关节/全刚体数组，供官方 MotionLoader 使用。

    频率由实际 tracking 配置的 timestep*decimation 推出。写 qpos 时按
    每个命名关节的地址映射，刚体位姿来自 mj_forward。该步骤没有积分
    接触动力学，转换成功不能证明动作可无穿地、无滑动地执行。
    """
    import mujoco
    from mjlab.asset_zoo.robots.unitree_g1.g1_constants import get_g1_robot_cfg
    from mjlab.entity import Entity
    from mjlab.tasks.registry import load_env_cfg
    import mjlab.tasks  # noqa: F401: populate registry

    source = root / 'motions' / 'raw' / f'{label}.csv'
    target = root / 'motions' / 'processed' / f'{label}.npz'
    if target.exists():
        raise FileExistsError(f'Preserve existing converted motion: {target}')
    cfg = load_env_cfg('Mjlab-Tracking-Flat-Unitree-G1')
    output_fps = 1 / (cfg.sim.mujoco.timestep * cfg.decimation)
    xyz, root_quat, sdk_joints = resample_csv(np.loadtxt(source, delimiter=','), input_fps, output_fps)
    robot = Entity(get_g1_robot_cfg())
    joint_names, body_names = list(robot.joint_names), list(robot.body_names)
    joints = sdk_joints[:, joint_permutation(list(SDK_JOINTS), joint_names)]
    model = robot.spec.compile()
    state = mujoco.MjData(model)
    free = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    if len(free) != 1:
        raise ValueError('Expected one floating base')
    root_address = int(model.jnt_qposadr[free[0]])
    # 模型中可能带实体前缀，匹配末段名字；无匹配或同名歧义均停止。
    def model_id(kind, name):
        """为独立编译后的模型查唯一关节/刚体 ID，避免硬编码数组地址。"""
        matches = [i for i in range(model.njnt if kind == mujoco.mjtObj.mjOBJ_JOINT else model.nbody)
                   if mujoco.mj_id2name(model, kind, i) is not None
                   if mujoco.mj_id2name(model, kind, i).split('/')[-1] == name]
        if len(matches) != 1:
            raise ValueError(f'Ambiguous/missing model name: {name}')
        return matches[0]
    joint_ids = [model_id(mujoco.mjtObj.mjOBJ_JOINT, n) for n in joint_names]
    addresses = model.jnt_qposadr[joint_ids]
    body_ids = [model_id(mujoco.mjtObj.mjOBJ_BODY, n) for n in body_names]
    positions = np.empty((len(xyz), len(body_ids), 3))
    orientations = np.empty((len(xyz), len(body_ids), 4))
    for frame in range(len(xyz)):
        state.qpos[root_address:root_address+3] = xyz[frame]
        state.qpos[root_address+3:root_address+7] = root_quat[frame]
        state.qpos[addresses] = joints[frame]
        mujoco.mj_forward(model, state)
        positions[frame] = state.xpos[body_ids]
        orientations[frame] = state.xquat[body_ids]
    orientations = continuous_quat(orientations)
    arrays = dict(fps=np.array(output_fps), joint_names=np.array(joint_names), body_names=np.array(body_names),
                  joint_pos=joints, joint_vel=np.gradient(joints, 1/output_fps, axis=0, edge_order=2),
                  body_pos_w=positions, body_quat_w=orientations,
                  body_lin_vel_w=np.gradient(positions, 1/output_fps, axis=0, edge_order=2),
                  body_ang_vel_w=angular_velocity_world(orientations, output_fps))
    arrays = {k: v.astype(np.float32) if k in REQUIRED else v for k, v in arrays.items()}
    limited = model.jnt_limited[joint_ids].astype(bool)
    ranges = model.jnt_range[joint_ids]
    violations = ((joints < ranges[:, 0] - .01) | (joints > ranges[:, 1] + .01)) & limited
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, **arrays)
    details = inspect(target, joint_names, body_names, output_fps)
    details.update(source=str(source), source_sha256=sha256(source), input_fps=input_fps,
                   joint_limit_violation_fraction=float(violations.mean()),
                   note='Kinematic conversion only; contact feasibility and dynamic tracking need evaluation.')
    write_json(target.with_suffix('.json'), details)
    return details


def inspect(path, joint_names=None, body_names=None, expected_fps=None):
    """不允许 pickle 地读取 NPZ，验证键、名字顺序、形状、有限值和单位四元数。

    提供期望名字/频率时必须完全对应当前模型；只有列数正确还不足以训练。
    返回可保存为 JSON 的概览，不创建仿真环境。
    """
    with np.load(path, allow_pickle=False) as data:
        for name in REQUIRED + ('fps', 'joint_names', 'body_names'):
            if name not in data:
                raise ValueError(f'{path}: missing {name}; convert CSV through prepare-motion')
        joints, bodies = data['joint_names'].tolist(), data['body_names'].tolist()
        if joint_names is not None and joints != list(joint_names):
            raise ValueError('Motion joint order differs from installed robot')
        if body_names is not None and bodies != list(body_names):
            raise ValueError('Motion body order differs from installed robot')
        fps, frames = scalar_float(data['fps'],'fps'), len(data['joint_pos'])
        if fps <= 0 or not np.isfinite(fps) or frames < 3:
            raise ValueError('Invalid motion frame rate or length')
        if expected_fps is not None and not np.isclose(fps, expected_fps, rtol=1e-6):
            raise ValueError(f'Motion fps={fps}, control fps={expected_fps}; reconvert motion')
        for name in REQUIRED:
            expected = (frames, len(joints)) if name.startswith('joint_') else (frames, len(bodies), 4 if name == 'body_quat_w' else 3)
            if data[name].shape != expected or not np.isfinite(data[name]).all():
                raise ValueError(f'Invalid shape/values: {name}, expected {expected}')
        if not np.allclose(np.linalg.norm(data['body_quat_w'], axis=-1), 1, atol=1e-4):
            raise ValueError('Body quaternions are not normalized')
    return {'path': str(Path(path).resolve()), 'sha256': sha256(path), 'frames': frames, 'fps': fps,
            'duration_seconds': (frames-1)/fps, 'joint_names': joints, 'body_names': bodies, 'quaternion': 'wxyz',
            'linear_velocity_frame': 'world', 'angular_velocity_frame': 'world'}
