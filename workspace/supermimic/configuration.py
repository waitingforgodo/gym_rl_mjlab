"""SuperMimic 参数、数据校验和来源记录；帮助只依赖标准库。"""
import argparse
from dataclasses import fields, is_dataclass
import hashlib
import importlib.metadata
import json
from pathlib import Path
from .paths import ROOT, UPSTREAM


def positive(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError('must be positive')
    return result


def train_parser():
    p = argparse.ArgumentParser(description='Train vendored SuperMimic-Reproduced in the existing runtime')
    p.add_argument('--config', type=Path, help='Partial upstream TrainConfig JSON; unknown fields are rejected')
    data = p.add_mutually_exclusive_group()
    data.add_argument('--motion', type=Path, help='One 50 Hz NPZ; requires --allow-config-changes')
    data.add_argument('--npz-dir', type=Path, help='50 Hz NPZ directory; default pinned 40-clip LAFAN1 dataset')
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--strict-upstream', dest='strict_upstream', action='store_true', default=True,
                      help='Require pinned upstream hyperparameters and full LAFAN1 data (default)')
    mode.add_argument('--allow-config-changes', dest='strict_upstream', action='store_false',
                      help='Explicit custom experiment; save all deviations from upstream defaults')
    p.add_argument('--clips', nargs='+', help='Clip stems within --npz-dir')
    p.add_argument('--envs', type=positive, help='Parallel simulation environments; upstream default 4096')
    p.add_argument('--iterations', type=positive, help='TOTAL vectorized env steps, not optimizer updates; default 320000')
    p.add_argument('--name', help='New run directory name; default timestamp')
    p.add_argument('--seed', type=int)
    p.add_argument('--device')
    p.add_argument('--resume', type=Path, help='Upstream checkpoint only; restores its config before explicit overrides')
    p.add_argument('--replay-capacity', type=positive)
    p.add_argument('--world-batch-size', type=positive)
    p.add_argument('--world-batches-per-update', type=positive)
    p.add_argument('--policy-batch-size', type=positive)
    p.add_argument('--horizon', type=positive)
    p.add_argument('--save-interval', type=positive)
    p.add_argument('--log-interval', type=positive)
    p.add_argument('--no-compile', action='store_true', help='Run eager PyTorch; useful for compatibility smoke tests')
    p.add_argument('--dry-run', action='store_true', help='Validate and print config, files, versions; no simulation')
    return p


def merge_config(cfg, values):
    """递归覆盖上游 dataclass；未知字段报错，避免拼写错误悄悄改变实验。"""
    allowed = {f.name for f in fields(cfg)}
    for key, value in values.items():
        if key not in allowed:
            raise ValueError(f'Unknown {type(cfg).__name__} field: {key}')
        old = getattr(cfg, key)
        if is_dataclass(old):
            if not isinstance(value, dict):
                raise ValueError(f'{key} must be an object')
            merge_config(old, value)
        else:
            if isinstance(old, tuple) and isinstance(value, list):
                value = tuple(value)
            setattr(cfg, key, value)
    return cfg


def data_files(cfg):
    """固定动作列表顺序，并验证上游不会自动纠正的关节顺序和身体顺序。"""
    from supermimic.envs.mdp.motion_library import discover_npz
    if cfg.data.library is not None:
        path = Path(cfg.data.library).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return [path]
    files = discover_npz(Path(cfg.data.npz_dir), names=cfg.data.clips,
                         limit=cfg.data.limit, manifest=cfg.data.manifest)
    if not files:
        raise FileNotFoundError(f'No clips found under {cfg.data.npz_dir}')
    from supermimic.data.motion_convert import G1_JOINT_NAMES
    from supermimic.envs.tracking_env_cfg import TRACKED_BODIES
    from workspace.common.motion import inspect
    body_names = None
    for path in files:
        report = inspect(path, joint_names=G1_JOINT_NAMES, expected_fps=50.)
        names = report['body_names']
        if not set(TRACKED_BODIES) <= set(names):
            raise ValueError(f'Missing tracked body names: {path}')
        # 上游 full library 使用首个文件的 body 下标处理所有片段。
        if body_names is not None and names != body_names:
            raise ValueError(f'Mixed body order across motion clips: {path}')
        body_names = names
    return [p.resolve() for p in files]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def provenance(files):
    """记录实际依赖和源码哈希，区分『上游算法』与『上游锁定运行环境』。"""
    info = json.loads((UPSTREAM / 'SOURCE_INFO.json').read_text())
    changed = [name for name, expected in info['files'].items()
               if not (UPSTREAM / name).is_file() or digest(UPSTREAM / name) != expected]
    if changed:
        raise ValueError(f'Vendored upstream source differs from pinned commit: {changed}')
    versions = {p: importlib.metadata.version(p) for p in
                ('torch', 'mjlab', 'mujoco', 'mujoco-warp', 'warp-lang')}
    import tomllib
    packages = tomllib.loads((UPSTREAM / 'uv.lock').read_text())['package']
    locked = {name: sorted({p['version'] for p in packages if p['name'] == name}) for name in versions}
    differences = {name: dict(installed=version, upstream_locked=locked[name])
                   for name, version in versions.items() if version not in locked[name]}
    from .runtime import configure_backend_compatibility
    return dict(implementation='SuperMimic-Reproduced', upstream_repository=info['repository'],
                upstream_commit=info['commit'], runtime_versions=versions,
                backend_compatibility=configure_backend_compatibility(),
                upstream_lock_environment=False,
                runtime_lock_differences=differences,
                upstream_mjlab_source_commit='8bb2bc9e187797689ec82c78c35019afd2fd0376',
                adapter_sha256=hashlib.sha256(''.join(digest(p) for p in sorted(Path(__file__).parent.glob('*.py'))).encode()).hexdigest(),
                data=[dict(path=str(p), sha256=digest(p)) for p in files])
