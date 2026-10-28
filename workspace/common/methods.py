"""方法选择与产物定位；按清单筛选模型，避免跨算法误用 latest。"""
import json
from pathlib import Path
from .contract import latest_checkpoint, resolve_checkpoint
from .profiles import method_of

DEFAULT_PROFILES = dict(amp='amp_dance_4096', beyondmimic='tracking_4096', velocity='velocity_flat_4096')


def require_method(method, manifest):
    actual = method_of(manifest)
    if actual != method:
        raise ValueError(f'{method} interface received a {actual} task/checkpoint')


def resolve_model(root, value='latest', method=None):
    """搜索新的分方法目录，也兼容原 outputs/runs；不搜索 benchmark。"""
    root = Path(root)
    if str(value) != 'latest':
        path = resolve_checkpoint(str(value), root/'outputs/runs')
        if method:
            manifest = path.parent/'run_manifest.json'
            if not manifest.exists():
                raise ValueError('Method-specific interfaces require a run_manifest.json')
            require_method(method, json.loads(manifest.read_text()))
        return path
    folders = [root/'outputs/runs', *(root/'outputs'/m/'runs' for m in DEFAULT_PROFILES)]
    candidates = []
    for folder in folders:
        for run in folder.glob('*'):
            if not run.is_dir() or not list(run.glob('model_*.pt')):
                continue
            manifest = run/'run_manifest.json'
            if method and (not manifest.exists() or method_of(json.loads(manifest.read_text())) != method):
                continue
            candidates.append(latest_checkpoint(run))
    if not candidates:
        raise FileNotFoundError(f'No trained checkpoint for {method or "PPO methods"}; run train first')
    return max(candidates, key=lambda p: p.stat().st_mtime_ns)


def output_dir(root, method, kind):
    return Path(root)/'outputs'/method/kind
