"""上游 checkpoint 格式、动作哈希和本方法 latest 选择。"""
import json
from pathlib import Path
from .paths import RUNS
from .configuration import digest


def resolve_checkpoint(value):
    """latest 仅搜索此实现的专用训练目录。"""
    if str(value) == 'latest':
        candidates = list(RUNS.glob('*/ckpt_latest.pt'))
        if not candidates:
            raise FileNotFoundError('No SuperMimic-Reproduced checkpoint; train with scripts/19_train_supermimic.sh')
        return max(candidates, key=lambda p: p.stat().st_mtime_ns)
    path = Path(value).expanduser().resolve()
    if path.is_dir():
        path = path / 'ckpt_latest.pt'
    if not path.is_file():
        raise FileNotFoundError(path)
    manifest = path.parent / 'run_manifest.json'
    if manifest.exists():
        implementation = json.loads(manifest.read_text()).get('implementation')
        if implementation != 'SuperMimic-Reproduced':
            raise ValueError('Unsupported checkpoint implementation; only SuperMimic-Reproduced is supported')
    return path


def load_checkpoint(path):
    """只加载可信的本地/上游权重；先检查结构，再交给上游网络恢复。"""
    import torch
    ck = torch.load(path, map_location='cpu', weights_only=False)
    if not isinstance(ck, dict) or not {'wm', 'policy', 'critic', 'config', 'iteration'} <= ck.keys():
        raise ValueError('Expected a SuperMimic-Reproduced checkpoint')
    return ck


def check_data_contract(checkpoint, files):
    """本地产生的模型须沿用相同动作及顺序，避免把换动作算成续训收益。"""
    manifest = Path(checkpoint).parent / 'run_manifest.json'
    if manifest.exists():
        saved = json.loads(manifest.read_text())
        actual = [digest(p) for p in files]
        if actual != [row['sha256'] for row in saved['data']]:
            raise ValueError('Reference motion hashes/order differ from checkpoint manifest')
