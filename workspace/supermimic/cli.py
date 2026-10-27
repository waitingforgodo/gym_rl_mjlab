"""SuperMimic 四个独立程序共用的轻量参数定义。"""
import argparse
from pathlib import Path


def checkpoint(value, root):
    path = Path(value).expanduser()
    if value == 'latest':
        candidates = list((root/'outputs'/'supermimic'/'runs').glob('*/supermimic_*.pt'))
        if not candidates:
            raise FileNotFoundError('No SuperMimic checkpoint found')
        return max(candidates, key=lambda p: (p.parent.name, int(p.stem.rsplit('_', 1)[1])))
    path = path.resolve()
    if path.is_dir():
        candidates = list(path.glob('supermimic_*.pt'))
        if not candidates:
            raise FileNotFoundError(f'No SuperMimic checkpoint under {path}')
        return max(candidates, key=lambda p: int(p.stem.rsplit('_', 1)[1]))
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def train_parser():
    p = argparse.ArgumentParser(description='Train the independent SuperMimic reproduction')
    p.add_argument('--profile', default='supermimic_4096', choices=['supermimic_4096'])
    p.add_argument('--motion')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--envs', type=int, default=4096)
    p.add_argument('--iterations', type=int)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--name', default='supermimic')
    p.add_argument('--checkpoint', type=Path, help='Optional SuperMimic checkpoint; replay warmup restarts')
    return p


def checkpoint_parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--checkpoint', default='latest')
    p.add_argument('--motion')
    p.add_argument('--device', default='cuda:0')
    return p


def add_viewer(p):
    p.add_argument('--viewer', choices=('viser', 'native'), default='viser')
    p.add_argument('--port', type=int, default=8080)
    return p
