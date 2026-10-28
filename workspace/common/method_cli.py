"""AMP、BeyondMimic、速度 PPO 的统一参数定义与功能分发。

每个方法有自己的包与入口，共用参数语义：iterations 是新增 PPO 更新数。
这里仅做接口和校验；训练/推理/导出的实际循环在相应公共模块中。
"""
import argparse
import json
from pathlib import Path
from .cli import ROOT, emit, positive
from .methods import DEFAULT_PROFILES, require_method, resolve_model
from .profiles import profile_paths, read_profile


def parser(method):
    p = argparse.ArgumentParser(prog=f'g1.sh {method}', description=f'{method}: train/resume/play/evaluate/export/deploy')
    sub = p.add_subparsers(dest='command', required=True)
    for command in ('train', 'resume', 'play', 'evaluate', 'export'):
        child = sub.add_parser(command)
        child.add_argument('--profile', choices=[f.stem for f in profile_paths(ROOT, method)])
        child.add_argument('--motion', help='Standard NPZ reference motion')
        child.add_argument('--device', default='cuda:0')
        if command != 'train':
            child.add_argument('--checkpoint', default='latest', help=f'Only {method} models are selected')
        if command in ('train', 'resume'):
            child.add_argument('--envs', type=positive)
            child.add_argument('--seed', type=int)
            child.add_argument('--iterations', type=positive, default=1000 if command == 'resume' else None,
                               help='ADDITIONAL PPO updates; resume default 1000')
            child.add_argument('--name', default=method)
            child.add_argument('--dry-run', action='store_true', help='Validate config; do not create simulation or train')
        if command == 'play':
            child.add_argument('--viewer', choices=('viser', 'native'), default='viser')
            child.add_argument('--port', type=positive, default=8080)
        if command == 'evaluate':
            child.add_argument('--envs', type=positive, default=16)
            child.add_argument('--seconds', type=positive, default=60)
            child.add_argument('--evaluation-seed', type=int)
            if method == 'velocity':
                child.add_argument('--velocity-cmd', nargs=3, type=float, help='VX VY WZ in body frame')
    child = sub.add_parser('deploy', help='Load an exported JIT bundle into simulation')
    child.add_argument('--bundle', required=True, type=Path)
    child.add_argument('--device', default='cuda:0')
    child.add_argument('--viewer', choices=('viser', 'native'), default='viser')
    child.add_argument('--port', type=positive, default=8080)
    child.add_argument('--headless', action='store_true')
    child.add_argument('--seconds', type=positive, default=60)
    if method in ('amp', 'beyondmimic'):
        child = sub.add_parser('prepare', help='Convert and validate reference data on CPU')
        if method == 'amp':
            child.add_argument('--source')
        else:
            child.add_argument('name', choices=('dance_102', 'gangnam_style'))
    return p


def main(method, argv=None):
    import importlib
    p = parser(method)
    args = p.parse_args(argv)
    args.legacy_velocity = False
    try:
        if args.command == 'train':
            args.profile = args.profile or DEFAULT_PROFILES[method]
            require_method(method, read_profile(ROOT, args.profile))
            if method == 'beyondmimic' and not args.motion:
                args.motion = str(ROOT/'motions/processed/dance_102.npz')
        if getattr(args, 'profile', None):
            require_method(method, read_profile(ROOT, args.profile))
        if args.command in ('resume', 'play', 'evaluate', 'export'):
            args.checkpoint = resolve_model(ROOT, args.checkpoint, method)
        if args.command == 'deploy':
            manifest = json.loads((args.bundle/'manifest.json').read_text())
            require_method(method, manifest)
        if args.command in ('train', 'resume'):
            api = importlib.import_module(f'workspace.{method}.training')
        elif args.command in ('play', 'evaluate'):
            api = importlib.import_module(f'workspace.{method}.inference')
        elif args.command in ('export', 'deploy'):
            api = importlib.import_module(f'workspace.{method}.deployment')
        else:
            api = importlib.import_module(f'workspace.{method}.data')
        result = getattr(api, args.command)(ROOT, args)
        if result is not None:
            emit(result)
    except (ValueError, FileNotFoundError) as exc:
        p.error(str(exc))
