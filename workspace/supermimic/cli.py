"""SuperMimic 功能入口：train/resume/play/evaluate/deploy。"""
import argparse
from pathlib import Path
import sys
from .configuration import train_parser, positive
from .inference import inference_parser


def parser():
    p = argparse.ArgumentParser(prog='g1.sh supermimic', description='SuperMimic：世界模型/BPTT，直接 checkpoint 仿真')
    sub = p.add_subparsers(dest='command', required=True)
    prepare = sub.add_parser('prepare', help='Download upstream LAFAN1 G1 40 CSV clips and convert to standard 50 Hz NPZ')
    prepare.add_argument('--workers', type=positive, default=2, help='Concurrent downloads only; conversion is memory-bounded')
    for command in ('train', 'resume', 'preflight'):
        child = sub.add_parser(command, parents=[train_parser()], add_help=False)
        if command == 'resume':
            child.add_argument('--checkpoint', type=Path, help='Alias for --resume; target TOTAL env steps required')
    for command in ('play', 'deploy', 'evaluate'):
        child = sub.add_parser(command, parents=[inference_parser(command)], add_help=False)
        if command == 'evaluate':
            child.add_argument('--output', type=Path)
        else:
            child.add_argument('--viewer', choices=('viser', 'native'), default='viser')
            child.add_argument('--port', type=positive, default=8080)
            child.add_argument('--headless', action='store_true')
            child.add_argument('--steps', type=positive, default=500)
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    try:
        if args.command == 'prepare':
            import json
            from workspace.common.datasets import prepare_lafan1
            print(json.dumps(prepare_lafan1(args.workers), ensure_ascii=False, indent=2))
        elif args.command == 'preflight':
            from .training import preflight
            preflight(args)
        elif args.command in ('train', 'resume'):
            if args.command == 'resume':
                args.resume = args.checkpoint or args.resume
                if args.resume is None:
                    p.error('resume requires --checkpoint or --resume')
            from .training import train
            train(args)
        elif args.command == 'deploy':
            from .deployment import deploy_sim
            deploy_sim(args)
        else:
            from .inference import play, evaluate
            (evaluate if args.command == 'evaluate' else play)(args)
    except (ValueError, FileNotFoundError, FileExistsError, FloatingPointError) as exc:
        p.error(str(exc))


def train_main():
    main(['train', *sys.argv[1:]])


def play_main():
    main(['play', *sys.argv[1:]])


def evaluate_main():
    main(['evaluate', *sys.argv[1:]])
