"""程序 1：采集真实 mjlab 转移并交替训练 SuperMimic 三张网络。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from workspace.cli import ROOT, training_lock
from workspace.supermimic.cli import train_parser


def main():
    args = train_parser().parse_args()
    if args.envs < 1 or (args.iterations is not None and args.iterations < 1):
        train_parser().error('--envs and --iterations must be positive')
    from workspace.supermimic.training import train
    with training_lock():
        train(ROOT, args)


if __name__ == '__main__':
    main()
