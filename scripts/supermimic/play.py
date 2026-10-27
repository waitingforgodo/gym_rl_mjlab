"""程序 2：加载 SuperMimic 训练 checkpoint 并可视化回放。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from workspace.cli import ROOT
from workspace.supermimic.cli import add_viewer, checkpoint, checkpoint_parser


def main():
    args = add_viewer(checkpoint_parser('Play a SuperMimic checkpoint')).parse_args()
    args.checkpoint = checkpoint(args.checkpoint, ROOT)
    print(f'CHECKPOINT={args.checkpoint}', flush=True)
    from workspace.supermimic.inference import play
    play(ROOT, args)


if __name__ == '__main__':
    main()
