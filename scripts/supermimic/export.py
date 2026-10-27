"""程序 3：将 SuperMimic actor 导出为独立 TorchScript 仿真部署包。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from workspace.cli import ROOT
from workspace.supermimic.cli import checkpoint, checkpoint_parser


def main():
    args = checkpoint_parser('Export a SuperMimic simulation bundle').parse_args()
    args.checkpoint = checkpoint(args.checkpoint, ROOT)
    from workspace.supermimic.exporting import export
    export(ROOT, args)


if __name__ == '__main__':
    main()
