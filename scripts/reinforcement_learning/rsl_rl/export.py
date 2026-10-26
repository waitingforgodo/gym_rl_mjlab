"""程序 5：导出 TorchScript、ONNX 和完整部署接口清单。

运行：bash g1.sh --python scripts/reinforcement_learning/rsl_rl/export.py \
    --checkpoint /完整路径/model_2999.pt
打印的 DEPLOY_BUNDLE 目录供部署程序使用；实际实现见 workspace/exporting.py。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from workspace.cli import ROOT, parser, resolve
from workspace.exporting import export_bundle


def main():
    """加载模型 → 采集校验样本 → 导出 → 数值/结构检查 → 写清单和哈希。"""
    args = parser().parse_args(['export', *sys.argv[1:]])
    checkpoint = resolve(args.checkpoint)
    print(f'CHECKPOINT={checkpoint}', flush=True)
    export_bundle(ROOT, args, checkpoint)


if __name__ == '__main__':
    main()
