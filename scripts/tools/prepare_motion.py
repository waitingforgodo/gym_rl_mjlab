"""准备程序 B：CSV → 当前 G1 模型正运动学 → 控制频率 NPZ。

运行：bash g1.sh --python scripts/tools/prepare_motion.py dance_102
也支持 gangnam_style。重采样、坐标变换和数组检查详见 workspace/common/motion.py。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from workspace.cli import ROOT, emit, parser
from workspace.common.motion import prepare
from workspace.common.runtime import initialize


def main():
    """验证依赖后在 CPU 上转换动作；已有结果按原规则保留。"""
    args = parser().parse_args(['prepare-motion', *sys.argv[1:]])
    initialize(ROOT)
    emit(prepare(ROOT, args.name))


if __name__ == '__main__':
    main()
