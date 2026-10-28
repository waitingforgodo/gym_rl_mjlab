"""源码导航：定位当前实际使用的环境、奖励、机器人、网络和 PPO 源码。

运行：bash g1.sh --python scripts/tools/show_sources.py --kind velocity
仅使用 Python 包元数据和 AST 读取源码，不导入 Torch、Warp 或创建仿真。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from workspace.cli import emit, parser
from workspace.common.source_map import locate_sources


def main():
    """输出文件绝对路径与关键函数行号，可直接在 IDE 中打开。"""
    args = parser().parse_args(['show-sources', *sys.argv[1:]])
    emit(locate_sources(args.kind))


if __name__ == '__main__':
    main()
