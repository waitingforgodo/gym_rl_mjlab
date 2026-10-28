"""准备程序 A：只复制参考 CSV；默认读取本工作空间内的 gym_rl。

运行：bash g1.sh --python scripts/tools/import_motions.py
可用 --source /其他/gym_rl 指定资料目录；不会执行其中的训练或部署程序。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from workspace.cli import ROOT, emit, parser
from workspace.common.motion import import_reference


def main():
    """按已知文件名找出两份动作，复制到 motions/raw 并记录来源哈希。"""
    args = parser().parse_args(['import-reference', *sys.argv[1:]])
    emit(import_reference(ROOT, args.source))


if __name__ == '__main__':
    main()
