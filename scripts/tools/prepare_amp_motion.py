"""独立程序：转换本地 AMP 舞蹈数据，不创建 GPU 仿真。

运行：bash g1.sh --python scripts/tools/prepare_amp_motion.py
也可在末尾指定 --source /原始AMP动作.npz。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from workspace.cli import main


if __name__ == '__main__':
    sys.argv.insert(1, 'prepare-amp-motion')
    main()
