"""兼容独立程序入口；实际实现集中在 workspace/supermimic/。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from workspace.supermimic.cli import train_main as main

if __name__ == "__main__":
    main()
