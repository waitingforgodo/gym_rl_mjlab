"""本地连接层与固定提交的完整上游源码位置。"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = Path(__file__).resolve().parent / 'upstream'
RUNS = ROOT / 'outputs/supermimic/runs'
