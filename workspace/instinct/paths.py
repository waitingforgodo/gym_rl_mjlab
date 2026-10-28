"""源码、附加依赖、训练产物的项目内路径。"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MJ_SOURCE = HERE/'upstream/InstinctMJ'
RL_SOURCE = HERE/'upstream/instinct_rl'
ADDONS = ROOT/'.cache/instinct-deps'
OUTPUTS = ROOT/'outputs/instinct'
