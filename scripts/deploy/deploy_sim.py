"""程序 6：使用导出包进行仿真部署，不读取训练 checkpoint。

运行：bash g1.sh --python scripts/deploy/deploy_sim.py \
    --bundle /导出目录 --headless --seconds 60
去掉 --headless 可网页回放；观测与动力学仍由相同 mjlab 后端构造。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from workspace.cli import ROOT, emit, parser
from workspace.deployment import deploy_sim


def main():
    """校验部署包，构建环境，用 JIT 前向推理驱动机器人并保存评估结果。"""
    args = parser().parse_args(['deploy-sim', *sys.argv[1:]])
    emit(deploy_sim(ROOT, args))


if __name__ == '__main__':
    main()
