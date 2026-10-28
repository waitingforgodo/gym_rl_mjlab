"""程序 1：从零训练 G1 策略。

运行：bash g1.sh --python scripts/reinforcement_learning/rsl_rl/train.py \
    --profile velocity_flat_4096 --envs 4096 --iterations 3000 --name walk
逐步实现见 workspace/common/training.py；PPO 本体位置用 show-sources 查看。
"""
import sys
from pathlib import Path

# 支持 python /绝对路径/train.py，从其他目录启动也能找到本空间源码。
# 这只加入项目源码路径，不改用户的解释器、环境或第三方安装文件。
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from workspace.cli import ROOT, emit, parser, training_lock
from workspace.common.telemetry import summarize
from workspace.common.training import train


def main():
    """参数检查 → 独占训练锁 → 训练 → 输出曲线和资源统计。"""
    # 共用参数定义，保证独立入口与 g1.sh train 的默认值和检查一致。
    args = parser().parse_args(['train', *sys.argv[1:]])
    with training_lock():
        # train 内部创建新目录，不向旧实验目录写入任何模型。
        run = train(ROOT, args)
    emit(summarize(run))


if __name__ == '__main__':
    main()
