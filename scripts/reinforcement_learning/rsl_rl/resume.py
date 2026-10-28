"""程序 2：恢复训练状态，在新运行目录继续学习。

运行：bash g1.sh --python scripts/reinforcement_learning/rsl_rl/resume.py \
    --checkpoint /完整路径/model_2999.pt --envs 256 --iterations 1000
iterations 是新增更新数；恢复模型、归一化和优化器，但不是逐位重放旧轨迹。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from workspace.cli import ROOT, emit, parser, resolve, training_lock
from workspace.common.telemetry import summarize
from workspace.common.training import train


def main():
    """选择 checkpoint → 验证训练契约 → 恢复并新增 PPO 更新。"""
    args = parser().parse_args(['resume', *sys.argv[1:]])
    # 选择数字最大的 model_N.pt；不会把导出的 policy_jit.pt 当 checkpoint。
    checkpoint = resolve(args.checkpoint)
    print(f'CHECKPOINT={checkpoint}', flush=True)
    with training_lock():
        run = train(ROOT, args, checkpoint)
    emit(summarize(run))


if __name__ == '__main__':
    main()
