"""程序 3：加载训练 checkpoint，交互回放策略。

运行：bash g1.sh --python scripts/reinforcement_learning/rsl_rl/play.py \
    --checkpoint latest --viewer viser --port 8080
打开终端打印的网页地址。此程序不训练，详细推理实现见 workspace/inference.py。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from workspace.cli import ROOT, parser, resolve
from workspace.inference import load_for_inference, view


def main():
    """恢复任务配置 → 加载 actor → 循环推理和仿真 → 释放环境。"""
    args = parser().parse_args(['play', *sys.argv[1:]])
    checkpoint = resolve(args.checkpoint)
    print(f'CHECKPOINT={checkpoint}', flush=True)
    env, _, policy, _ = load_for_inference(ROOT, args, checkpoint)
    try:
        # Viser/native viewer 自己管理循环；policy 接收环境构造的观测字典。
        view(env, policy, args.viewer, args.port)
    finally:
        env.close()


if __name__ == '__main__':
    main()
