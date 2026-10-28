"""程序 4：无窗口定量评估，记录失败事件、存活比例和跟踪误差。

运行：bash g1.sh --python scripts/reinforcement_learning/rsl_rl/evaluate.py \
    --checkpoint latest --envs 16 --seconds 60 --velocity-cmd 0.5 0 0
跟踪任务不传 --velocity-cmd；该参数仅适用于速度命令任务。
"""
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from workspace.cli import ROOT, emit, parser, resolve
from workspace.common.contract import write_json
from workspace.common.inference import evaluate, load_for_inference


def main():
    """选择模型和评估协议，在固定仿真时长内累计指标并保存 JSON。"""
    args = parser().parse_args(['evaluate', *sys.argv[1:]])
    checkpoint = resolve(args.checkpoint)
    print(f'CHECKPOINT={checkpoint}', flush=True)
    env, _, policy, manifest = load_for_inference(ROOT, args, checkpoint, args.envs)
    try:
        result = evaluate(env, policy, args.seconds)
        # 把命令范围、噪声、推力开关等条件和结果一起保存，便于公平比较。
        result.update(checkpoint=str(checkpoint), task=manifest['task'],
                      evaluation_conditions=manifest['evaluation_conditions'])
        target = ROOT / 'outputs' / 'evaluations' / f'{datetime.now():%Y%m%d_%H%M%S_%f}.json'
        write_json(target, result)
        print(f'EVALUATION={target}', flush=True)
        emit(result)
    finally:
        env.close()


if __name__ == '__main__':
    main()
