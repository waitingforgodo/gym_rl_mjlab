"""用相同环境数、时长、随机种子和命令协议比较多个策略。

例：bash g1.sh --python scripts/tools/compare_policies.py \
    --checkpoints /基线/model_2999.pt /候选/model_9999.pt --envs 16 --seconds 60
速度任务依次测试固定前进、横移和转向；tracking 按同一个动作比较。
新旧奖励函数不同，因此报告不按总奖励评出“最好”，主要看失败和 RMSE。
"""
import argparse
from datetime import datetime
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from workspace.cli import ROOT, emit, positive, resolve
from workspace.common.contract import sha256, write_json
from workspace.common.inference import evaluate, load_for_inference
from workspace.common.runtime import checkpoint_context


def main():
    """先核对任务/动作一致，再逐个创建环境，避免多个策略同时抢 GPU。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoints', nargs='+', required=True)
    parser.add_argument('--envs', type=positive, default=16)
    parser.add_argument('--seconds', type=positive, default=60)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output')
    parser.add_argument('--protocols', nargs='+',
                        choices=['forward_0.5', 'forward_1.0', 'lateral_0.3', 'yaw_0.5'],
                        help='Optional subset for velocity tasks; default tests all four commands')
    args = parser.parse_args()
    checkpoints = [resolve(value) for value in args.checkpoints]
    if len(set(checkpoints)) < 2:
        parser.error('Provide at least two different checkpoints')
    contexts = [checkpoint_context(ROOT, checkpoint)[0] for checkpoint in checkpoints]
    # 原始物理后端和任务必须相同；两种 tracking 观测版本也不混成一组。
    contract = (contexts[0]['task'], contexts[0].get('motion_sha256'))
    if any((context['task'], context.get('motion_sha256')) != contract for context in contexts):
        parser.error('Checkpoints must use the same task and reference motion')
    tracking = contexts[0].get('motion_file') is not None
    protocols = [('motion', None)] if tracking else [
        ('forward_0.5', (0.5, 0., 0.)),
        ('forward_1.0', (1., 0., 0.)),
        ('lateral_0.3', (0., 0.3, 0.)),
        ('yaw_0.5', (0., 0., 0.5)),
    ]
    if args.protocols:
        if tracking:
            parser.error('--protocols selects velocity commands and cannot be used for motion tracking')
        by_name = dict(protocols)
        protocols = [(name, by_name[name]) for name in dict.fromkeys(args.protocols)]
    target = Path(args.output).expanduser().resolve() if args.output else (
        ROOT / 'outputs/evaluations' / f'comparison_{datetime.now():%Y%m%d_%H%M%S_%f}.json')
    report = dict(seed=args.seed, num_envs=args.envs, seconds_per_protocol=args.seconds,
                  task=contract[0], motion_sha256=contract[1], policies=[],
                  interpretation='Compare failure statistics and physical RMSE. Rewards may differ across profiles; no automatic winner is claimed.')
    for checkpoint, context in zip(checkpoints, contexts):
        row = dict(checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint),
                   profile=context['profile'], results={})
        report['policies'].append(row)
        for label, command in protocols:
            print(f'COMPARING={checkpoint} PROTOCOL={label}', flush=True)
            options = SimpleNamespace(profile=None, motion=None, legacy_velocity=False,
                                      device=args.device, velocity_cmd=command, evaluation_seed=args.seed)
            env, _, policy, manifest = load_for_inference(ROOT, options, checkpoint, args.envs)
            try:
                result = evaluate(env, policy, args.seconds)
                result['evaluation_conditions'] = manifest['evaluation_conditions']
                row['results'][label] = result
            finally:
                env.close()
            # 每个协议完成就落盘；后续中断时已完成的数据仍可检查。
            write_json(target, report)
    print(f'COMPARISON={target}', flush=True)
    emit(report)


if __name__ == '__main__':
    main()
