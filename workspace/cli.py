"""命令行入口：解析参数、选择运行目录、调用核心功能并打印结果。

轻量命令先只加载标准库；需要 GPU 的命令再延迟导入 runtime，避免读
配置、帮助或汇总时启动仿真。scripts/*.sh 都通过 g1.sh 委托到这里。
没有安装依赖、修改驱动、停止其他训练或发送真机控制指令的分支。
"""
import argparse
import importlib.metadata
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from .contract import choose_parallelism, json_safe, resolve_checkpoint, write_json

ROOT = Path(__file__).resolve().parents[1]


def emit(value):
    """将结果格式化为可保存、可复制的严格 JSON；不输出二进制模型。"""
    print(json.dumps(json_safe(value), ensure_ascii=False, indent=2, allow_nan=False), flush=True)


def parser():
    """统一定义子命令；脚本默认参数放前面，用户末尾参数可以覆盖。

    latest 是本空间正式运行；g1_continue 是旧隔离环境的特殊只读入口。
    --iterations 表示本次新增 PPO 更新数，不是最终模型编号。
    """
    result = argparse.ArgumentParser(description='G1 / mjlab 1.6.0 isolated simulation workspace')
    sub = result.add_subparsers(dest='command', required=True)
    sub.add_parser('doctor', help='Read-only runtime and hardware report')
    sub.add_parser('test', help='CPU unit tests; no simulation or optimization')
    check = sub.add_parser('self-check', help='Four environments, 32 steps, no training; runtime smoke check')
    check.add_argument('--device', default='cuda:0')
    check.add_argument('--profile', default='velocity_flat')
    check.add_argument('--motion')
    ref = sub.add_parser('import-reference', help='Copy only the two CSV motions; never execute reference code')
    # 优先使用工作空间内的参考资料；没有本地目录时兼容旧 Windows 路径。
    ref.add_argument('--source', default=str(ROOT / 'gym_rl') if (ROOT / 'gym_rl').is_dir()
                     else '/mnt/d/reference/reinforce/gym_rl')
    prep = sub.add_parser('prepare-motion', help='CPU kinematics: 60 Hz CSV -> control-rate NPZ')
    prep.add_argument('name', choices=['dance_102', 'gangnam_style'])
    inspect = sub.add_parser('inspect-motion')
    inspect.add_argument('file')
    for command in ('train', 'resume', 'play', 'evaluate', 'export'):
        p = sub.add_parser(command)
        p.add_argument('--profile', choices=[f.stem for f in sorted((ROOT/'profiles').glob('*.json'))])
        p.add_argument('--motion')
        p.add_argument('--device', default='cuda:0')
        if command != 'train':
            p.add_argument('--checkpoint', default='latest', help='Run directory, model_N.pt, latest, or old g1_continue')
            p.add_argument('--legacy-velocity', action='store_true', help='Explicitly assume official pinned flat G1 for old models without a manifest')
        if command in ('train', 'resume'):
            p.add_argument('--envs', type=positive)
            p.add_argument('--iterations', type=positive, help='ADDITIONAL PPO iterations for resume')
            p.add_argument('--name', default='g1')
            p.add_argument('--seed', type=int, default=42)
            p.add_argument('--benchmark-run', action='store_true', help=argparse.SUPPRESS)
        if command == 'play':
            p.add_argument('--viewer', choices=['viser', 'native'], default='viser')
            p.add_argument('--port', type=positive, default=8080)
        if command == 'evaluate':
            p.add_argument('--envs', type=positive, default=16)
            p.add_argument('--seconds', type=positive, default=60)
            p.add_argument('--velocity-cmd', nargs=3, type=float, metavar=('VX', 'VY', 'WZ'),
                           help='Fixed BODY-frame velocity, m/s m/s rad/s; disables random command branches')
    dep = sub.add_parser('deploy-sim', help='Replay exported TorchScript, without loading training checkpoint')
    dep.add_argument('--bundle', required=True)
    dep.add_argument('--device', default='cuda:0')
    dep.add_argument('--viewer', choices=['viser', 'native'], default='viser')
    dep.add_argument('--port', type=positive, default=8080)
    dep.add_argument('--headless', action='store_true')
    dep.add_argument('--seconds', type=positive, default=60)
    bench = sub.add_parser('benchmark', help='Sequential short PPO tests; never launch automatically')
    bench.add_argument('--envs', type=positive, nargs='+', default=[128, 256, 512])
    bench.add_argument('--iterations', type=positive, default=30)
    bench.add_argument('--profile', default='velocity_flat', choices=[f.stem for f in (ROOT/'profiles').glob('*.json')])
    bench.add_argument('--motion')
    bench.add_argument('--timeout', type=positive, default=900)
    summ = sub.add_parser('summary')
    summ.add_argument('--run', default='latest')
    config = sub.add_parser('show-config', help='Read saved manifest and full YAML configs; no simulation')
    config.add_argument('--checkpoint', default='latest')
    sources = sub.add_parser('show-sources', help='Locate installed environment/reward/PPO source; no simulation')
    sources.add_argument('--kind', choices=['all', 'velocity', 'tracking'], default='all')
    return result


def positive(value):
    """环境数、时长、端口等必须是正整数；在创建 GPU 环境之前检查。"""
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError('must be positive')
    return number


def resolve(value):
    """解析旧目录别名，再按数字模型编号选择 checkpoint。

    旧 g1_continue 与新 g1_2048_continue 不同：后者位于 outputs/runs，
    用 latest 或明确路径；本别名只搜索旧运行时的 g1_velocity 目录。
    """
    if value == 'g1_continue':
        runtime = Path((ROOT / '.runtime-path').read_text().strip())
        parent = runtime / 'outputs' / 'rsl_rl' / 'g1_velocity'
        runs = [p for p in parent.glob('*g1_continue*') if p.is_dir() and list(p.glob('model_*.pt'))]
        if not runs:
            raise FileNotFoundError(f'No old g1_continue under {parent}')
        value = str(max(runs, key=lambda p: p.name))
    return resolve_checkpoint(value, ROOT / 'outputs' / 'runs')


def doctor():
    """只读列出解释器、依赖版本和剩余资源；不把清单检查当 GPU 验收。"""
    from .telemetry import available_ram_gib, gpu_free_mib
    versions = {}
    for package in ('mjlab', 'torch', 'warp-lang', 'mujoco', 'mujoco-warp', 'rsl-rl-lib', 'onnx', 'tensorboard'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = 'MISSING'
    if versions['mjlab'] != '1.6.0':
        raise RuntimeError(f'Expected mjlab 1.6.0, found {versions["mjlab"]}')
    return dict(workspace=str(ROOT), interpreter=sys.executable, versions=versions,
                available_wsl_ram_gib=available_ram_gib(), gpu_free_mib=gpu_free_mib(),
                cpu_threads=os.environ.get('G1_CPU_THREADS', '2'),
                isolation='Separate files/caches/logs; reuses existing pinned interpreter without installing packages. CPU/RAM/GPU remain shared.',
                check_scope='Read-only inventory, not a GPU kernel or policy convergence test')


def self_check(args):
    """4 环境执行 32 步零动作，检查观测/奖励有限性，不做优化。

    零动作对应 default offset 的关节目标，不表示静止世界状态；机器人
    可能运动或摔倒，所以这个检查只证明软件链路能运行。
    """
    import torch
    from .runtime import initialize, build_configs
    initialize(ROOT)
    _, cfg, agent, _ = build_configs(ROOT, args.profile, 4, args.motion)
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device=args.device), agent.clip_actions)
    try:
        obs = env.get_observations()
        shapes = {name: list(value.shape) for name, value in obs.items()}
        for _ in range(32):
            obs, reward, _, _ = env.step(torch.zeros((env.num_envs, env.num_actions), device=args.device))
            if not torch.isfinite(reward).all() or not all(torch.isfinite(v).all() for v in obs.values()):
                raise FloatingPointError('Non-finite observations or rewards')
        return dict(status='passed', steps=32, num_envs=4, observation_shapes=shapes,
                    action_dim=env.num_actions, control_dt=env.unwrapped.step_dt,
                    note='No optimization; this verifies runtime only, not stable walking')
    finally:
        env.close()


def benchmark(args):
    """从小到大启动独立短训子进程，按资源和有效计时选择吞吐候选。

    子进程 stdout/stderr 写入每档日志，超时只终止这个子进程。主进程用
    单调时钟计总耗时；PPO 内计时由 runtime 兼容层处理。status=passed
    只表示进程成功，timing_valid 才决定计时能否参与推荐。
    """
    from .telemetry import gpu_free_mib, available_ram_gib, summarize
    if args.iterations < 10:
        raise ValueError('At least 10 iterations needed; first five are excluded from throughput statistics')
    destination = ROOT / 'outputs' / 'benchmarks' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    destination.mkdir(parents=True)
    rows = []
    for count in sorted(set(args.envs)):
        # 开始前余量门槛较高，为短测预留空间；它不是显存硬配额。
        free, ram = gpu_free_mib(), available_ram_gib()
        if free is None or ram is None or free < 2048 or ram < 3:
            rows.append(dict(num_envs=count, status='skipped', reason='Resource inventory unavailable, or free GPU<2 GiB / WSL RAM<3 GiB'))
            break
        print(f'Benchmark {count} environments; {args.iterations} PPO updates. Log: {destination / str(count)}.log', flush=True)
        command = [sys.executable, '-m', 'workspace.cli', 'train', '--profile', args.profile,
                   '--envs', str(count), '--iterations', str(args.iterations), '--name', f'benchmark_{count}', '--benchmark-run']
        # 参数数组不经 shell 拼接，带空格的动作路径仍是一个参数。
        if args.motion:
            command += ['--motion', str(Path(args.motion).resolve())]
        log_file = destination / f'{count}.log'
        started = time.monotonic()
        with log_file.open('w', encoding='utf-8') as output:
            try:
                process = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT, timeout=args.timeout, check=False)
                status = 'passed' if process.returncode == 0 else 'failed'
            except subprocess.TimeoutExpired:
                status = 'timeout'
        lines = log_file.read_text(encoding='utf-8', errors='replace').splitlines()
        run_lines = [line.split('=', 1)[1] for line in lines if line.startswith('RUN_DIR=')]
        row = dict(num_envs=count, status=status, wall_seconds=time.monotonic()-started, log=str(log_file))
        if run_lines and Path(run_lines[-1]).is_dir():
            try:
                row.update(summarize(Path(run_lines[-1])))
                row['status'] = status
            except (OSError, ValueError) as error:
                row['summary_error'] = str(error)
        rows.append(row)
        write_json(destination / 'results.json', {'profile': args.profile, 'rows': rows})
        emit({key: row.get(key) for key in ('num_envs', 'status', 'timing_valid', 'median_fps', 'min_free_vram_mib', 'min_available_ram_gib')})
        if row.get('timing_valid') is not True:
            print('Elapsed-time data invalid or missing; this row is excluded from recommendations.', flush=True)
        if status != 'passed' or (row.get('min_free_vram_mib') or 0) < 1024 or (row.get('min_available_ram_gib') or 0) < 2:
            print('Stopping larger tests: failure or resource margin is too small.', flush=True)
            break
    result = dict(profile=args.profile, rows=rows, recommended_num_envs=choose_parallelism(rows),
                  decision='Require valid elapsed times, >=1 GiB free GPU and >=2 GiB WSL RAM; choose smallest batch within 5% of best FPS.',
                  limits='Short benchmark; repeat for tracking separately. GPU thermals and desktop responsiveness still need observation.')
    write_json(destination / 'results.json', result)
    print(f'BENCHMARK_RESULT={destination / "results.json"}', flush=True)
    return result


def main():
    """分发命令并处理输出位置，推理/评估始终在 finally 关闭环境。

    训练/benchmark 用文件锁防重复启动。show-config 只读文本，不执行
    旧 YAML 中的 Python 类型标签。模型是否可信仍由其来源决定。
    """
    args = parser().parse_args()
    command = args.command
    if command == 'doctor':
        emit(doctor())
    elif command == 'test':
        import unittest
        suite = unittest.defaultTestLoader.discover(str(ROOT / 'tests'))
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        if not result.wasSuccessful():
            raise SystemExit(1)
    elif command == 'self-check':
        emit(self_check(args))
    elif command == 'import-reference':
        from .motion import import_reference
        emit(import_reference(ROOT, args.source))
    elif command == 'prepare-motion':
        from .runtime import initialize
        from .motion import prepare
        initialize(ROOT)
        emit(prepare(ROOT, args.name))
    elif command == 'inspect-motion':
        from .motion import inspect
        emit(inspect(Path(args.file).expanduser().resolve()))
    elif command == 'benchmark':
        with training_lock():
            emit(benchmark(args))
    elif command in ('train', 'resume'):
        from .training import train
        from .telemetry import summarize
        checkpoint = resolve(args.checkpoint) if command == 'resume' else None
        if args.benchmark_run:
            run = train(ROOT, args, checkpoint)
        else:
            with training_lock():
                run = train(ROOT, args, checkpoint)
        emit(summarize(run))
    elif command == 'summary':
        from .telemetry import summarize
        emit(summarize(resolve(args.run).parent))
    elif command == 'show-config':
        run = resolve(args.checkpoint).parent
        for relative in ('run_manifest.json', 'params/agent.yaml', 'params/env.yaml'):
            path = run / relative
            print(f'\n===== {path} =====')
            print(path.read_text(encoding='utf-8') if path.is_file() else '(not present in this old run)')
    elif command == 'show-sources':
        from .source_map import locate_sources
        emit(locate_sources(args.kind))
    elif command in ('play', 'evaluate', 'export'):
        from .inference import load_for_inference, view, evaluate
        from .exporting import export_bundle
        checkpoint = resolve(args.checkpoint)
        print(f'CHECKPOINT={checkpoint}', flush=True)
        if command == 'export':
            export_bundle(ROOT, args, checkpoint)
            return
        env, _, policy, manifest = load_for_inference(ROOT, args, checkpoint, getattr(args, 'envs', 1))
        try:
            if command == 'play':
                view(env, policy, args.viewer, args.port)
            else:
                result = evaluate(env, policy, args.seconds)
                result.update(checkpoint=str(checkpoint), task=manifest['task'],
                              evaluation_conditions=manifest['evaluation_conditions'])
                target = ROOT / 'outputs' / 'evaluations' / f'{datetime.now():%Y%m%d_%H%M%S_%f}.json'
                write_json(target, result)
                print(f'EVALUATION={target}', flush=True)
                emit(result)
        finally:
            env.close()
    elif command == 'deploy-sim':
        from .deployment import deploy_sim
        emit(deploy_sim(ROOT, args))


def training_lock():
    """返回 Linux 文件锁上下文，进程退出自动释放。

    只协调本空间，不会停止其他目录的训练。benchmark 父进程持锁，
    子进程通过内部标志跳过重复取锁。
    """
    from contextlib import contextmanager
    @contextmanager
    def lock():
        """非阻塞取锁：已有训练立即报错，避免排队后意外启动耗时任务。"""
        import fcntl
        path = ROOT / '.cache' / 'training.lock'
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('a') as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError('Another training/benchmark process is using this workspace') from None
            yield
    return lock()


if __name__ == '__main__':
    main()
