"""Locate one current formal training run without importing training frameworks."""
from pathlib import Path
import sys


def current_logdir(root):
    """按启动清单选择最近一次正式 run，排除 checks/benchmarks 等历史检查。"""
    root = Path(root).resolve()
    outputs = root / 'outputs'
    parents = [outputs / 'runs'] + [outputs / method / 'runs'
               for method in ('amp', 'beyondmimic', 'supermimic', 'velocity', 'instinct')]
    candidates = []
    for parent in parents:
        if not parent.is_dir():
            continue
        for run in parent.iterdir():
            if not run.is_dir():
                continue
            logdir = run / 'tb' if parent.parent.name == 'supermimic' or (run / 'tb').is_dir() else run
            manifest = run / 'run_manifest.json'
            if manifest.is_file():
                # 新训练尚未写出第一条 event 时也选中新 run；不因旧日志
                # 后来被触碰或复制而误切回旧训练。
                stamp = manifest.stat().st_mtime_ns
            else:
                events = list(logdir.glob('events.out.tfevents.*'))
                if not events:
                    continue
                stamp = max(path.stat().st_mtime_ns for path in events)
            candidates.append((stamp, str(run), logdir))
    if not candidates:
        raise FileNotFoundError('没有找到正式训练日志；请先开始训练，或用 --logdir 指定目录')
    return max(candidates, key=lambda item: item[:2])[2]


if __name__ == '__main__':
    try:
        print(current_logdir(sys.argv[1]))
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)
