"""Locate one current formal training run without importing training frameworks."""
from pathlib import Path
import sys


def port_available(port):
    """仅将地址已占用视为冲突，权限等错误不能伪装成可换端口。"""
    import errno
    import socket
    with socket.socket() as listener:
        try:
            listener.bind(('127.0.0.1', port))
        except OSError as exc:
            if exc.errno == errno.EADDRINUSE:
                return False
            raise
    return True


def existing_logdir(port):
    """只读取本机 TensorBoard API；不用环境代理访问 loopback。"""
    import json
    import urllib.error
    import urllib.request
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(f'http://127.0.0.1:{port}/data/environment', timeout=1) as response:
            environment = json.load(response)
            if not isinstance(environment, dict):
                return None
            value = environment.get('data_location')
            return value if isinstance(value, str) else None
    except (OSError, ValueError, urllib.error.URLError):
        return None


def select_endpoint(logdir, port, *, reuse=True):
    """复用同目录的 TensorBoard；其他占用则另选端口，不停止任何进程。"""
    if not 0 <= port <= 65535:
        raise ValueError('TensorBoard port must be between 0 and 65535')
    if port == 0:
        return 'start', 0
    for candidate in range(port, min(port + 100, 65536)):
        if port_available(candidate):
            return 'start', candidate
        location = existing_logdir(candidate) if reuse else None
        if location is not None and (location == logdir or Path(location).resolve() == Path(logdir).resolve()):
            return 'reuse', candidate
    raise RuntimeError('没有找到空闲端口，请用 --port 指定其他端口')


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
        if sys.argv[1] == '--endpoint':
            action, port = select_endpoint(sys.argv[2], int(sys.argv[3]),
                                           reuse=len(sys.argv) < 5 or sys.argv[4] == '1')
            print(action, port)
        else:
            print(current_logdir(sys.argv[1]))
    except (OSError, ValueError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)
