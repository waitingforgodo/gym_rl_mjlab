"""程序 4：不读取训练 checkpoint，只用导出 JIT 做 mjlab 闭环部署。"""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from workspace.cli import ROOT


def main():
    p = argparse.ArgumentParser(description='Deploy an exported SuperMimic actor in mjlab')
    p.add_argument('--bundle', required=True, type=Path)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--viewer', choices=('viser', 'native'), default='viser')
    p.add_argument('--port', type=int, default=8080)
    p.add_argument('--headless', action='store_true')
    p.add_argument('--seconds', type=int, default=60)
    args = p.parse_args()
    from workspace.supermimic.deployment import deploy
    print(json.dumps(deploy(ROOT, args), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
