"""统一入口：g1.sh <amp|beyondmimic|supermimic|velocity|instinct> <功能>。

doctor/test/benchmark 等公共命令及原通用命令继续转发至 common.cli。
导入这个模块不会创建环境、加载模型或启动训练。
"""
import importlib
import sys

# 兼容旧脚本对参数解析、文件锁等公共函数的导入。
from .common.cli import *  # noqa: F401,F403


def main():
    if len(sys.argv) > 1 and sys.argv[1] in ('amp', 'beyondmimic', 'supermimic', 'velocity', 'instinct'):
        method = sys.argv[1]
        importlib.import_module(f'workspace.{method}.cli').main(sys.argv[2:])
    elif len(sys.argv) > 1 and sys.argv[1] in ('-h', '--help'):
        print('按方法使用：g1.sh {amp,beyondmimic,supermimic,velocity,instinct} <功能> [参数]\n')
        from .common.cli import main as common_main
        common_main()
    else:
        from .common.cli import main as common_main
        common_main()


if __name__ == '__main__':
    main()
