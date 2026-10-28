"""velocity 独立命令行；公共参数与分发位于 common.method_cli。"""


def main(argv=None):
    from workspace.common.method_cli import main as dispatch
    dispatch("velocity", argv)
