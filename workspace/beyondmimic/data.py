"""BeyondMimic 参考动作转换入口，输出标准 50 Hz NPZ。"""


def prepare(root, args):
    from workspace.common.runtime import initialize
    from workspace.common.motion import prepare as implementation
    initialize(root)
    return implementation(root, args.name)
