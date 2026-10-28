"""AMP 专家动作的转换入口；转换算法仍在 conversion.py。"""


def prepare(root, args):
    from workspace.common.runtime import initialize
    from .conversion import prepare_amp_motion
    initialize(root)
    return prepare_amp_motion(root, args.source)
