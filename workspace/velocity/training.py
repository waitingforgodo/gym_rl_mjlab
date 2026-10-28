"""从零训练与续训；复用公共 PPO 循环，保留本方法的模型格式。"""


def train(root, args):
    from workspace.common.workflows import train as implementation
    return implementation(root, args)


def resume(root, args):
    from workspace.common.workflows import train as implementation
    return implementation(root, args, resume=True)
