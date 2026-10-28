"""模型回放与定量评测；接口只接受本方法模型。"""


def play(root, args):
    from workspace.common.workflows import play as implementation
    return implementation(root, args)


def evaluate(root, args):
    from workspace.common.workflows import evaluate as implementation
    return implementation(root, args, "amp")
