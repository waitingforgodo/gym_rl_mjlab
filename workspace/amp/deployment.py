"""导出 actor 为 JIT/ONNX 部署包，以及同后端仿真部署。"""


def export(root, args):
    from workspace.common.workflows import export as implementation
    return implementation(root, args)


def deploy(root, args):
    from workspace.common.workflows import deploy as implementation
    return implementation(root, args)
