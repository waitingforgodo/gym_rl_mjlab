"""只对当前进程加载 Instinct 源码和本地附加依赖，不覆盖原环境。"""
import importlib.metadata
import json
import os
import sys
from .paths import ROOT, HERE, MJ_SOURCE, RL_SOURCE, ADDONS
from workspace.common.contract import sha256


def activate():
    """源码优先，附加依赖仅对 Instinct 命令生效。"""
    for path in (ADDONS, RL_SOURCE, MJ_SOURCE/'src'):
        sys.path.insert(0, str(path))
    import warp as wp
    import torch
    if importlib.metadata.version('mjlab') != '1.6.0':
        raise RuntimeError('Instinct backend targets the existing mjlab 1.6.0 environment')
    wp.config.kernel_cache_dir = str(ROOT/'.cache/warp')
    torch.set_num_threads(int(os.environ.get('G1_CPU_THREADS', '2')))
    from mjlab.utils.torch import configure_torch_backends
    configure_torch_backends()
    import instinct_mj.tasks  # noqa: F401


def provenance(verify=True):
    """固定两个上游提交，并记录实际运行版本；不把代码变动伪装成同一模型。"""
    sources = []
    for folder in (MJ_SOURCE, RL_SOURCE):
        info = json.loads((folder/'SOURCE_INFO.json').read_text())
        if verify:
            for relative, expected in {**info['files'], **info['assets']}.items():
                file = folder/relative
                if not file.is_file() or sha256(file) != expected:
                    raise ValueError(f'Upstream source/asset differs or is missing: {file}; run instinct setup')
        sources.append(dict(repository=info['repository'], commit=info['commit']))
    return dict(backend='instinct_mj', sources=sources,
                runtime_versions={name:importlib.metadata.version(name) for name in
                                  ('mjlab','torch','mujoco','mujoco-warp','warp-lang')})
