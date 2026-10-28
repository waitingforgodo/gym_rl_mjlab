"""上游源码导入与原运行环境初始化；不执行依赖安装。"""
import os
from pathlib import Path
import sys
from .paths import ROOT, UPSTREAM


def activate():
    """直接导入已固定提交的源码，优先于任何同名 site-packages 安装。"""
    source = UPSTREAM / 'src'
    if not (source / 'supermimic/config.py').is_file():
        raise FileNotFoundError(f'SuperMimic source missing: {source}')
    loaded = sys.modules.get('supermimic')
    if loaded is not None and Path(loaded.__file__).resolve().parent != source / 'supermimic':
        raise RuntimeError('A different supermimic package was already imported')
    sys.path.insert(0, str(source))


def initialize():
    """使用原运行环境，把运行缓存放回当前工作空间。没有包管理操作。"""
    activate()
    import torch
    import warp as wp
    wp.config.kernel_cache_dir = str(ROOT / '.cache/warp')
    torch.set_num_threads(int(os.environ.get('G1_CPU_THREADS', '2')))
