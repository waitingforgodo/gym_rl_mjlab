"""依据 arXiv:2610.28480v1 编写的 SuperMimic 独立复现。

官方截至 2026-10-27 未公开训练源码。本目录只声称复现论文公开的算法、
网络和训练接口；不能把本地单动作实验写成论文 129514 动作结果的复现。
"""

from .config import SuperMimicConfig


def source_sha256(root):
    """将全部 SuperMimic Python 源码纳入 checkpoint 契约。"""
    import hashlib
    digest = hashlib.sha256()
    for path in sorted((root/'workspace'/'supermimic').glob('*.py')):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


__all__ = ['SuperMimicConfig', 'source_sha256']
