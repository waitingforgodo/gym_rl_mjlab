"""旧导入路径兼容；协议实际实现位于 workspace.common.contract。

AMP 已有源码和模型仍引用此路径，因此保留轻量转发。
"""
from .common.contract import *  # noqa: F401,F403
