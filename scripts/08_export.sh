#!/usr/bin/env bash
# 加载指定 checkpoint，导出 JIT/ONNX/配置。包含 32 组 JIT 数值一致性检查。
# 导出会短暂创建 1 个 GPU 环境，不做 PPO，也不发送真机指令。
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
exec bash "$G1_ROOT/g1.sh" export --checkpoint "$G1_CHECKPOINT" "$@"
