#!/usr/bin/env bash
# 导出包含归一化器的 JIT/ONNX，做真实观测数值一致性检查。
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
exec bash "$G1_ROOT/g1.sh" instinct export --checkpoint "$G1_CHECKPOINT" "$@"
