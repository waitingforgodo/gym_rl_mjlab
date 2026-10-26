#!/usr/bin/env bash
# 读取 TensorBoard 最后指标和资源采样，汇总运行情况；不加载仿真或策略。
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
exec bash "$G1_ROOT/g1.sh" summary --run "$G1_CHECKPOINT" "$@"
