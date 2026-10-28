#!/usr/bin/env bash
# 仅选择 InstinctMJ 模型；--profile 可限定 native 任务族。
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
exec bash "$G1_ROOT/g1.sh" instinct play --checkpoint "$G1_CHECKPOINT" "$@"
