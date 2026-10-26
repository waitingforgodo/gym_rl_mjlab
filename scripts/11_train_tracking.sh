#!/usr/bin/env bash
# 单独的动作模仿实验；当前 G1 跟踪任务已完成 2048 档并行测试。
# 只覆盖执行时的并行数，保留原 tracking profile 字节以兼容旧模型续训。
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
exec bash "$G1_ROOT/g1.sh" train --profile tracking \
  --motion "$G1_ROOT/motions/processed/dance_102.npz" --envs "${G1_TRACKING_ENVS:-2048}" \
  --iterations 10000 --name dance_102 "$@"
