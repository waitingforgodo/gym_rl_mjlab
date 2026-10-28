#!/usr/bin/env bash
# 单独的动作模仿实验；当前 G1 跟踪任务已完成 4096 档并行容量测试。
# 使用独立 profile，保留旧 tracking profile 字节以兼容旧模型续训。
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
exec bash "$G1_ROOT/g1.sh" beyondmimic train --profile tracking_4096 \
  --motion "$G1_ROOT/motions/processed/dance_102.npz" --envs "${G1_TRACKING_ENVS:-4096}" \
  --iterations 10000 --name dance_102 "$@"
