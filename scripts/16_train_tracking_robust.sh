#!/usr/bin/env bash
# 动作模仿研究候选：4096 环境、30000 更新，失败片段自适应采样。
# 无状态估计版本在末尾加 --profile tracking_no_state_estimation_robust_4096。
# 可用 --motion /其他动作.npz 覆盖；已有完整状态模型不能跨版本续训。
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
exec bash "$G1_ROOT/g1.sh" train --profile tracking_robust_4096 \
  --motion "$G1_ROOT/motions/processed/dance_102.npz" --envs "${G1_TRACKING_ENVS:-4096}" \
  --name tracking_robust "$@"
