#!/usr/bin/env bash
# 单独的动作模仿实验；平地行走的 2048 实测结论不能直接迁移过来。
# 跟踪暂用 256 环境，待 profile=tracking 的 benchmark 实测后再增加。
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
exec bash "$G1_ROOT/g1.sh" train --profile tracking \
  --motion "$G1_ROOT/motions/processed/dance_102.npz" --envs "${G1_TRACKING_ENVS:-256}" \
  --iterations 10000 --name dance_102 "$@"
