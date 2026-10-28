#!/usr/bin/env bash
# 从头建立平地速度跟踪实验，不读取已有策略；3000 次更新是观察节点。
# 训练种子默认 42，PPO/奖励等保持 velocity_flat 配置，可在参数末尾覆盖。
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
exec bash "$G1_ROOT/g1.sh" velocity train --profile velocity_flat_4096 --envs "$G1_ENV_COUNT" \
  --iterations 3000 --name g1_flat_4096 "$@"
