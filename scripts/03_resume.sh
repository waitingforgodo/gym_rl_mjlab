#!/usr/bin/env bash
# 续训本工作空间最近运行中的最高数字模型，默认新增 1000 次 PPO 更新。
# 每次写新目录。混用多个任务时，应显式传 --checkpoint /具体运行目录。
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
exec bash "$G1_ROOT/g1.sh" resume --checkpoint "$G1_CHECKPOINT" --envs "$G1_ENV_COUNT" \
  --iterations 1000 --name g1_2048_continue "$@"
