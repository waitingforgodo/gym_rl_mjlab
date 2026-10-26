#!/usr/bin/env bash
# 仅用于从旧隔离环境迁入 g1_continue；已经迁入后，请使用 03_resume.sh。
# --legacy-velocity 表示采用同版本官方平地 G1 假设，无法追溯验证旧奖励改动。
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
exec bash "$G1_ROOT/g1.sh" resume --checkpoint g1_continue --legacy-velocity \
  --envs "$G1_ENV_COUNT" --iterations 1000 --name imported_continue "$@"
