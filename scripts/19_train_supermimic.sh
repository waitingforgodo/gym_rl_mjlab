#!/usr/bin/env bash
# SuperMimic-Reproduced：直接使用上游源码和原解释器，不安装/升级依赖。
# 本机默认：4096 环境、batch=512、H=16、eager、LAFAN1 40 段。
# 显式 --strict-upstream / --config / --resume 时，直接使用调用者设置。
# --iterations 计目标总仿真控制步数，UTD=0.4，不是 PPO 更新数。
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
for argument in "$@"; do
  case "$argument" in
    --strict-upstream|--config|--config=*|--resume|--resume=*)
      exec bash "$G1_ROOT/g1.sh" supermimic train "$@"
      ;;
  esac
done
exec bash "$G1_ROOT/g1.sh" supermimic train \
  --config "$G1_ROOT/workspace/supermimic/profiles/lafan1_4096_h16_eager.json" \
  --allow-config-changes "$@"
