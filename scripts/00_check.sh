#!/usr/bin/env bash
# 依次检查依赖/资源、CPU 回归测试、4 环境 32 步仿真，不做 PPO 更新。
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
if [ "${1:-}" = --help ] || [ "${1:-}" = -h ]; then
  echo '依次执行 doctor、CPU tests、4 环境 self-check；额外参数交给 self-check。'
  exit 0
fi
bash "$G1_ROOT/g1.sh" doctor
bash "$G1_ROOT/g1.sh" test
exec bash "$G1_ROOT/g1.sh" self-check "$@"
