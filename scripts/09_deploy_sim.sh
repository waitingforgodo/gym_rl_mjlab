#!/usr/bin/env bash
# 只加载导出包内的 JIT 策略做同后端仿真回归，必须指定 --bundle /完整目录。
# --headless --seconds 60 可输出数值评估，不打开浏览器。
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
exec bash "$G1_ROOT/g1.sh" deploy-sim "$@"
