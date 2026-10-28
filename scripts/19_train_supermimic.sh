#!/usr/bin/env bash
# SuperMimic-Reproduced：直接使用上游源码和原解释器，不安装/升级依赖。
# 默认 4096 环境；--iterations 现在计总仿真控制步数，UTD=0.4。
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
exec bash "$G1_ROOT/g1.sh" supermimic train "$@"
