#!/usr/bin/env bash
# 从本地标准 NPZ 转为 Instinct torso-root 数据，包含 waist 反向链校正和 FK 检查。
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
exec bash "$G1_ROOT/g1.sh" instinct prepare "$@"
