#!/usr/bin/env bash
# 将 gym_rl 中的 AMP 舞蹈数据按当前 G1 重建为 50 Hz NPZ；仅使用 CPU。
# 已有相同来源的转换结果会校验后复用，不覆盖其他来源的数据。
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
exec bash "$G1_ROOT/g1.sh" amp prepare "$@"
