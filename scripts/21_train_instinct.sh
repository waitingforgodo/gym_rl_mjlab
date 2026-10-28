#!/usr/bin/env bash
# 新 InstinctMJ 后端：默认行走、4096 环境；--profile beyondmimic/shadowing 可覆盖。
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
exec bash "$G1_ROOT/g1.sh" instinct train --profile locomotion --envs "$G1_ENV_COUNT" "$@"
