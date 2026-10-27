#!/usr/bin/env bash
# 顺序短测，不并发抢显卡；每档 100 次更新，结果不会进入正式模型 latest。
# 例如在末尾加 --envs 2048 4096 --iterations 200，可覆盖本脚本默认值。
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
exec bash "$G1_ROOT/g1.sh" benchmark --envs 1024 2048 4096 --iterations 100 "$@"
