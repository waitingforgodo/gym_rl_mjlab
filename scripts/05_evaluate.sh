#!/usr/bin/env bash
# 使用官方 play 的随机命令，16 环境各观察 60 秒，失败后仍保留自动复位。
# JSON 同时记录摔倒事件、受影响环境、无失败比例、RMSE 和实际命令协议。
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
exec bash "$G1_ROOT/g1.sh" evaluate --checkpoint "$G1_CHECKPOINT" --envs 16 --seconds 60 "$@"
