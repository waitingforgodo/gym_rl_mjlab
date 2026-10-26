#!/usr/bin/env bash
# 浏览器 3D 回放。打开 http://localhost:8080；终端 Ctrl+C 停止回放。
# --checkpoint 可指定 .pt 或整个运行目录；--port 可更换占用的端口。
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
exec bash "$G1_ROOT/g1.sh" play --checkpoint "$G1_CHECKPOINT" --viewer viser --port 8080 "$@"
