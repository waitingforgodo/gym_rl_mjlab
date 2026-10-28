#!/usr/bin/env bash
# 独立仿真部署，传 --bundle；--engine jit/onnx，--headless --steps 500。
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
exec bash "$G1_ROOT/g1.sh" instinct deploy "$@"
