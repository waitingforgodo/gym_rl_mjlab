#!/usr/bin/env bash
# AMP 舞蹈：4096 环境、30000 次更新，独立于 BeyondMimic tracking 模型。
# 先执行 17_prepare_amp_motion.sh；--iterations/--envs/--motion 可在末尾覆盖。
# 旧 2048 checkpoint 会从运行清单读取嵌入配置；从零训练只使用 4096 profile。
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
exec bash "$G1_ROOT/g1.sh" train --profile amp_dance_4096 \
  --envs "${G1_AMP_ENVS:-4096}" --name amp_dance "$@"
