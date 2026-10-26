#!/usr/bin/env bash
# AMP 舞蹈：2048 环境、30000 次更新，独立于 BeyondMimic tracking 模型。
# 先执行 17_prepare_amp_motion.sh；--iterations/--envs/--motion 可在末尾覆盖。
# 256 档小规模实验可加 --profile amp_dance --envs 256；两档判别器批量不同。
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
exec bash "$G1_ROOT/g1.sh" train --profile amp_dance_2048 \
  --envs "${G1_AMP_ENVS:-2048}" --name amp_dance "$@"
