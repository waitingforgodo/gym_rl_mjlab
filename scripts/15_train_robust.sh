#!/usr/bin/env bash
# 新的平地研究候选：2048 环境、10000 更新，独立 profile 和运行目录。
# 课程、失败/下肢代价在 workspace/recipes.py；PPO 数值与理由见 profile。
# 比较用旧基线仍运行 02_train.sh；此处不加载旧模型的优化器。
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
exec bash "$G1_ROOT/g1.sh" train --profile velocity_robust --envs "$G1_ENV_COUNT" \
  --name g1_robust "$@"
