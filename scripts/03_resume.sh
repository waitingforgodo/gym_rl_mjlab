#!/usr/bin/env bash
# 续训默认继承模型保存的并行数和种子，新增 1000 次 PPO 更新。
# 在 common.sh 填入平地默认值前，记住用户是否显式设置了 G1_ENV_COUNT。
G1_RESUME_ENVS="${G1_ENV_COUNT:-}"
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
G1_RESUME_ARGS=(resume --checkpoint "$G1_CHECKPOINT" --iterations 1000 --name g1_continue)
if [ -n "$G1_RESUME_ENVS" ]; then
  G1_RESUME_ARGS+=(--envs "$G1_RESUME_ENVS")
fi
# 命令末尾的 --envs/--seed 仍可显式覆盖；旧实验不需要修改 profile 文件。
exec bash "$G1_ROOT/g1.sh" "${G1_RESUME_ARGS[@]}" "$@"
