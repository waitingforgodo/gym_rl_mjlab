#!/usr/bin/env bash
# 打印本次运行真实保存的 manifest、agent.yaml 和 env.yaml，不靠默认值猜配置。
# 训练文件中的 !!python/name 等 YAML 标签按文本展示，不执行或反序列化这些标签。
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
exec bash "$G1_ROOT/g1.sh" show-config --checkpoint "$G1_CHECKPOINT" "$@"
