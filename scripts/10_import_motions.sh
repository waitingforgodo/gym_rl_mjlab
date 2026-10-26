#!/usr/bin/env bash
# 只复制参考项目两份 CSV；随后在 CPU 转成当前 G1 模型所需的 NPZ。
# 已存在的 NPZ 仅检查、不覆盖；参考路径可在末尾加 --source /路径。
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
if [ "${1:-}" = --help ] || [ "${1:-}" = -h ]; then
  exec bash "$G1_ROOT/g1.sh" import-reference "$@"
fi
bash "$G1_ROOT/g1.sh" import-reference "$@"
for name in dance_102 gangnam_style; do
  if [ -f "$G1_ROOT/motions/processed/$name.npz" ]; then
    bash "$G1_ROOT/g1.sh" inspect-motion "$G1_ROOT/motions/processed/$name.npz"
  else
    bash "$G1_ROOT/g1.sh" prepare-motion "$name"
  fi
done
