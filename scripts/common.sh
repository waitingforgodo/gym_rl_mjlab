#!/usr/bin/env bash
# 所有快捷脚本共用的定位逻辑；本文件由 source 加载，不单独启动任务。
# 根据脚本自身路径找根目录，因此无论当前终端在哪个目录都能使用。
set -euo pipefail
G1_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
[ -f "$G1_ROOT/.installed-workspace" ] || {
  echo '请先在 Ubuntu 执行 setup.sh 或 apply_update.sh，再运行安装目录中的脚本。' >&2
  exit 1
}
# 4096 已在当前 RTX 5070 Laptop 上分别通过 AMP、完整状态跟踪和
# 无状态估计跟踪的 30 更新容量短测；环境变量仍可显式降低规模。
G1_ENV_COUNT="${G1_ENV_COUNT:-4096}"
G1_CHECKPOINT="${G1_CHECKPOINT:-latest}"
