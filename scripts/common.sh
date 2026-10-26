#!/usr/bin/env bash
# 所有快捷脚本共用的定位逻辑；本文件由 source 加载，不单独启动任务。
# 根据脚本自身路径找根目录，因此无论当前终端在哪个目录都能使用。
set -euo pipefail
G1_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
[ -f "$G1_ROOT/.installed-workspace" ] || {
  echo '请先在 Ubuntu 执行 setup.sh 或 apply_update.sh，再运行安装目录中的脚本。' >&2
  exit 1
}
# 2048 来自这台电脑 velocity_flat 的实测，不自动用于动作跟踪任务。
G1_ENV_COUNT="${G1_ENV_COUNT:-2048}"
G1_CHECKPOINT="${G1_CHECKPOINT:-latest}"
