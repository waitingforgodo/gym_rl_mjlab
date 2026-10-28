#!/usr/bin/env bash
# 所有任务的底层启动器：切到正确目录、选择固定解释器、隔离缓存、限制 CPU。
# 本脚本通过 bash 作为子进程执行，export/unset 不会修改父终端的环境。
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
[ -f "$ROOT/.installed-workspace" ] || { echo 'Run setup.sh first.' >&2; exit 1; }
RUNTIME=$(cat "$ROOT/.runtime-path")
# 防止误用用户正在做其他工作的 Conda、PYTHONPATH 或 Python 标准库路径。
unset VIRTUAL_ENV CONDA_PREFIX PYTHONPATH PYTHONHOME
export XDG_CACHE_HOME="$ROOT/.cache"
export TORCH_HOME="$ROOT/.cache/torch"
export HF_HOME="$ROOT/.cache/huggingface"
export CUDA_CACHE_PATH="$ROOT/.cache/cuda"
export WANDB_MODE=disabled
export WANDB_DIR="$ROOT/outputs"
export WANDB_CACHE_DIR="$ROOT/.cache/wandb"
# nice 仅降低 CPU 调度优先级，线程数仅约束相关 CPU 库；都不是 GPU 配额。
export OMP_NUM_THREADS="${G1_CPU_THREADS:-2}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export TMPDIR="$ROOT/.cache/tmp"
export MPLCONFIGDIR="$ROOT/.cache/matplotlib"
export NUMBA_CACHE_DIR="$ROOT/.cache/numba"
# 动作转换用 CPU 正运动学；纯文件导入与检查也不需要访问显卡。
if [ "${1:-}" = prepare-motion ] || [ "${1:-}" = prepare-amp-motion ] || [ "${1:-}" = import-reference ] || [ "${1:-}" = inspect-motion ]; then
  export CUDA_VISIBLE_DEVICES=""
fi
# 按方法的数据转换也仅使用 CPU。
if { [ "${1:-}" = amp ] || [ "${1:-}" = beyondmimic ] || [ "${1:-}" = instinct ]; } && [ "${2:-}" = prepare ]; then
  export CUDA_VISIBLE_DEVICES=""
fi
if [ "${1:-}" = supermimic ] && [ "${2:-}" = prepare ]; then
  export CUDA_VISIBLE_DEVICES=""
fi
if [ "${1:-}" = instinct ] && { [ "${2:-}" = prepare-lafan ] || [ "${2:-}" = prepare-scene ]; }; then
  export CUDA_VISIBLE_DEVICES=""
fi
cd "$ROOT"
mkdir -p "$ROOT/.cache/cuda" "$TMPDIR"
# "$@" 保留每个参数的边界，支持带空格的路径，不使用 eval 拼命令。
if [ "${1:-}" = --python ]; then
  # 独立程序仍复用固定解释器和同一套缓存、线程设置，无需再次安装包。
  # 示例：bash g1.sh --python scripts/reinforcement_learning/rsl_rl/train.py --help
  shift
  [ "$#" -gt 0 ] || { echo 'Usage: g1.sh --python SCRIPT.py [arguments...]' >&2; exit 2; }
  case "$1" in
    */import_motions.py|*/prepare_motion.py|*/prepare_amp_motion.py) export CUDA_VISIBLE_DEVICES="" ;;
  esac
  exec nice -n 10 "$RUNTIME/.venv/bin/python" "$@"
fi
exec nice -n 10 "$RUNTIME/.venv/bin/python" -m workspace.cli "$@"
