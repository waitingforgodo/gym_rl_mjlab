#!/usr/bin/env bash
# 在单独终端运行：只读取 event 文件，不启动训练，不安装 TensorFlow。
# Windows 浏览器打开 http://localhost:6006。关闭本服务不停止另一个终端的训练。
# 可以用 --port 6007，或 --logdir /其他日志根目录 覆盖默认值。
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
G1_RUNTIME=$(cat "$G1_ROOT/.runtime-path")
export CUDA_VISIBLE_DEVICES=""  # 即使安装了 TensorFlow，也不给网页服务使用 GPU。
export OMP_NUM_THREADS="${G1_CPU_THREADS:-2}"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
exec nice -n 10 "$G1_RUNTIME/.venv/bin/python" -m tensorboard.main \
  --logdir "$G1_ROOT/outputs/runs" --host 127.0.0.1 --port 6006 "$@"
