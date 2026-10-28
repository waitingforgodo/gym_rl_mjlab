#!/usr/bin/env bash
# 在单独终端运行：只读取 event 文件，不启动训练，不安装 TensorFlow。
# Windows 浏览器打开 http://localhost:6006。关闭本服务不停止另一个终端的训练。
# 可以用 --port 6007，或 --logdir /其他日志根目录 覆盖默认值。
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
G1_RUNTIME=$(cat "$G1_ROOT/.runtime-path")

# 提前输出浏览器地址，避免被后续 TensorFlow/数据加载提示淹没。
# 只读取参数副本；原来的 "$@" 仍完整传给 TensorBoard 做正式解析。
# 同时支持 --port 6007 和 --port=6007，重复指定时采用最后一个值。
G1_TB_PORT=6006
G1_TB_NEXT_PORT=0
G1_TB_HELP=0
for G1_TB_ARG in "$@"; do
  if [ "$G1_TB_NEXT_PORT" = 1 ]; then
    G1_TB_PORT="$G1_TB_ARG"
    G1_TB_NEXT_PORT=0
    continue
  fi
  case "$G1_TB_ARG" in
    --port) G1_TB_NEXT_PORT=1 ;;
    --port=*) G1_TB_PORT="${G1_TB_ARG#--port=}" ;;
    -h|--help|--helpfull|--helpshort) G1_TB_HELP=1 ;;
  esac
done
if [ "$G1_TB_HELP" = 0 ]; then
  if [ "$G1_TB_PORT" = 0 ]; then
    # 端口 0 由系统自动选择，实际链接由 TensorBoard 启动后输出。
    printf '\nTensorBoard 正在启动，自动端口的网页地址请看后续输出。\n\n'
  else
    printf '\nTensorBoard 正在启动，网页地址：\nhttp://localhost:%s/\n' "$G1_TB_PORT"
    printf '服务就绪后，在 VS Code 终端按住 Ctrl 点击链接访问。\n\n'
  fi
fi

export CUDA_VISIBLE_DEVICES=""  # 即使安装了 TensorFlow，也不给网页服务使用 GPU。
export OMP_NUM_THREADS="${G1_CPU_THREADS:-2}"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
exec nice -n 10 "$G1_RUNTIME/.venv/bin/python" -m tensorboard.main \
  --logdir "$G1_ROOT/outputs" --host 127.0.0.1 --port 6006 "$@"
