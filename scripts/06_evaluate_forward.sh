#!/usr/bin/env bash
# 基础前进测试：机体系 vx=0.5 m/s、vy=0、yaw_rate=0 rad/s，从静止起步。
# 不混入随机站立/航向/世界系命令；不改变训练参数，不适用于动作跟踪策略。
# 可覆盖 --velocity-cmd 1.0 0 0 或 --seconds 120，比较时两模型须用相同参数。
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
exec bash "$G1_ROOT/g1.sh" evaluate --checkpoint "$G1_CHECKPOINT" --envs 16 --seconds 60 \
  --velocity-cmd 0.5 0 0 "$@"
