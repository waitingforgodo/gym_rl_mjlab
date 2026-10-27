#!/usr/bin/env bash
# SuperMimic 论文设定：4096 环境；默认动作可用 --motion 覆盖。
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
exec bash "$G1_ROOT/g1.sh" --python scripts/supermimic/train.py --envs 4096 "$@"
