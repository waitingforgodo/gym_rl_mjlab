#!/usr/bin/env bash
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
exec bash "$G1_ROOT/g1.sh" --python scripts/supermimic/export.py "$@"
