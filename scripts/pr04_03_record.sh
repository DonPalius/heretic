#!/usr/bin/env bash
# Launch only an explicitly configured single recorded trial; never select experiments here.
set -euo pipefail
heretic_check=0
heretic_offline=1
for heretic_arg in "$@"; do
    case "$heretic_arg" in
        --check) heretic_check=1 ;;
        --download) heretic_offline=0 ;;
        -h|--help)
            echo "Usage: bash scripts/pr04_03_record.sh [--check] [--download]"
            echo "Reads this Heretic repository's config.toml; --check validates without running a trial."
            exit 0 ;;
        *) echo "Unknown argument: $heretic_arg" >&2; exit 2 ;;
    esac
done
heretic_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$heretic_root"
if [[ ! -x .venv/bin/python ]]; then
    echo "Prepare the separate Heretic environment first (uv sync --python 3.11)." >&2
    exit 1
fi
export PYTHONPATH="$heretic_root/src"
export HF_HUB_OFFLINE="$heretic_offline"
export HF_DATASETS_OFFLINE="$heretic_offline"
.venv/bin/python - <<'PY'
import json
from heretic.ara_audit import validate_recording_config
print(json.dumps(validate_recording_config(), indent=2))
PY
if [[ "$heretic_check" == 1 ]]; then
    exit 0
fi
exec .venv/bin/heretic
