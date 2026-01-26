#!/usr/bin/env bash
set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found in PATH"
  exit 1
fi

uv run python - <<'PY'
import torch

available = torch.cuda.is_available()
print(
    f"torch={torch.__version__} "
    f"cuda_runtime={torch.version.cuda} "
    f"cuda_available={available}"
)
if available:
    print(f"device={torch.cuda.get_device_name(0)}")
PY
