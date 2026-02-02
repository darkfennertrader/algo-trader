#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/show_return_tensor.sh [--head N] [DATA_CLEANING_DIR]

If DATA_CLEANING_DIR is omitted, the script reads DATA_LAKE_SOURCE and uses
the latest YYYY-WW directory within it.

Defaults:
  --head 5
USAGE
}

head_count=5
target_dir=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --head)
      if [[ -z "${2:-}" ]]; then
        echo "--head requires a value" >&2
        exit 1
      fi
      head_count="$2"
      shift 2
      ;;
    *)
      if [[ -n "$target_dir" ]]; then
        echo "Unexpected argument: $1" >&2
        usage
        exit 1
      fi
      target_dir="$1"
      shift
      ;;
  esac
done

resolve_latest_dir() {
  local root="$1"
  if [[ ! -d "$root" ]]; then
    echo "DATA_LAKE_SOURCE is not a directory: $root" >&2
    exit 1
  fi
  local latest
  latest=$(ls -1 "$root" 2>/dev/null | grep -E '^[0-9]{4}-[0-9]{2}$' | sort | tail -n 1 || true)
  if [[ -z "$latest" ]]; then
    echo "No YYYY-WW directories found under: $root" >&2
    exit 1
  fi
  echo "$root/$latest"
}

if [[ -z "$target_dir" ]]; then
  if [[ -z "${DATA_LAKE_SOURCE:-}" ]]; then
    fallback_root="/home/ray/projects/data_sources/data_lake"
    target_dir="$(resolve_latest_dir "$fallback_root")"
  else
    target_dir="$(resolve_latest_dir "$DATA_LAKE_SOURCE")"
  fi
fi

tensor_path="$target_dir/return_tensor.pt"
if [[ ! -f "$tensor_path" ]]; then
  echo "return_tensor.pt not found at: $tensor_path" >&2
  exit 1
fi

tensor_path="$tensor_path" head_count="$head_count" uv run python - <<'PY'
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import torch

tensor_path = Path(os.environ["tensor_path"])
head_count = int(os.environ.get("head_count", "0"))
payload = torch.load(tensor_path, map_location="cpu")
metadata_path = tensor_path.parent / "tensor_metadata.json"
metadata = {}
if metadata_path.exists():
    try:
        metadata = json.loads(metadata_path.read_text())
    except json.JSONDecodeError:
        metadata = {}

def epoch_hours_to_iso(value: int) -> str:
    seconds = int(value) * 3600
    return datetime.fromtimestamp(seconds, tz=timezone.utc).isoformat()

print(f"path: {tensor_path}")
print(f"keys: {sorted(payload.keys())}")

values = payload.get("values")
timestamps = payload.get("timestamps")
missing_mask = payload.get("missing_mask")
tensor_meta = metadata.get("tensor", {}) if isinstance(metadata, dict) else {}
assets = tensor_meta.get("assets")
scale = tensor_meta.get("scale", 1)
if assets:
    print(f"assets: {assets}")
if scale:
    print(f"scale: {scale}")

if values is not None:
    print(f"values.shape: {tuple(values.shape)}")
    print(f"values.dtype: {values.dtype}")
if timestamps is not None:
    print(f"timestamps.shape: {tuple(timestamps.shape)}")
    print(f"timestamps.dtype: {timestamps.dtype}")
    if timestamps.numel():
        first = int(timestamps.min().item())
        last = int(timestamps.max().item())
        print(f"timestamps.min_epoch_hours: {first}")
        print(f"timestamps.max_epoch_hours: {last}")
        print(f"timestamps.min_utc: {epoch_hours_to_iso(first)}")
        print(f"timestamps.max_utc: {epoch_hours_to_iso(last)}")
if missing_mask is not None:
    print(f"missing_mask.shape: {tuple(missing_mask.shape)}")
    print(f"missing_mask.dtype: {missing_mask.dtype}")
    if missing_mask.numel():
        missing = int(missing_mask.sum().item())
        total = int(missing_mask.numel())
        print(f"missing_mask.count: {missing} / {total}")

if head_count > 0 and values is not None and timestamps is not None:
    print(f"\nhead_rows: {head_count}")
    row_count = min(head_count, values.shape[0])
    value_rows = values[:row_count].cpu().numpy()
    time_rows = timestamps[:row_count].cpu().numpy()
    mask_rows = (
        missing_mask[:row_count].cpu().numpy()
        if missing_mask is not None
        else None
    )
    for i in range(row_count):
        ts_value = int(time_rows[i])
        ts_iso = epoch_hours_to_iso(ts_value)
        row_values = value_rows[i].astype(float)
        if scale:
            row_values = row_values / float(scale)
        if mask_rows is not None:
            row_missing = mask_rows[i]
            row_values = [
                None if bool(is_missing) else float(value)
                for value, is_missing in zip(
                    row_values, row_missing, strict=False
                )
            ]
        else:
            row_values = [float(value) for value in row_values]
        print(f"{i}: {ts_value} {ts_iso} {row_values}")
PY
