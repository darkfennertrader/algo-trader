#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
CONFIG_PATH="${REPO_ROOT}/config/tickers.yml"
RUN_CMD=(uv run algotrader historical)
START_TIME_EPOCH="$(date +%s)"
START_TIME_FMT="$(date '+%Y-%m-%d_%H:%M:%S')"

print_timing() {
  local end_time_epoch
  local end_time_fmt
  local elapsed_seconds
  end_time_epoch="$(date +%s)"
  end_time_fmt="$(date '+%Y-%m-%d_%H:%M:%S')"
  elapsed_seconds=$((end_time_epoch - START_TIME_EPOCH))
  echo "Start: ${START_TIME_FMT}"
  echo "End: ${end_time_fmt}"
  echo "Elapsed seconds: ${elapsed_seconds}"
}

usage() {
  echo "Usage: $0 <start_year> <end_year>" >&2
  echo "Example: $0 2024 2025" >&2
}

validate_year() {
  local value="$1"
  [[ "${value}" =~ ^[0-9]{4}$ ]]
}

if [[ "$#" -ne 2 ]]; then
  usage
  exit 1
fi

start_year="$1"
end_year="$2"
if ! validate_year "${start_year}" || ! validate_year "${end_year}"; then
  echo "Years must be 4-digit numbers." >&2
  usage
  exit 1
fi

start_year_num=$((10#${start_year}))
end_year_num=$((10#${end_year}))
if (( start_year_num > end_year_num )); then
  echo "Start year must be less than or equal to end year." >&2
  usage
  exit 1
fi

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Missing config: ${CONFIG_PATH}" >&2
  exit 1
fi

update_month() {
  local month_value="$1"
  local tmp_file
  tmp_file="$(mktemp "${CONFIG_PATH}.XXXXXX")"
  if ! awk -v month_value="${month_value}" '
    BEGIN { replaced = 0 }
    {
      if (!replaced && $0 ~ /^month:/) {
        line = $0
        comment = ""
        hash = index(line, "#")
        if (hash > 0) {
          comment = substr(line, hash)
        }
        if (comment != "") {
          print "month: \"" month_value "\"", comment
        } else {
          print "month: \"" month_value "\""
        }
        replaced = 1
        next
      }
      print
    }
    END { if (!replaced) exit 1 }
  ' "${CONFIG_PATH}" > "${tmp_file}"; then
    rm -f "${tmp_file}"
    return 1
  fi
  mv "${tmp_file}" "${CONFIG_PATH}"
}

failures=0
for ((year=start_year_num; year<=end_year_num; year++)); do
  for month in 01 02 03 04 05 06 07 08 09 10 11 12; do
    current="${year}-${month}"
    echo "Downloading ${current} at $(date '+%Y-%m-%d %H:%M:%S')"
    if ! update_month "${current}"; then
      echo "Failed to update month in ${CONFIG_PATH}" >&2
      failures=$((failures + 1))
      continue
    fi
    if ! "${RUN_CMD[@]}"; then
      echo "Download failed for ${current}" >&2
      failures=$((failures + 1))
    fi
done
done

print_timing

if [[ "${failures}" -gt 0 ]]; then
  echo "Completed with ${failures} failure(s)." >&2
  exit 1
fi
