#!/usr/bin/env bash
# Run the complete Eilon computational campaign from a terminal.
#
# The Python driver checkpoints each completed run in results/eilon/runs.csv.
# Therefore, this script may be stopped and executed again without repeating
# completed experiments.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
TIME_LIMIT="${TIME_LIMIT:-1800}"
RELAXATION_TIME_LIMIT="${RELAXATION_TIME_LIMIT:-120}"
SDP_TIME_LIMIT="${SDP_TIME_LIMIT:-300}"
WARM_SDP_TIME_LIMIT="${WARM_SDP_TIME_LIMIT:-300}"
MIP_GAP="${MIP_GAP:-0.0001}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/results/eilon}"

mkdir -p "$OUTPUT_DIR/logs"
TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
LOG_FILE="$OUTPUT_DIR/logs/campaign_$TIMESTAMP.log"

COMMAND=(
  "$PYTHON_BIN" "$SCRIPT_DIR/run_eilon_experiments.py"
  --instances "$SCRIPT_DIR/instances"
  --output "$OUTPUT_DIR"
  --profile full
  --sizes 20 30 40 50
  --facilities 2 3 5
  --taus 1 1.5 2 3 inf
  --time-limit "$TIME_LIMIT"
  --relaxation-time-limit "$RELAXATION_TIME_LIMIT"
  --sdp-time-limit "$SDP_TIME_LIMIT"
  --warm-sdp-time-limit "$WARM_SDP_TIME_LIMIT"
  --warm-random-starts 10
  --warm-iterations 50
  --mip-gap "$MIP_GAP"
  --seed 0
)

{
  echo "Eilon experiment campaign"
  echo "Started: $(date)"
  echo "Python: $PYTHON_BIN"
  echo "Output: $OUTPUT_DIR"
  echo "Time limit per exact solve: $TIME_LIMIT seconds"
  echo "Command: ${COMMAND[*]}"
  echo
} | tee "$LOG_FILE"

"${COMMAND[@]}" 2>&1 | tee -a "$LOG_FILE"

{
  echo
  echo "Completed: $(date)"
  echo "Results: $OUTPUT_DIR/runs.csv"
  echo "Detailed records: $OUTPUT_DIR/details"
  echo "Figures: $OUTPUT_DIR/figures"
  echo "Log: $LOG_FILE"
} | tee -a "$LOG_FILE"
