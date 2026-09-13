#!/usr/bin/env bash
# One command, everything in order, nothing stops the rest.
#   ./go.sh
# Then paste run.log back to Claude.
cd "$(dirname "$0")"
source .venv/bin/activate
LOG=run.log
: > "$LOG"

step () {
  echo ""                                  | tee -a "$LOG"
  echo "############ $1" | tee -a "$LOG"
  echo "" | tee -a "$LOG"
  shift
  "$@" 2>&1 | tee -a "$LOG"
  echo "---- exit: ${PIPESTATUS[0]} ----" | tee -a "$LOG"
}

step "1/4  tests (2 seconds)" \
     python -m pytest tests/ -q

step "2/4  baseline smoke test, 40 queries" \
     python scripts/run_baseline.py --limit 40

step "3/4  baseline FULL, 215 english queries  <-- the real number" \
     python scripts/run_baseline.py

step "4/4  mjsynth verify only (no extraction yet)" \
     python scripts/prepare_mjsynth.py --verify-only

echo "" | tee -a "$LOG"
echo "ALL DONE. Paste run.log to Claude." | tee -a "$LOG"
