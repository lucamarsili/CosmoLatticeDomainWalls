#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# run_z3_phys_stat_binfix.sh — rerun the N=512 Z3 *PHYSICAL* stat scan with the
#                             BINNING-FIXED DWZ3 binary, 3 seeds/config.
#
# Pair script: run_z3_prs_stat_binfix.sh (run that on the OTHER machine).
#
# Why: the Z3 sector-binning bug (fieldfunctionals.h, fixed 2026-06-27) only
# corrupted the wall-area counter scal1(->A1) and the velBadj mask. The fix does
# NOT touch the EOM, so energies/spectra/velA/xi reproduce IDENTICALLY; this
# rerun gives a CLEAN scal1/velBadj to confirm the old N=512 A1 was trustworthy.
#
# Subset: 5 beta_N x 3 seeds (s01,s02,s03) = 15 runs, 32 ranks each.
#   Per-run ~1.7-3.7 h.  Output -> NEW *_binfix dir (ORIGINALS PRESERVED).
#
# Usage:   ./run_z3_phys_stat_binfix.sh [MAX_PARALLEL_JOBS]
#   MAX_JOBS default 1 (one 32-rank run at a time). Set 2 only if >=64 cores.
#   Env overrides: RANKS=32 MAX_JOBS=2 OUT_ROOT=/path SEEDS="s01 s02 s03"
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail

SCAN="scan_z3_phys_stat"                      # <<< this script: PHYSICAL only

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # build dir holding DWZ3
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"                         # repo root (parent of build dir)
RANKS=${RANKS:-32}
MAX_JOBS=${1:-${MAX_JOBS:-1}}
SEEDS=${SEEDS:-"s01 s02 s03"}
BIN=${BIN:-"$SCRIPT_DIR/DWZ3"}                # binning-FIXED binary (rebuilt 2026-06-28)
OUT_ROOT=${OUT_ROOT:-/mt/user-batch/dpasari}
LOG_DIR=${LOG_DIR:-"$REPO/logs/${SCAN}_binfix"}

module purge 2>/dev/null || true
module load mpi/openmpi-x86_64 2>/dev/null || true
command -v mpirun >/dev/null || { echo "ERROR: mpirun not found (load your MPI module)"; exit 1; }
[ -x "$BIN" ] || { echo "ERROR: binary not found/executable: $BIN"; exit 1; }
mkdir -p "$LOG_DIR"

echo "════════ Z3 $SCAN binning-fix rerun (3 seeds) ════════"
echo "  Binary : $BIN  ($(stat -c %y "$BIN"))"
echo "  Ranks  : $RANKS    Concurrent: $MAX_JOBS    Seeds: $SEEDS"
echo "  Output : $OUT_ROOT/${SCAN}_binfix/   (originals preserved)"

srcdir="$REPO/src/models/parameter-files/$SCAN"
[ -d "$srcdir" ] || { echo "ERROR: missing $srcdir"; exit 1; }
dstdir="$REPO/src/models/parameter-files/${SCAN}_binfix3"; mkdir -p "$dstdir"

declare -a CARDS=()
for seed in $SEEDS; do
    for src in "$srcdir"/*"${seed}"*.in; do          # seed token anywhere in filename
        [ -e "$src" ] || continue
        base="$(basename "$src")"
        origname="$(basename "$(grep -E '^outputfile' "$src" | awk '{print $3}')")"
        newout="$OUT_ROOT/${SCAN}_binfix/${origname}/"
        dst="$dstdir/$base"
        sed -E "s#^outputfile *=.*#outputfile = ${newout}#" "$src" > "$dst"
        mkdir -p "$newout"; CARDS+=("$dst")
    done
done
total=${#CARDS[@]}
[ "$total" -gt 0 ] || { echo "ERROR: no cards selected"; exit 1; }
echo "  Cards  : $total  (corrected cards in $dstdir/)"
echo "──────────────────────────────────────────────────────"

run_job() {
    local fpath="$1" jobname logfile
    jobname="$(basename "${fpath%.in}")"; logfile="$LOG_DIR/${jobname}.log"
    echo "[$(date '+%H:%M:%S')] START   $jobname"
    if mpirun -n "$RANKS" "$BIN" input="$fpath" >"$logfile" 2>&1; then
        echo "[$(date '+%H:%M:%S')] OK      $jobname"
    else
        echo "[$(date '+%H:%M:%S')] FAILED  $jobname  →  $logfile"
    fi
}
export -f run_job; export RANKS BIN LOG_DIR

submitted=0
for fpath in "${CARDS[@]}"; do
    run_job "$fpath" &
    (( submitted++ )) || true
    while [ "$(jobs -rp | wc -l)" -ge "$MAX_JOBS" ]; do sleep 5; done
    printf "  queued %d/%d  (%s)\n" "$submitted" "$total" "$(basename "$fpath")"
done
wait
echo "════════ done: $total PHYS runs. Output $OUT_ROOT/${SCAN}_binfix/, logs $LOG_DIR/ ════════"
