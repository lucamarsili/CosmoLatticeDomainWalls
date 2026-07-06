#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# runner_z4_ext.sh  —  Z4 EXTREME-beta scan (2026-06-24)
#
# Extends the {0.1..0.9} grid to the extremes beta4 in {0.005, 0.02, 0.95, 0.98},
# 3 seeds (s01-s03, paired with the existing stat scans), PHYSICAL + PRS.
# Cards:  ../src/models/parameter-files/scan_z4_phys_ext/  (physical, prsWall=false)
#         ../src/models/parameter-files/scan_z4_prs_ext/   (PRS, prsWall=true)
#
# Uses the JUNCTION binary (buildZ4_junctions/DWZ4) so every run ALSO writes the
# extra wall_diagnostics file (junction-string + wall-velocity diagnostics) for
# these "chosen" extreme-beta points — superset of the standard output, same physics.
#
# Modeled on buildZ4_stable/runner_phys_stat.sh (same run_job + MAX_JOBS pattern).
#
# Run from:  buildZ4_junctions/
# Usage:     ./runner_z4_ext.sh [MAX_PARALLEL_JOBS]      (default 1 = sequential)
#   SCAN=all|phys|prs  selects which family to run (default all).
#   Pack the 36-core node:  SCAN=phys MAX_JOBS=2 MPI_PROCS=16 ./runner_z4_ext.sh 2
# Gotcha: benign tachyonic m2<0 warning at init; FAILED is printed on ANY non-zero
#   exit (incl. Ctrl-C) — not necessarily a crash. High-beta physical may blow up
#   (steep potential); that is the experiment.
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail

module purge 2>/dev/null || true
module load mpi/openmpi-x86_64 2>/dev/null || true
command -v mpirun >/dev/null || { echo "ERROR: mpirun not found (load your MPI module first)"; exit 1; }

MAX_JOBS=${1:-1}
MPI_PROCS=${MPI_PROCS:-32}
SCAN=${SCAN:-all}
EXECUTABLE="./DWZ4"                                  # junction build (writes wall_diagnostics)
PHYS_DIR="../src/models/parameter-files/scan_z4_phys_ext"
PRS_DIR="../src/models/parameter-files/scan_z4_prs_ext"
LOG_DIR="logs/z4_ext"

[ -x "$EXECUTABLE" ] || { echo "ERROR: $EXECUTABLE not found (run from buildZ4_junctions/ after make)"; exit 1; }
mkdir -p "$LOG_DIR"

DIRS=()
case "$SCAN" in
    all)  DIRS=("$PHYS_DIR" "$PRS_DIR") ;;
    phys) DIRS=("$PHYS_DIR") ;;
    prs)  DIRS=("$PRS_DIR") ;;
    *)    echo "ERROR: SCAN must be all|phys|prs (got '$SCAN')"; exit 1 ;;
esac

FILES=()
for d in "${DIRS[@]}"; do
    [ -d "$d" ] || { echo "ERROR: $d missing"; exit 1; }
    while IFS= read -r f; do FILES+=("$f"); done < <(ls -1 "$d"/*.in 2>/dev/null | sort)
done
total=${#FILES[@]}
[ "$total" -gt 0 ] || { echo "ERROR: no cards found (SCAN=$SCAN)"; exit 1; }

echo "════════ Z4 EXTREME-beta scan (junction binary) ════════"
echo " SCAN=${SCAN}   Cards: ${total}   Parallel: ${MAX_JOBS}   MPI/job: ${MPI_PROCS}   Logs: ${LOG_DIR}/"

for fpath in "${FILES[@]}"; do
    outdir=$(grep '^outputfile' "$fpath" | awk '{print $3}')
    mkdir -p "$outdir"
done

run_job() {
    local fpath="$1" jobname logfile
    jobname=$(basename "${fpath%.in}")
    logfile="${LOG_DIR}/${jobname}.log"
    echo "[$(date '+%H:%M:%S')] START   ${jobname}"
    if mpirun -n "${MPI_PROCS}" "${EXECUTABLE}" input="${fpath}" >"${logfile}" 2>&1; then
        echo "[$(date '+%H:%M:%S')] OK      ${jobname}"
    else
        echo "[$(date '+%H:%M:%S')] FAILED  ${jobname}  →  ${logfile}"
    fi
}
export -f run_job
export MPI_PROCS EXECUTABLE LOG_DIR

submitted=0
for fpath in "${FILES[@]}"; do
    run_job "$fpath" &
    (( submitted++ )) || true
    while [ "$(jobs -rp | wc -l)" -ge "${MAX_JOBS}" ]; do sleep 5; done
    printf "  Queued %d/%d  (%s)\n" "$submitted" "$total" "$(basename "${fpath}")"
done
wait
echo "════════ Z4 extreme scan: all ${total} runs finished. Logs: ${LOG_DIR}/ ════════"
