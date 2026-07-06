#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# runner_junction.sh  —  Z4 junction-diagnostics scan (2026-06-24)
#
# Runs the 4 cards in src/models/parameter-files/scan_z4_junction/
# (beta4 in {0.05, 0.10} x {physical, PRS}, 1 seed each), using the
# buildZ4_junctions/DWZ4 binary (writes an extra wall_diagnostics output).
# Cards: DWZ4_jct_{phys,prs}_b40p{05,1}_s01.in.
#
# Modeled on buildZ4_stable/runner_phys_stat.sh (same run_job + MAX_JOBS pattern).
#
# Run from:  buildZ4_junctions/   (after `make`, i.e. ./DWZ4 exists)
# Usage:     ./runner_junction.sh [MAX_PARALLEL_JOBS]
#   MAX_JOBS default 1 (one 32-rank job at a time = strictly sequential).
#   Pack the 36-core node with:  MAX_JOBS=2 MPI_PROCS=16 ./runner_junction.sh 2
# Gotcha: benign tachyonic m2<0 warning at init; prints FAILED on ANY non-zero exit
#   (including Ctrl-C, exit 130) — that does NOT necessarily mean a crash.
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail

module purge 2>/dev/null || true
module load mpi/openmpi-x86_64 2>/dev/null || true
command -v mpirun >/dev/null || { echo "ERROR: mpirun not found (load your MPI module first)"; exit 1; }

MAX_JOBS=${1:-1}
MPI_PROCS=${MPI_PROCS:-32}
EXECUTABLE="./DWZ4"
PARAM_DIR="../src/models/parameter-files/scan_z4_junction"
LOG_DIR="logs/junction"

[ -x "$EXECUTABLE" ] || { echo "ERROR: $EXECUTABLE not found (run from buildZ4_junctions/ after make)"; exit 1; }
[ -d "$PARAM_DIR" ]  || { echo "ERROR: $PARAM_DIR missing"; exit 1; }
mkdir -p "$LOG_DIR"

mapfile -t FILES < <(ls -1 "${PARAM_DIR}"/DWZ4_jct_*.in 2>/dev/null | sort)
total=${#FILES[@]}
[ "$total" -gt 0 ] || { echo "ERROR: no cards in $PARAM_DIR"; exit 1; }

echo "════════ Z4 junction-diagnostics scan ════════"
echo " Cards: ${total}   Parallel: ${MAX_JOBS}   MPI/job: ${MPI_PROCS}   Logs: ${LOG_DIR}/"

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
echo "════════ Z4 junction scan: all ${total} runs finished. Logs: ${LOG_DIR}/ ════════"
