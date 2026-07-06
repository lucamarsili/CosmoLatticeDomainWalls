#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# runner_junction_N1024.sh  —  N=1024 physical resolution test (2026-06-25)
#
# Runs the card(s) in src/models/parameter-files/scan_z4_junction_N1024/
# (currently 1: DWZ4_jct_phys_b40p05_s01_N1024.in — β4=0.05, physical, N=1024,
#  identical physics to the N=512 junction twin, only N 512->1024 + dt halved),
# using the buildZ4_junctions/DWZ4 binary (writes wall_diagnostics).
# Tests whether v2_adj saturates once the wall stays resolved (cells/core=1 moves
# a=36 -> a=72): v2 should plateau where the N=512 twin's v2 fell (a in [36,71]).
#
# Same template as runner_junction.sh (run_job + MAX_JOBS pattern). Launch it
# inside a tmux pane and detach — the script blocks on `wait` until done.
#
# Run from:  buildZ4_junctions/   (after `make`, i.e. ./DWZ4 exists)
# Usage:     ./runner_junction_N1024.sh [MAX_PARALLEL_JOBS]
#   MAX_JOBS default 1 (one job at a time).  MPI_PROCS default 32 (32|1024).
#   Faster:  MPI_PROCS=64 ./runner_junction_N1024.sh   (64|1024, all hwthreads).
# Gotcha: benign tachyonic m2<0 warning at init; prints FAILED on ANY non-zero exit
#   (including Ctrl-C, exit 130) — that does NOT necessarily mean a crash.
#   CosmoLattice does NOT mkdir its outputfile dir, so we create it below.
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail

module purge 2>/dev/null || true
module load mpi/openmpi-x86_64 2>/dev/null || true
command -v mpirun >/dev/null || { echo "ERROR: mpirun not found (load your MPI module first)"; exit 1; }

MAX_JOBS=${1:-1}
MPI_PROCS=${MPI_PROCS:-32}
EXECUTABLE="./DWZ4"
PARAM_DIR="../src/models/parameter-files/scan_z4_junction_N1024"
LOG_DIR="logs/junction_N1024"

[ -x "$EXECUTABLE" ] || { echo "ERROR: $EXECUTABLE not found (run from buildZ4_junctions/ after make)"; exit 1; }
[ -d "$PARAM_DIR" ]  || { echo "ERROR: $PARAM_DIR missing"; exit 1; }
mkdir -p "$LOG_DIR"

mapfile -t FILES < <(ls -1 "${PARAM_DIR}"/DWZ4_jct_*.in 2>/dev/null | sort)
total=${#FILES[@]}
[ "$total" -gt 0 ] || { echo "ERROR: no cards in $PARAM_DIR"; exit 1; }

echo "════════ N=1024 physical resolution test ════════"
echo " Cards: ${total}   Parallel: ${MAX_JOBS}   MPI/job: ${MPI_PROCS}   Logs: ${LOG_DIR}/"

for fpath in "${FILES[@]}"; do
    outdir=$(grep '^outputfile' "$fpath" | awk '{print $3}')
    mkdir -p "$outdir"          # CosmoLattice will NOT create this itself
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
echo "════════ N=1024 test: all ${total} run(s) finished. Logs: ${LOG_DIR}/ ════════"
