#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# runner_Z4.sh  —  run all files from the beta/kCutOff/H0 parameter scan
#
# Run from:  build_Z4/
# Usage:     ./runner_Z4.sh [MAX_PARALLEL_JOBS]
#
# Each job calls:  mpirun -n MPI_PROCS ./DWZ4 input=<file>
# Default parallel jobs = 2  (= 64 MPI ranks total with MPI_PROCS=32).
# Raise MAX_JOBS if your node has more cores / you are using a scheduler.
#
# Logs:  logs/scan/<jobname>.log
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail

MAX_JOBS=${1:-2}        # parallel simulations
MPI_PROCS=32            # MPI ranks per simulation
EXECUTABLE="./DWZ4"
PARAM_DIR="../src/models/parameter-files/scan_z4_targeted"
LOG_DIR="logs/scan"

# ── Sanity checks ─────────────────────────────────────────────────────────────
if [ ! -x "$EXECUTABLE" ]; then
    echo "ERROR: $EXECUTABLE not found or not executable (run from build_Z4/)"
    exit 1
fi

mkdir -p "$LOG_DIR"

mapfile -t FILES < <(ls "${PARAM_DIR}"/DWZ4*.in 2>/dev/null | sort)

if [ ${#FILES[@]} -eq 0 ]; then
    echo "ERROR: no .in files found in ${PARAM_DIR}"
    echo "Run analysis/parameter_file_creator_Z4.ipynb first to generate scan files."
    exit 1
fi

total=${#FILES[@]}
echo "══════════════════════════════════════════════════════"
echo " Parameter scan runner"
echo " Files:          ${total}"
echo " Parallel jobs:  ${MAX_JOBS}"
echo " MPI ranks/job:  ${MPI_PROCS}"
echo " Logs:           ${LOG_DIR}/"
echo "══════════════════════════════════════════════════════"
echo ""

# ── Create all output directories upfront ────────────────────────────────────
echo "Creating output directories..."
for fpath in "${FILES[@]}"; do
    outdir=$(grep '^outputfile' "$fpath" | awk '{print $3}')
    mkdir -p "$outdir"
done
echo "Done."
echo ""

# ── Job function ──────────────────────────────────────────────────────────────
run_job() {
    local fpath="$1"
    local jobname
    jobname=$(basename "${fpath%.in}")
    local logfile="${LOG_DIR}/${jobname}.log"

    echo "[$(date '+%H:%M:%S')] START   ${jobname}"
    if mpirun -n "${MPI_PROCS}" "${EXECUTABLE}" input="${fpath}" \
            >"${logfile}" 2>&1; then
        echo "[$(date '+%H:%M:%S')] OK      ${jobname}"
    else
        echo "[$(date '+%H:%M:%S')] FAILED  ${jobname}  →  ${logfile}"
    fi
}

export -f run_job
export MPI_PROCS EXECUTABLE LOG_DIR

# ── Dispatch ──────────────────────────────────────────────────────────────────
submitted=0
for fpath in "${FILES[@]}"; do
    run_job "$fpath" &
    (( submitted++ )) || true

    # throttle: wait until a slot is free
    while [ "$(jobs -rp | wc -l)" -ge "${MAX_JOBS}" ]; do
        sleep 5
    done

    printf "  Queued %d/%d  (%s)\n" "$submitted" "$total" "$(basename "${fpath}")"
done

wait   # all background jobs

echo ""
echo "══════════════════════════════════════════════════════"
printf " All %d runs finished.\n" "$total"
echo " Logs in: ${LOG_DIR}/"
echo "══════════════════════════════════════════════════════"
