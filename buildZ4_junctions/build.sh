#!/bin/bash
# Build the DWZ4 binary WITH junction-string + wall-velocity diagnostics.
# (junction counter = geodesic plaquette-winding; written to a separate
#  "wall_diagnostics" file on the INFREQUENT output schedule.)
#
# Usage:  ./build.sh            (from anywhere)
set -uo pipefail

module purge 2>/dev/null || true
module load mpi/openmpi-x86_64 2>/dev/null || true
command -v mpicxx >/dev/null || { echo "ERROR: load MPI module first"; exit 1; }

SRC=/mt/home/dpasari/CosmoLattice_Zn
BUILD=$SRC/buildZ4_junctions
FFTW=$SRC/dependencies/MyFFTW3          # local, self-contained FFTW (the old
                                        # CosmoLatticeDomainWalls-main path is gone)

cmake -S "$SRC" -B "$BUILD" -DMODEL=DWZ4 -DMPI=ON \
  -DFFTW_INCLUDES=$FFTW/include \
  -DFFTW_LIB=$FFTW/lib/libfftw3.a \
  -DFFTW_MPI_LIB=$FFTW/lib/libfftw3_mpi.a \
  -DFFTW_THREADS_LIB=$FFTW/lib/libfftw3_threads.a

cmake --build "$BUILD" -j 16
echo "Built: $BUILD/DWZ4"
