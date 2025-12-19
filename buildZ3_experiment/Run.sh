#!/bin/bash
#SBATCH --account=som
#SBATCH --job-name=sim
#SBATCH --output=./.out
#SBATCH --error=./.err
#SBATCH --qos=mpi
#SBATCH --ntasks=128
#SBATCH --cpus-per-task=1
#SBATCH --time=20:00:00

export PATH=/usr/mpi/gcc/openmpi-4.1.7rc1/bin:$PATH
export LD_LIBRARY_PATH=/usr/mpi/gcc/openmpi-4.1.7rc1/bin:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/lhome/ific/m/marlu/cosmolattice/dependencies/MyLibs/lib:$LD_LIBRARY_PATH
srun ./DWZ3 input=../src/models/parameter-files/DWZ3.in 

