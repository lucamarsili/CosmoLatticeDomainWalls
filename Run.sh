#!/bin/bash
#SBATCH --account=som
#SBATCH --job-name=DW
#SBATCH --output=./.out
#SBATCH --error=./.err
#SBATCH --qos=s24h
#SBATCH --ntasks=56
#SBATCH --cpus-per-task=1
#SBATCH --time=08:00:00

export PATH=/usr/mpi/gcc/openmpi-4.1.7rc1/bin:$PATH
export LD_LIBRARY_PATH=/usr/mpi/gcc/openmpi-4.1.7rc1/bin:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/lhome/ific/m/marlu/cosmolattice/dependencies/MyLibs/lib:$LD_LIBRARY_PATH
srun ./DWZ3 input=../src/models/parameter-files/DWZ3.in 

