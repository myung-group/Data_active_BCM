#!/bin/sh
#PBS -V
#PBS -N MLMD_800K
#PBS -q normal
#PBS -A vasp
#PBS -l select=4:ncpus=64:ompthreads=1
#PBS -l walltime=48:00:00

module purge
module load intel/oneapi_21.2 impi/oneapi_21.2

cd $PBS_O_WORKDIR
./run.sh
