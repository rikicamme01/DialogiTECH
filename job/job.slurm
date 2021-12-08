#!/bin/sh

#SBATCH --job-name=GPU-Shared
#SBATCH --error=myjob.err
#SBATCH --output=myjob.out
#SBATCH --partition=allgroups
#SBATCH --ntasks=1
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --constraint=dellcuda1
#SBATCH --gres=mps:1

SHAREDIR="/conf/shared-software/Singularity/CUDA/"

singularity exec -B \
	    ${SHAREDIR}/driver/`hostname`:/nvdriver \
	    ./container.sif \
	    ./hello.py