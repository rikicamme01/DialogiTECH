#!/bin/sh

#SBATCH --job-name=bert-rep
#SBATCH --error=job.err
#SBATCH --output=job.out
#SBATCH --partition=allgroups
#SBATCH --ntasks=1
#SBATCH --mem=1G
#SBATCH --time=01:00:00
#SBATCH --constraint=dellcuda1
#SBATCH --gres=mps:1

SHAREDIR="/conf/shared-software/Singularity/CUDA/"

singularity exec -B \
	    ${SHAREDIR}/driver/`hostname`:/nvdriver \
            ./container.sif
            ./hello.py
