#!/bin/bash
#SBATCH --b-name=”Rep”
#SBATCH --error=job.err
#SBATCH --output=job.out
#SBATCH --partition=allgroups
#SBATCH --ntasks=1
#SBATCH --mem=1G
#SBATCH --time=01:00:00
echo “HELLO WORLD”