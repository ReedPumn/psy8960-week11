#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=12
#SBATCH --mem=5gb
#SBATCH -t 00:10:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=pries153@umn.edu
#SBATCH -p amdsmall
cd ~/week11-cluster
module load R/4.2.2-openblas
Rscript week11-cluster.R
