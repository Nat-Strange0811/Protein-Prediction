#!/bin/bash
#SBATCH --job-name=Utils
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --time=1:00:00
#SBATCH --output=Logs/Utils/%x.%j.log

cd /data/PHURI-Langenberg/people/Nat/Protein-Prediction/

echo "Starting Job: $SLURM_JOB_NAME with Job ID: $SLURM_JOB_ID"

module load miniforge
mamba activate protein_prediction

python Code/99-Utils/panel_gen.py