#!/bin/bash
#SBATCH --job-name=Protein_Prediction
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=Logs/Train/%x.%j.log

cd /data/PHURI-Langenberg/people/Nat/Protein-Prediction/

echo "Starting Job: $SLURM_JOB_NAME with Job ID: $SLURM_JOB_ID"

module load miniforge
mamba activate protein_prediction

python Code/09-Scripts/train.py --config Configs/models/baseline.yaml 