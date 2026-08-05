#!/bin/bash
#SBATCH -A pilot_sae_gpu
#SBATCH -p sae
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1  
#SBATCH --job-name=Protein_Prediction
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=Logs/Train/%x.%j.log

cd /data/PHURI-Langenberg/people/Nat/Protein-Prediction/

echo "Starting Job: $SLURM_JOB_NAME with Job ID: $SLURM_JOB_ID"

module load miniforge
mamba activate protein_prediction

python -u Code/09-Scripts/train.py --config Configs/models/baseline_small_MLP.yaml