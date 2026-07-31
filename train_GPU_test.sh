#!/bin/bash
#SBATCH -J jobname
#SBATCH -o Logs/Train/%x.%j.log
#SBATCH -p gpushort          # request gpushort partition
#SBATCH --ntasks=1           # a single python process, not a multi-task/MPI job
#SBATCH --cpus-per-task=1    # pipeline is single-threaded/GPU-bound; extra cores go unused
#SBATCH -t 1:0:0             # 1 hour runtime (required to run on the short partition)
#SBATCH --mem=88G            # flat request, no longer tied to core count
#SBATCH --gres=gpu:1         # request 1 GPU of any type

cd /data/PHURI-Langenberg/people/Nat/Protein-Prediction/

echo "Starting Job: $SLURM_JOB_NAME with Job ID: $SLURM_JOB_ID"

module load miniforge
mamba activate protein_prediction

python -u Code/09-Scripts/train.py --config Configs/models/d_script_and_esm.yaml