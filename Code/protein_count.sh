#!/bin/bash
#$ -cwd
#$ -N Protein_Count
#$ -pe smp 1
#$ -l h_vmem=20G
#$ -l h_rt=1:00:00
#$ -o Protein-Prediction/Logs/$JOB_NAME.$JOB_ID.log
#$ -j y

echo "Starting Job: $JOB_NAME with Job ID: $JOB_ID"

module load miniforge
mamba activate protein_prediction

python Protein-Prediction/Code/protein_count.py