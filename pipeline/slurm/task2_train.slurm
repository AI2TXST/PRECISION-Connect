#!/bin/bash
#SBATCH --job-name=task2_330_optuna
#SBATCH --partition=himem
#SBATCH --mem=200G
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=task2_330_optuna_output.txt
#SBATCH --error=task2_330_optuna_error.txt
#SBATCH --mail-type=ALL
#SBATCH --mail-user=fpd16@txstate.edu

source ~/anaconda3/etc/profile.d/conda.sh
conda activate venv

python /home/fpd16/independentStudy/Connect-2.0/pipeline/task2/train_optuna.py
