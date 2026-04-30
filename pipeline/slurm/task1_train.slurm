#!/bin/bash
#SBATCH --job-name=task1_optuna_310
#SBATCH --partition=himem
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --nodes=1
#SBATCH --mem=400Gb
#SBATCH --time=28-00:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=fpd16@txstate.edu
#SBATCH --output=task1_optuna_310_output.txt
#SBATCH --error=task1_optuna_310_error.txt

echo ""
echo "Starting at `date`"
echo "Running on hosts: $SLURM_NODELIST"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"

cd /home/fpd16/independentStudy/Connect-2.0
echo "Current working directory is `pwd`"

source /home/fpd16/anaconda3/etc/profile.d/conda.sh
conda activate venv

python pipeline/task1/train_optuna.py

conda deactivate

echo ""
echo "Program finished with exit code $? at: `date`"
echo ""
