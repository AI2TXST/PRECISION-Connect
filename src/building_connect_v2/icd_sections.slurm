#!/bin/bash
#SBATCH --job-name=icd_sections
#SBATCH --partition=shared
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --mem=240Gb
#SBATCH --time=14-00:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=m_e172@txstate.edu
#SBATCH --output=job_icd_output.txt
#SBATCH --error=job_icd_error.txt

echo ""
echo "Starting at `date`"
echo "Running on hosts: $SLURM_NODELIST"
echo "Running on $SLURM_NNODES nodes."
echo "Running on $SLURM_NPROCS processors."

# path to the correct directory
cd /home/m_e172/OASIS/OASIS_v2/
echo "Current working directory is `pwd`"

# Load Conda (adjust module name if your cluster uses something different)
module load anaconda

# Activate pre-built Spark environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate spark_env

python icd_sections.py

conda deactivate
echo ""
echo "Program finished with exit code $? at: `date`"
echo ""
