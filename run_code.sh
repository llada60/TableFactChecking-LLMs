#!/bin/bash
#SBATCH --job-name=ds-origin             # Name of your job
#SBATCH --output=logs/ds-origin.out
#SBATCH --partition=V100             # Partition to submit to (A100, V100, etc.)
#SBATCH --gres=gpu:1                  # Request 1 GPU
#SBATCH --cpus-per-task=8             # Request 8 CPU cores
#SBATCH --mem=64G                     
#SBATCH --time=24:00:00               # Time limit for the job (hh:mm:ss)

# Activate the environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate llavaov

echo "Job name: $SLURM_JOB_NAME"
echo "ds Origin Test"

# Execute the Python scjobjobript with specific arguments
srun python src/llm_parallel.py --data_path ./data/promptDataset/test_examples_with_csv_direct_prompt.json --model_name deepseek-ai/DeepSeek-V2-Lite

echo "Job name: $SLURM_JOB_NAME"
echo "ds Origin Test"