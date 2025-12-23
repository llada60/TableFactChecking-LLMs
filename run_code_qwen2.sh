#!/bin/bash 
#SBATCH --job-name=qwen-paraphrased             # Name of your job
#SBATCH --output=logs/qwen-paraphrased.out 
#SBATCH --partition=L40S             # Partition to submit to (A100, V100, etc.)
#SBATCH --gres=gpu:1                  # Request 1 GPU
#SBATCH --cpus-per-task=8             # Request 8 CPU cores
#SBATCH --mem=64G                     
#SBATCH --time=24:00:00               # Time limit for the job (hh:mm:ss)

# Activate the environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate inf

echo "Job name: $SLURM_JOB_NAME"

# Execute the Python scjobjobript with specific arguments
srun python src/baselines/LLM/llm_json.py --data_path ./data/test_examples_with_csv_paraphrased_flattened.json --model_name Qwen/Qwen2.5-7B-Instruct-1M --max_new_tokens 20

echo "Job name: $SLURM_JOB_NAME"
