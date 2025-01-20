#!/bin/bash
#SBATCH --job-name=phi2_model_load       # Job name
#SBATCH --output=phi_model_load_%j.out  # Output file
#SBATCH --error=phi_model_load_%j.err   # Error file
#SBATCH --partition=gpu                  # GPU partition/queue
#SBATCH --gres=gpu:1                     # Request 1 GPU
#SBATCH --cpus-per-task=16                # Number of CPUs
#SBATCH --mem=48G                        # Memory per node
#SBATCH --time=200:00:00                  # Time limit

# Load modules (adjust based on your environment)
module load cuda/11.8.0
module load python/3.8.15

# Activate your Python environment (if applicable)
source ~/my_env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install required Python libraries
#pip install -q -U bitsandbytes transformers peft accelerate datasets einops evaluate trl rouge_score

# Run the Python script
# python phi2finetuning.py
#python phi-2-finetuning.py
#export HF_DATASETS_CACHE=/scratch/dsu.local/khanh.nguyen/hf_datasets_cache

pip install -q -U bitsandbytes
pip install -U flash_attn
#python download_dataset.py
python download-model-tokenizer.py