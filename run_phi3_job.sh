#!/bin/bash
#SBATCH --job-name=phi3_model_load       # Job name
#SBATCH --output=phi3_model_load_%j.out  # Output file
#SBATCH --error=phi3_model_load_%j.err   # Error file
#SBATCH --partition=gpu                  # GPU partition/queue
#SBATCH --gres=gpu:1                     # Request 1 GPU
#SBATCH --cpus-per-task=16               # Number of CPUs
#SBATCH --mem=64G                        # Memory per node
#SBATCH --time=36:00:00                  # Time limit

# Load modules (adjust based on your environment)
module load cuda/11.8.0
module load python/3.8.15

# Activate your Python environment (if applicable)
source ~/my_env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install required Python libraries
#pip install nvidia-ml-py3
pip install -q -U bitsandbytes 
pip install -q -U transformers 
pip install -q -U peft 
pip install -q -U accelerate datasets trl
pip install -q -U einops 
pip install -U flash_attn 
pip install -q -U triton wandb 
pip install -q -U rouge_score

# Evaluate the summarization
#python evaluation.py
#python import.py
# Run the Python script
python phi3-finetuning.py

# Download the model
# python test.py
