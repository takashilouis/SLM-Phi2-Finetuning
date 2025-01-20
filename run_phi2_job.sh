#!/bin/bash
#SBATCH --job-name=phi2_model_load       # Job name
#SBATCH --output=phi2_model_load_%j.out  # Output file
#SBATCH --error=phi2_model_load_%j.err   # Error file
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
#pip install --upgrade pip

# Install required Python libraries
#pip install -q -U bitsandbytes peft accelerate datasets einops evaluate trl rouge_score
#pip install -U transformers[sentencepiece]
#pip install -U sentencepiece
# Run the Python script
# python phi2finetuning.py
#python phi-2-finetuning.py
#python finetuning-phi2.py
python evaluation.py
#python download-model-tokenizer.py
