#!/bin/bash
#SBATCH --job-name=phi2_model_load       # Job name
#SBATCH --output=phi2_model_load_%j.out  # Output file (%j = Job ID)
#SBATCH --error=phi2_model_load_%j.err   # Error file (%j = Job ID)
#SBATCH --partition=gpu                  # GPU partition/queue
#SBATCH --gres=gpu:1                     # Request 1 GPU
#SBATCH --cpus-per-task=16               # Number of CPUs
#SBATCH --mem=48G                        # Memory allocation
#SBATCH --time=200:00:00                 # Time limit (HH:MM:SS)

# Load necessary modules (adjust based on your environment)
module load cuda/11.8.0
module load python/3.8.15
module load git
# Activate your Python virtual environment
source /scratch/dsu.local/khanh.nguyen/my_env/bin/activate

# Upgrade pip (optional, ensures compatibility with newer packages)
# echo "Upgrading pip..."
# pip install --upgrade pip
# pip install wandb

#pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
#pip install flash-attn==2.7.3
# Install dependencies from requirements.txt if it exists
# pip install -r requirements.txt
# Clone the flash-attention repository and build it manually
# if [ ! -d "flash-attention" ]; then
#     echo "Cloning the flash-attention repository..."
#     git clone https://github.com/HazyResearch/flash-attention.git
# else
#     echo "flash-attention repository already exists."
# fi
#pip install wheel

#cd flash-attention
# Build and install flash-attention
#echo "Building and installing flash-attention..."
#python setup.py install

# Navigate back to your working directory
#cd ..


# Run the Python script (customize based on the task)
# Uncomment the script you want to execute
#python evaluation.py
python phi-2tuning-diagsum.py
#python phi2finetuning.py
#python finetuning-phi2.py
# python download-model-tokenizer.py

echo "Deactivating virtual environment..."
deactivate

echo "Job completed successfully!"
