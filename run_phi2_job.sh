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

# Activate your Python virtual environment
source /scratch/dsu.local/khanh.nguyen/my_env/bin/activate

# Upgrade pip (optional, ensures compatibility with newer packages)
# echo "Upgrading pip..."
# pip install --upgrade pip

# Install dependencies from requirements.txt if it exists
#pip install -r requirements.txt
pip install -U evaluate
pip show pynvml tqdm
# Run the Python script (customize based on the task)
# Uncomment the script you want to execute
# python phi2finetuning.py
# python phi-2-finetuning.py
# python finetuning-phi2.py
python evaluation.py
# python download-model-tokenizer.py

# Deactivate the virtual environment after the job completes
echo "Deactivating virtual environment..."
deactivate

echo "Job completed successfully!"
