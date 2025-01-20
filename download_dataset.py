from datasets import load_dataset
import os
from huggingface_hub import login
from dotenv import load_dotenv

# Load environment variables for Hugging Face
load_dotenv()
login(token=os.getenv("HF_HUB_TOKEN"))

print("Login successfully!")

# Load the dataset
huggingface_dataset_name = "abisee/cnn_dailymail"
dataset = load_dataset(huggingface_dataset_name, '3.0.0')

# Save the dataset to the local disk
dataset.save_to_disk("/scratch/dsu.local/khanh.nguyen/CNNDLM_datasets")