#For Downloading phi2-3 models

##### IMPORT LIBRARIES ########
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    set_seed,
    pipeline
)
from trl import SFTTrainer 
import torch
import os
from random import randrange
from peft import LoraConfig, prepare_model_for_kbit_training, TaskType, PeftModel
import psutil
import time
import transformers
from pynvml import *
import evaluate

################## MAIN #################
# Load environment variables for Hugging Face
from dotenv import load_dotenv
from huggingface_hub import login
load_dotenv()
huggingface_token = os.getenv("HF_HUB_TOKEN")

# Login to Hugging Face
if huggingface_token:
    login(token=huggingface_token)
    print("Login successful!")
else:
    print("Error: Hugging Face token not found in environment variables.")
    exit()

#bitsandbytes configuration
#4-bit format
#purpose: to reduce memory consumption considerably
compute_dtype = getattr(torch, "float16")
bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True#False,
    )
#model_name='microsoft/phi-2'
model_name='microsoft/Phi-3-mini-4k-instruct'
device_map = {"": 0}
original_model = AutoModelForCausalLM.from_pretrained(model_name,
                                                      device_map=device_map,
                                                      quantization_config=bnb_config,
                                                      trust_remote_code=True,
                                                      token=True)#use_auth_token is deprecated

tokenizer = AutoTokenizer.from_pretrained(model_name,
                                        trust_remote_code=True,
                                        padding_side="left",
                                        add_eos_token=True,
                                    add_bos_token=True,use_fast=False)
tokenizer.pad_token = tokenizer.eos_token

print("Tokenizer loaded successfully from the local directory.")
################## SAVE MODEL AND TOKENIZER ##################
# Specify the directory to save the quantized model and tokenizer
#save_directory = "./phi2_quantized_model"
save_directory = "./phi3_quantized_model"
# Save the quantized model
original_model.save_pretrained(save_directory, quantization_config=bnb_config)

# Save the tokenizer
tokenizer.save_pretrained(save_directory)

print(f"Quantized model and tokenizer saved successfully to {save_directory}")