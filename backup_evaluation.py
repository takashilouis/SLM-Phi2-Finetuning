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
import random
from pynvml import *
import evaluate
import numpy as np

######## CUSTOM FUNCTIONS ##########
# Define your memory tracking function
def track_memory(label):
    memory_allocated = print_gpu_utilization()
    print(f"Memory Usage {label}: {memory_allocated:.2f} MB")

# Function to print GPU memory utilization and return value
def print_gpu_utilization():
    from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    memory_used_mb = info.used // 1024 ** 2
    return memory_used_mb

# Function to generate a summary
def gen(model, p, maxlen=150, sample=True, pad_token_id=None):
    toks = tokenizer(p, return_tensors="pt")
    
    # Set pad_token_id if not provided
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id
    #Abstractive Summarization:
    #Use top_p = 0.85–0.9 for tasks requiring more creative or abstracted summaries, like transforming text into new formats.

    res = model.generate(
        **toks.to("cuda"), 
        max_new_tokens=maxlen, 
        do_sample=sample, 
        num_return_sequences=1,
        temperature=0.1, 
        num_beams=1, 
        top_p=0.85,
        pad_token_id=pad_token_id
    ).to("cpu")
    
    return tokenizer.batch_decode(res, skip_special_tokens=True)

# Summarization function
def summarize_article(index, dataset, model, tokenizer):
    # Extract the article and reference summary
    article = dataset['test'][index]['article']
    reference_summary = dataset['test'][index]['highlights']
    
    # Format the input prompt
    formatted_prompt = f"Instruct: Summarize the following article.\n{article}\nOutput:\n"
    
    # Generate the summary
    output = gen(model, formatted_prompt, maxlen=150, pad_token_id=tokenizer.pad_token_id)[0]
    output_summary = output.split("Output:\n")[-1].strip()  # Extract the summary part
    
    # Display the results
    dash_line = '-' * 100
    print(dash_line)
    print(f"INPUT ARTICLE:\n{article}\n")
    print(dash_line)
    print(f"REFERENCE SUMMARY:\n{reference_summary}\n")
    print(dash_line)
    print(f"MODEL-GENERATED SUMMARY:\n{output_summary}\n")

#Preprocessing dataset
#1.convert the dialog-summary (prompt-response) pairs into explicit instructions for the LLM
def create_prompt_formats(sample):
    """
    Formats the sample's fields into a structured prompt for summarization.
    Specifically designed for the CNN/DailyMail dataset.
    
    :param sample: A dictionary containing 'article' and 'highlights'.
    :return: The modified sample with a new 'text' field containing the formatted prompt.
    """
    INTRO_BLURB = (
        "Below is an instruction that describes a task. Write a response that appropriately completes the request."
    )
    INSTRUCTION_KEY = "### Instruct: Summarize the below article."
    RESPONSE_KEY = "### Output:"
    END_KEY = "### End"

    # Retrieve fields with safe defaults
    article_content = sample.get("article", "")
    summary_content = sample.get("highlights", "")

    # Construct individual parts of the prompt
    blurb = f"\n{INTRO_BLURB}"
    instruction = f"{INSTRUCTION_KEY}"
    article = f"{article_content}" if article_content else None
    response = f"{RESPONSE_KEY}\n{summary_content}" if summary_content else None
    end = f"{END_KEY}"

    # Filter out None or empty parts to avoid unnecessary gaps
    parts = [part for part in [blurb, instruction, article, response, end] if part]

    # Join all parts with double newlines
    formatted_prompt = "\n\n".join(parts)

    # Add the formatted prompt to the sample
    sample["text"] = formatted_prompt

    return sample 

#2. use model tokenizer to process these prompts into tokenized ones.
from functools import partial

# SOURCE https://github.com/databrickslabs/dolly/blob/master/training/trainer.py
def get_max_length(model):
    conf = model.config
    max_length = None
    for length_setting in ["n_positions", "max_position_embeddings", "seq_length"]:
        max_length = getattr(model.config, length_setting, None)
        if max_length:
            print(f"Found max lenth: {max_length}")
            break
    if not max_length:
        max_length = 1024
        print(f"Using default max length: {max_length}")
    return max_length


def preprocess_batch(batch, tokenizer, max_length):
    """
    Tokenizes a batch of text data.
    :param batch: A batch of examples with a 'text' field containing input strings.
    :param tokenizer: The tokenizer instance to process text.
    :param max_length: Maximum sequence length for tokenized inputs.
    :return: Tokenized inputs with truncation and optional padding applied.
    """
    return tokenizer(
        batch["text"],
        max_length=max_length,
        truncation=True,
    )

# SOURCE https://github.com/databrickslabs/dolly/blob/master/training/trainer.py
def preprocess_dataset(tokenizer: AutoTokenizer, max_length: int,seed, dataset):
    """Format & tokenize it so it is ready for training
    :param tokenizer (AutoTokenizer): Model Tokenizer
    :param max_length (int): Maximum number of tokens to emit from tokenizer
    """
    print("Preprocessing dataset...")
    print(f"Dataset columns: {dataset.column_names}")
    
    # Add prompt to each sample
    print("Preprocessing dataset...")
    dataset = dataset.map(create_prompt_formats)#, batched=True)
    
    # Apply preprocessing to each batch of the dataset & and remove 'instruction', 'context', 'response', 'category' fields
    _preprocessing_function = partial(preprocess_batch, max_length=max_length, tokenizer=tokenizer)
    dataset = dataset.map(
        _preprocessing_function,
        batched=True,
        remove_columns=["article", "highlights", "id"],  # Remove unused fields
    )

    # Filter out samples that have input_ids exceeding max_length
    dataset = dataset.filter(lambda sample: len(sample["input_ids"]) < max_length)
    
    # Shuffle dataset
    dataset = dataset.shuffle(seed=seed)

    return dataset

############### LOADING DATASETS #############
#loading dataset from HuggingFace
#Dataset: abisee/cnn_dailymail
from datasets import load_from_disk

try:
    dataset_path = "/home/dsu.local/khanh.nguyen/datasets"
    dataset = load_from_disk(dataset_path)
    print("Dataset successfully downloaded and loaded.")
    print(dataset)  # SLURM will capture this output
except Exception as e:
    print("Failed to download or load the dataset:", e)

####### LOADING BASE MODEL  ############
#Track initial memory usage
track_memory("Initial Memory Before Loading Original Model")

# For comparision with and without 4-bit quantization
model_path = "./phi3_quantized_model"
device_map = {"": 0}
base_model = AutoModelForCausalLM.from_pretrained(model_path, 
                                                      device_map=device_map,
                                                      trust_remote_code=True,
                                                      token=True)#use_auth_token is deprecated

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
tokenizer.padding_side = 'right'

print("Loaded model and tokenizer successfully!")

# Track initial memory usage
track_memory("Initial Memory After Loading Original Model without 4bit-quantization")

####### LOADING PEFT MODEL  ############
track_memory("Before Loading Trained PEFT Model")
peft_model = PeftModel.from_pretrained(
    base_model,
    "/home/dsu.local/khanh.nguyen/working/peft-phi-3-summary-training-final",
    torch_dtype=torch.float16,
    is_trainable=False
)
track_memory("After Loading Trained PEFT Model")

########### Sample summarization ##############
# Set random seed for reproducibility
seed = 42
set_seed(seed)
index = 0
summarize_article(index, dataset, peft_model, tokenizer)

########## DATASET ARTICLES SUMMARIZATION ######
import pandas as pd

dialogues = dataset['test']['article']
human_baseline_summaries = dataset['test']['highlights']

original_model_summaries = []
instruct_model_summaries = []
peft_model_summaries = []

for idx, dialogue in enumerate(dialogues):
    human_baseline_text_output = human_baseline_summaries[idx]
    prompt = f"Instruct: Summarize the following article.\n{dialogue}\nOutput:\n"

    original_model_res = gen(base_model,prompt,150,)
    original_model_text_output = original_model_res[0].split('Output:\n')[1]

    peft_model_res = gen(peft_model,prompt,150,)
    peft_model_output = peft_model_res[0].split('Output:\n')[1]
    if idx < 10:
        print(f"{idx} \n")
        print(original_model_text_output)
        print(peft_model_output)
    
    if idx % 1000 == 0:
        print(f"{idx} \n")
        print(original_model_text_output)
        print(peft_model_output)
        
    peft_model_text_output, success, result = peft_model_output.partition('###')

    original_model_summaries.append(original_model_text_output)
    peft_model_summaries.append(peft_model_text_output)

zipped_summaries = list(zip(human_baseline_summaries, original_model_summaries, peft_model_summaries))

df = pd.DataFrame(zipped_summaries, columns = ['human_baseline_summaries', 'original_model_summaries', 'peft_model_summaries'])

############# EVALUATION #############
rouge = evaluate.load('rouge')

original_model_results = rouge.compute(
    predictions=original_model_summaries,
    references=human_baseline_summaries[0:len(original_model_summaries)],
    use_aggregator=True,
    use_stemmer=True,
)

peft_model_results = rouge.compute(
    predictions=peft_model_summaries,
    references=human_baseline_summaries[0:len(peft_model_summaries)],
    use_aggregator=True,
    use_stemmer=True,
)

print('ORIGINAL MODEL:')
print(original_model_results)
print('PEFT MODEL:')
print(peft_model_results)

print("Absolute percentage improvement of PEFT MODEL over ORIGINAL MODEL")

improvement = (np.array(list(peft_model_results.values())) - np.array(list(original_model_results.values())))
for key, value in zip(peft_model_results.keys(), improvement):
    print(f'{key}: {value*100:.2f}%')