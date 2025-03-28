#EVALUATION:
#PHI-2 MODEL
#CNN_DailyMail dataset

##### IMPORT LIBRARIES ########
from datasets import load_dataset, load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
    BitsAndBytesConfig
)
from peft import peft_model
from functools import partial
from pynvml import *
from peft import LoraConfig, prepare_model_for_kbit_training, TaskType, PeftModel
import torch
import evaluate
import numpy as np
import pandas as pd
import os
import time
from tqdm import tqdm
from datetime import datetime
from rouge_score import rouge_scorer

######## CUSTOM FUNCTIONS ##########
# Define memory tracking function
def track_memory(label):
    memory_allocated = print_gpu_utilization()
    print(f"Memory Usage {label}: {memory_allocated:.2f} MB", flush=True)

# Function to print GPU memory utilization
def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    return info.used #// 1024 ** 2

# Function to generate a summary
def gen(model, prompt, maxlen, sample=True, pad_token_id=None):
    toks = tokenizer(
        prompt,
        return_tensors="pt",
        #max_length=maxlen,  # Ensure uniform length
        #padding=True,       # Pad shorter sequences
        #truncation=True     # Truncate longer sequences
    )
    
    # Set pad_token_id if not provided
    #if pad_token_id is None:
    #    pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id
    pad_token_id = pad_token_id or tokenizer.eos_token_id
    #Abstractive Summarization:
    #Use top_p = 0.85–0.9 for tasks requiring more creative or abstracted summaries, like transforming text into new formats.
    res = model.generate(
        **toks.to("cuda"), 
        max_new_tokens=maxlen, 
        do_sample=sample, 
        num_return_sequences=1,
        temperature=0.1, #0.1
        num_beams=1, #1
        top_p=0.1,#0.95->0.85
        pad_token_id=pad_token_id,
        early_stopping=True,#when num_beams > 1 ;if there are too many redundant tokens, it will stop after certain amount of tokens.
        repetition_penalty=1.2
    ).to("cpu")
    
    return tokenizer.batch_decode(res, skip_special_tokens=True)

from rouge_score import rouge_scorer
# Summarization function
def summarize_article(index, dataset, model, tokenizer):
    article = dataset['test'][index]['article']
    summary = dataset['test'][index]['highlights']

    formatted_prompt = f"Instruct: Summarize the below article.\nArticle:\n{article}.\nOutput:\n"

    pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id
    res = gen(model,formatted_prompt,100,pad_token_id=pad_token_id)
    #print(res[0])
    output = res[0].split('Output:\n')[1]
        
    # Further split to remove any content after "##OUTPUT"
    if "##OUTPUT" in output:
        generated_summary = output.split("##OUTPUT", 1)[0].strip()
    elif "#" in output:
        generated_summary = output.split("#", 1)[0].strip()
    else:
        generated_summary = output.strip() 
    
    dash_line = '-'.join('' for x in range(100))
    print(dash_line)
    print(f'INPUT PROMPT:\n{formatted_prompt}')
    print(dash_line)
    print(f'BASELINE HUMAN SUMMARY:\n{summary}\n')
    print(dash_line)
    print(f'MODEL SUMMARY:\n{generated_summary}')
    print(dash_line)

    # Initialize the scorer with desired ROUGE metrics
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL', 'rougeLsum'], use_stemmer=True)

    # Compute scores
    scores = scorer.score(summary, generated_summary)

    # Print the F1 scores as percentages
    for metric, score in scores.items():
        print(f"{metric}: {score.fmeasure * 100:.2f}%")
 


def evaluate_model(base_model, peft_model, dataset, tokenizer, batch_size=10, max_articles=100):
    articles = dataset['test']['article'][:max_articles]
    human_baseline_summaries = dataset['test']['highlights'][:max_articles]
    original_model_summaries = []
    peft_model_summaries = []
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL', 'rougeLsum'], use_stemmer=True)

    # Process in batches
    for start_idx in tqdm(range(0, len(articles), batch_size),desc= "Processing batches"):
        end_idx = min(start_idx + batch_size, len(articles))
        batch_articles = articles[start_idx:end_idx]
        summary_length = 256
        pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id
        #pad_token_id = tokenizer.eos_token_id or tokenizer.pad_token_id

        print(f"Processing batch: {start_idx} to {end_idx}... ({end_idx} out of {len(articles)} rows)", flush=True)
        
        # Generate summaries
        formatted_prompts = [f"Instruct: Summarize the following article.\n{d}\nOutput:\n" for d in batch_articles]
        original_batch_summaries = gen(base_model,formatted_prompts,summary_length,pad_token_id=pad_token_id)
        peft_batch_summaries = gen(peft_model,formatted_prompts,summary_length,pad_token_id=pad_token_id)

        # Store summaries
        original_model_summaries.extend(original_batch_summaries)
        peft_model_summaries.extend(peft_batch_summaries)
        
        torch.cuda.empty_cache()
        print(f"Completed batch: {start_idx} to {end_idx}.", flush=True)

    # Compute and print individual ROUGE scores for each article
    print("\n--- Individual ROUGE Scores ---\n")
    for i, (gen_summary, ref_summary) in enumerate(zip(original_model_summaries, human_baseline_summaries)):
        print(f"Article {i + 1}:")
        scores = scorer.score(ref_summary, gen_summary)
        for metric, score in scores.items():
            print(f"  {metric}: {score.fmeasure * 100:.2f}%")
        print()  # Blank line for better readability

    # Compute ROUGE scores for the first 100 articles
    rouge = evaluate.load('rouge')
    # Adjust references to match the length of predictions
    original_model_results = rouge.compute(
        predictions=original_model_summaries,
        references=human_baseline_summaries[0:end_idx],
        use_aggregator=True,
        use_stemmer=True,
    )
    peft_model_results = rouge.compute(
        predictions=peft_model_summaries,
        references=human_baseline_summaries[0:end_idx],
        use_aggregator=True,
        use_stemmer=True,
    )

    # Print ROUGE scores
    print(f"ORIGINAL MODEL RESULTS: {original_model_results}", flush=True)
    print(f"PEFT MODEL RESULTS: {peft_model_results}", flush=True)

    # Print absolute percentage improvement
    print("Absolute percentage improvement of PEFT MODEL over ORIGINAL MODEL", flush=True)
    improvement = np.array(list(peft_model_results.values())) - np.array(list(original_model_results.values()))
    for key, value in zip(peft_model_results.keys(), improvement):
        print(f"{key}: {value * 100:.2f}%", flush=True)

######################### MAIN PROGRAM ###########################
print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Phi2-PEFT-model Evaluation")
print("evaluation.py")

# Login to Hugging Face
from dotenv import load_dotenv
from huggingface_hub import login
load_dotenv()
huggingface_token = os.getenv("HF_HUB_TOKEN")

if huggingface_token:
    login(token=huggingface_token)
    print("Login successful!")
else:
    print("Error: Hugging Face token not found in environment variables.")
    exit()
    
dataset_path = "/scratch/dsu.local/khanh.nguyen/CNNDLM_datasets"
try:
     dataset = load_from_disk(dataset_path)
     print("Dataset successfully loaded.", flush=True)

except Exception as e:
     print(f"Failed to load dataset: {e}", flush=True)
     
#huggingface_dataset_name = "abisee/cnn_dailymail"
#dataset = load_dataset(huggingface_dataset_name, '3.0.0')
####### LOADING BASE MODEL ############
track_memory("Before Loading Base Model")

bnb_config = BitsAndBytesConfig(
     load_in_4bit=True,
     bnb_4bit_quant_type='nf4',
     bnb_4bit_compute_dtype=torch.float16,
     bnb_4bit_use_double_quant=False,
)

base_model_id = "microsoft/phi-2"
base_model = AutoModelForCausalLM.from_pretrained(base_model_id,
                                                      device_map='auto',
                                                      quantization_config=bnb_config,
                                                      trust_remote_code=True,
                                                      use_auth_token=True)

tokenizer = AutoTokenizer.from_pretrained(
        base_model_id,  # Path to the saved quantized model
        trust_remote_code=True,
        padding_side="left",
        add_eos_token=True,
        add_bos_token=True,
        use_fast=False,
    )
tokenizer.pad_token = tokenizer.eos_token

#tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
print("Base model loaded successfully!", flush=True)
track_memory("After Loading Base Model")

####### LOADING PEFT MODEL ############
track_memory("Before Loading PEFT Model")
peft_model_path = "/scratch/dsu.local/khanh.nguyen/working/peft-phi-2-model-2-epochs"
ft_model = PeftModel.from_pretrained(
    base_model, 
    peft_model_path,
    torch_dtype=torch.float16
)
print(f"peft model: {peft_model_path}")
print(ft_model)
print("PEFT model loaded successfully!", flush=True)
track_memory("After Loading PEFT Model")

from safetensors.torch import load_file

adapter_weights = load_file(f"{peft_model_path}/adapter_model.safetensors")
for key, value in adapter_weights.items():
    print(f"{key}: {value.flatten()[:5]}")  # Print first 5 values


########### Sample summarization ##############
seed = 42
set_seed(seed)

for index in range(0,100,10):
    print(f"Index={index}")
    print("BASE MODEL:")
    summarize_article(index, dataset, base_model, tokenizer)
    print("\nPEFT MODEL:")
    summarize_article(index, dataset, ft_model, tokenizer)
    print("---------------------------------------------------")

# index=200
# print("\nIndex=200")
# print("BASE MODEL:")
# summarize_article(index, dataset, base_model, tokenizer)

# print("\nPEFT MODEL:")
# summarize_article(index, dataset, ft_model, tokenizer)

# Process dataset in batches
batch_size = 10  # Reduced batch size
evaluate_model(base_model, ft_model, dataset, tokenizer, batch_size=batch_size, max_articles=20)

