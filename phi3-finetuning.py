#Model:
# Phi-3-4k quantized model (4-bit QLoRA)
#
#Dataset: CNN/DailyMail
#PEFT: LoRA
#LoRA config
# 'qkv_proj',  #  
# 'o_proj',  #  
# 'gate_up_proj',  # 
# 'down',   #  
#
# Training Arguments
# do_train=undefined,
# gradient_accumulation_steps=2

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

def process_and_output_summaries(base_model, ft_model, dataset, tokenizer, batch_size=10, max_articles=100):
    dialogues = dataset['test']['article'][:max_articles]
    human_baseline_summaries = dataset['test']['highlights'][:max_articles]

    original_model_summaries = []
    peft_model_summaries = []

    # Process in batches
    for start_idx in range(0, len(dialogues), batch_size):
        end_idx = min(start_idx + batch_size, len(dialogues))
        batch_dialogues = dialogues[start_idx:end_idx]
        print(f"Processing batch: {start_idx} to {end_idx}... ({end_idx} out of {len(dialogues)} rows)", flush=True)

        # Generate summaries
        batch_prompts = [f"Instruct: Summarize the following article.\n{d}\nOutput:\n" for d in batch_dialogues]
        original_batch_summaries = gen(base_model, tokenizer, batch_prompts, maxlen=150)
        peft_batch_summaries = gen(ft_model, tokenizer, batch_prompts, maxlen=150)

        # Store summaries
        original_model_summaries.extend(original_batch_summaries)
        peft_model_summaries.extend(peft_batch_summaries)

        torch.cuda.empty_cache()
        print(f"Completed batch: {start_idx} to {end_idx}.", flush=True)

    # Compute ROUGE scores for the first 100 articles
    rouge = evaluate.load('rouge')
    original_model_results = rouge.compute(
        predictions=original_model_summaries,
        references=human_baseline_summaries,
        use_aggregator=True,
        use_stemmer=True,
    )
    peft_model_results = rouge.compute(
        predictions=peft_model_summaries,
        references=human_baseline_summaries,
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

    # Output the first 10 summaries
    print("\n--- First 10 Summaries ---", flush=True)
    for i in range(10):
        print(f"\nArticle {i+1}:", flush=True)
        print(f"Original Model Summary: {original_model_summaries[i]}", flush=True)
        print(f"PEFT Model Summary: {peft_model_summaries[i]}", flush=True)
        print(f"Reference Summary: {human_baseline_summaries[i]}", flush=True)

############### LOADING DATASETS #############
torch.cuda.empty_cache()  # Clears unused memory
torch.cuda.reset_peak_memory_stats()

from dotenv import load_dotenv
from huggingface_hub import login
load_dotenv()
huggingface_token = os.getenv("HF_HUB_TOKEN")

#loading dataset from HuggingFace
#Dataset: abisee/cnn_dailymail
from datasets import load_from_disk

try:
    dataset_path = "/scratch/dsu.local/khanh.nguyen/CNNDLM_datasets"
    dataset = load_from_disk(dataset_path)
    print("Dataset successfully downloaded and loaded.")
    print(dataset)  # SLURM will capture this output
except Exception as e:
    print("Failed to download or load the dataset:", e)

######### LOADING MODEL ########
# Load the quantized model from local disk
#model_path = "./phi3_quantized_model"
model_name='microsoft/Phi-3-mini-4k-instruct'
device_map = {"": 0}

try:

    original_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",#device_map,
        trust_remote_code=True,
    )
    print(f"Model loaded successfully from '{model_path}'.")

    #Track memory usage
    track_memory("After loading model from local directory")

    #Set up the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                            trust_remote_code=True,
                                            padding_side="left",add_eos_token=True,
                                            add_bos_token=True,use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token

    print("Tokenizer loaded successfully from the local directory.")

except Exception as e:
    print(f"Failed to load the model & tokenizer '{model_path}':", e)

############ PREPROCESSING ############
#Pre-process dataset
max_length = get_max_length(original_model)
seed = 42
train_dataset = preprocess_dataset(tokenizer, max_length,seed, dataset['train'])
eval_dataset = preprocess_dataset(tokenizer, max_length,seed, dataset['validation'])

print("Preprocessed the dataset succesfully!")

########### LoRA configuration #########
#Preparing the model for QLora
#Setup PEFT for Fine-tuning
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# 'set_seed(1234)' sets the random seed for reproducibility.
set_seed(1234)
# LoRA configuration for the model
# 'lora_r' is the dimension of the LoRA attention.
lora_r = 32

# 'lora_alpha' is the alpha parameter for LoRA scaling.
lora_alpha = 32

# 'lora_dropout' is the dropout probability for LoRA layers.
lora_dropout = 0.05

config = LoraConfig(
    r=lora_r, #Rank
    lora_alpha=lora_alpha,
    target_modules=[
        'qkv_proj',  # Combined query, key, value projection in attention
        'o_proj',    # Output projection in attention
        'gate_up_proj',    # Gate projection in the MLP (feedforward network)
        'down_proj',       # Down projection in the MLP
    ],
    lora_dropout=lora_dropout,  # Conventional
    task_type="CAUSAL_LM",
)

# 1 - Enabling gradient checkpointing to reduce memory usage during fine-tuning
original_model.gradient_checkpointing_enable()

# 2 - Using the prepare_model_for_kbit_training method from PEFT
original_model = prepare_model_for_kbit_training(original_model)

print("LoRA configuration setup succesfully!")

########### Finetuning configuration #########
peft_model = get_peft_model(original_model, config)
#train PEFT adapter
output_dir = f'./peft-dialogue-summary-training-{str(int(time.time()))}'
import transformers

peft_training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="steps",
        do_eval=True,
        do_train=True,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        per_device_eval_batch_size=16,
        log_level="debug",
        save_strategy="steps",
        save_total_limit=3,
        learning_rate=5e-5,
        eval_steps=500,#1000 is too much for evaluation, need to be adjusted to 5000
        save_steps=100,
        max_steps=1000,
        #num_train_epochs=1,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,  # Enable gradient checkpointing
)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

peft_model.config.use_cache = False

peft_trainer = transformers.Trainer(
    model=peft_model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=peft_training_args,
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
print("Original Model config:")
print(original_model.config)

########### TRAINING #############
os.environ["WANDB_DISABLED"] = "true"

# Start training with memory tracking
print("Starting PEFT Training with Memory Tracking:")
start_train_time = time.time()

# Track memory before training
track_memory("(Before Finetuning)")

peft_trainer.train()

# Track memory after training
track_memory("After Training")
end_train_time = time.time()

# Total training time
training_time = end_train_time - start_train_time
print(f"Total Training Time: {training_time:.2f} seconds")

peft_trainer.save_model("/scratch/dsu.local/khanh.nguyen/working/peft-phi-3-4k-model")

ft_model = PeftModel.from_pretrained(
    original_model,
    "/scratch/dsu.local/khanh.nguyen/working/peft-phi-3-4k-model",
    torch_dtype=torch.float16,
    is_trainable=False
)
batch_size = 10  # Reduced batch size
process_and_output_summaries(original_model, ft_model, dataset, tokenizer, batch_size=batch_size, max_articles=100)