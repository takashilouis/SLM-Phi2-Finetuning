#Model: 
# Phi-2 quantized model (4-bit QLoRA)

#Dataset: Diagsum
#PEFT: LoRA
#LoRA config
# 'q_proj',  # query projection in attention
# 'k_proj',  # key projection in attention
# 'v_proj',  # value projection in the MLP (feedforward network)
# 'dense',   # dense projection in the MLP
#
# Training Arguments
# do_train=undefined,
# gradient_accumulation_steps=
# ##### IMPORT LIBRARIES ########
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
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
import evaluate
from pynvml import *
from tqdm import tqdm

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
def gen(model,p, maxlen=100, sample=True, pad_token_id=None):
    toks = tokenizer(p, return_tensors="pt")# Set pad_token_id to eos_token_id if not provided

    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id

    res = model.generate(
            **toks.to("cuda"), max_new_tokens=maxlen, do_sample=sample,num_return_sequences=1,
            temperature=0.1,num_beams=1,top_p=0.95,
            pad_token_id=pad_token_id  # Explicitly set pad_token_id
        ).to('cpu')

    return tokenizer.batch_decode(res,skip_special_tokens=True)

# Summarization function
def summarize_dialogues(index, dataset, model, tokenizer):
    dialogue = dataset['test'][index]['dialogue']
    summary = dataset['test'][index]['summary']

    formatted_prompt = f"Instruct: Summarize the following conversation.\n{dialogue}\nOutput:\n"

    pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id
    res = gen(model,formatted_prompt,100,pad_token_id=pad_token_id)
    #print(res[0])
    output = res[0].split('Output:\n')[1]
        
    dash_line = '-'.join('' for x in range(100))
    print(dash_line)
    print(f'INPUT PROMPT:\n{formatted_prompt}')
    print(dash_line)

    print(f'BASELINE HUMAN SUMMARY:\n{summary}\n')
    print(dash_line)
    print(f'MODEL SUMMARY:\n{output}')
    print(dash_line)

#Preprocessing dataset
#1.convert the dialog-summary (prompt-response) pairs into explicit instructions for the LLM
def create_prompt_formats(sample):
    """
    Format various fields of the sample ('instruction','output')
    Then concatenate them using two newline characters
    :param sample: Sample dictionnary
    """
    INTRO_BLURB = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
    INSTRUCTION_KEY = "### Instruct: Summarize the below conversation."
    RESPONSE_KEY = "### Output:"
    END_KEY = "### End"

    blurb = f"\n{INTRO_BLURB}"
    instruction = f"{INSTRUCTION_KEY}"
    input_context = f"{sample['dialogue']}" if sample["dialogue"] else None
    response = f"{RESPONSE_KEY}\n{sample['summary']}"
    end = f"{END_KEY}"

    parts = [part for part in [blurb, instruction, input_context, response, end] if part]

    formatted_prompt = "\n\n".join(parts)
    sample["text"] = formatted_prompt

    return sample

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
        remove_columns=['id', 'topic', 'dialogue','summary'],  # Remove unused fields
    )

    # Filter out samples that have input_ids exceeding max_length
    dataset = dataset.filter(lambda sample: len(sample["input_ids"]) < max_length)
    
    # Shuffle dataset
    dataset = dataset.shuffle(seed=seed)

    return dataset

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
        #padding="max_length",# Explicitly add padding for uniform input sizes
    )
from tqdm import tqdm

def process_and_output_summaries(base_model, peft_model, dataset, tokenizer, batch_size=10, max_dialogues=100):
    dialogues = dataset['test']['dialogue'][:max_dialogues]
    human_baseline_summaries = dataset['test']['summary'][:max_dialogues]
    original_model_summaries = []
    peft_model_summaries = []
    # Process in batches
    for start_idx in tqdm(range(0, len(dialogues), batch_size),desc= "Processing batches"):
        end_idx = min(start_idx + batch_size, len(dialogues))
        batch_dialogues = dialogues[start_idx:end_idx]
        tokens = tokenizer(batch_dialogues, return_tensors="pt", padding=True,truncation=True)
        summary_length = tokens["input_ids"].shape[1]
        pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id
        #pad_token_id = tokenizer.eos_token_id or tokenizer.pad_token_id

        print(f"Processing batch: {start_idx} to {end_idx}... ({end_idx} out of {len(dialogues)} rows)", flush=True)
        
        # Generate summaries
        formatted_prompts = [f"Instruct: Summarize the following dialogue.\n{d}\nOutput:\n" for d in batch_dialogues]
        original_batch_summaries = gen(base_model,formatted_prompts,summary_length,pad_token_id=pad_token_id)
        peft_batch_summaries = gen(peft_model,formatted_prompts,summary_length,pad_token_id=pad_token_id)

        # Store summaries
        original_model_summaries.extend(original_batch_summaries)
        peft_model_summaries.extend(peft_batch_summaries)
        torch.cuda.empty_cache()
        print(f"Completed batch: {start_idx} to {end_idx}.", flush=True)
    # Compute ROUGE scores for the first 100 dialogues
    rouge = evaluate.load('rouge')
    # Adjust references to match the length of predictions
    references_for_original = human_baseline_summaries[:len(original_model_summaries)]
    references_for_peft = human_baseline_summaries[:len(peft_model_summaries)]
    original_model_results = rouge.compute(
        predictions=original_model_summaries,
        references=references_for_original,
        use_aggregator=True,
        use_stemmer=True,
    )
    peft_model_results = rouge.compute(
        predictions=peft_model_summaries,
        references=references_for_peft,
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
        print(f"\nDialogues {i+1}:", flush=True)
        print(f"Original Model Summary: {original_model_summaries[i]}", flush=True)
        print(f"PEFT Model Summary: {peft_model_summaries[i]}", flush=True)
        print(f"Reference Summary: {human_baseline_summaries[i]}", flush=True)

############### LOADING DATASETS #############
from datetime import datetime

print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Phi2-PEFT-model with knkarthick/dialogsum evaluation")
print("finetuning-phi2.py")

from dotenv import load_dotenv
from huggingface_hub import login
load_dotenv()
huggingface_token = os.getenv("HF_HUB_TOKEN")
print("Loaded huggingface token!")

#loading dataset from HuggingFace
#Dataset: abisee/cnn_dailymail
from datasets import load_from_disk

try:
    huggingface_dataset_name = "knkarthick/dialogsum"
    dataset = load_dataset(huggingface_dataset_name)
    print("knkarthick/dialogsum dataset successfully downloaded!")
    print(dataset)  # SLURM will capture this output
except Exception as e:
    print("Failed to download or load the dataset:", e)

######### LOADING MODEL ########
# Load the quantized model from local disk
#model_path = "./phi2_quantized_model"
model_name='microsoft/Phi-2'
device_map = {"": 0}

if torch.cuda.is_bf16_supported():
  compute_dtype = torch.bfloat16
  attn_implementation = 'flash_attention_2'
else:
  compute_dtype = torch.float16
  attn_implementation = 'sdpa'

try:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=False,
    )
    original_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        attn_implementation=attn_implementation,
        quantization_config=bnb_config,
        trust_remote_code=True,
        token=True
    )
    print(f"Model loaded successfully !")
    #Track memory usage
    track_memory("After loading model from local directory")
    # Load the tokenizer from the same directory
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,  # Path to the saved quantized model
        trust_remote_code=True,
        padding_side="left",
        add_eos_token=True,
        add_bos_token=True,
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.eos_token  # Ensure proper padding
    print("Tokenizer loaded successfully!")
except Exception as e:
    print(f"Failed to load the model '{model_name}' & tokenizer", e)

############ INFERENCE TESTING ON QUANTIZED MODEL #######
index = 20
seed = 42
set_seed(seed)
summarize_dialogues(index,dataset,original_model,tokenizer)

############ PREPROCESSING ############
#Pre-process dataset
max_length = get_max_length(original_model)
#seed = 42
#set_seed(seed)
train_dataset = preprocess_dataset(tokenizer, max_length,seed, dataset['train'])
eval_dataset = preprocess_dataset(tokenizer, max_length,seed, dataset['validation'])

print("Preprocessed the dataset succesfully!")

# ########### LoRA configuration #########
# #Preparing the model for QLora
# #Setup PEFT for Fine-tuning
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# # 'set_seed(1234)' sets the random seed for reproducibility.
set_seed(1234)
# # LoRA configuration for the model
# # 'lora_r' is the dimension of the LoRA attention.
lora_r = 32

# # 'lora_alpha' is the alpha parameter for LoRA scaling.
lora_alpha = 32

# # 'lora_dropout' is the dropout probability for LoRA layers.
lora_dropout = 0.05

config = LoraConfig(
     r=lora_r, #Rank
     lora_alpha=lora_alpha,
     target_modules=[
         'q_proj',  # Combined query, key, value projection in attention
         'k_proj',    # Output projection in attention
         'v_proj',    # Gate projection in the MLP (feedforward network)
         'dense',       # Down projection in the MLP
     ],
     bias="none",
     lora_dropout=lora_dropout,  # Conventional
     task_type="CAUSAL_LM",
)

# # 1 - Enabling gradient checkpointing to reduce memory usage during fine-tuning
original_model.gradient_checkpointing_enable()

# # 2 - Using the prepare_model_for_kbit_training method from PEFT
original_model = prepare_model_for_kbit_training(original_model)

print("LoRA configuration setup succesfully!")

# ########### Finetuning configuration #########
peft_model = get_peft_model(original_model, config)
# #train PEFT adapter
output_dir = f'./peft-dialogues-summary-training-{str(int(time.time()))}'

peft_training_args = TrainingArguments(
     output_dir=output_dir,
     eval_strategy="no",
     do_train=True,
     per_device_train_batch_size=2,
     gradient_accumulation_steps=2,
     log_level="debug",
     save_strategy="epoch",
     save_total_limit=1,
     learning_rate=2e-4,
     logging_steps=500,
     fp16 = not torch.cuda.is_bf16_supported(),
     bf16 = torch.cuda.is_bf16_supported(),
     num_train_epochs=1,
     warmup_ratio=0.1,
     lr_scheduler_type="linear",
     optim="paged_adamw_8bit",
     gradient_checkpointing=False,  # Enable gradient checkpointing
)
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
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

################# TRAINING ################
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
peft_trainer.save_model("/scratch/dsu.local/khanh.nguyen/working/peft-phi-2-model-1-epoch-diagsum")

torch.cuda.empty_cache()

ft_model = PeftModel.from_pretrained(
    original_model, 
    "/scratch/dsu.local/khanh.nguyen/working/peft-phi-2-model-1-epoch-diagsum",
    torch_dtype=torch.float16,
    is_trainable=False
)

print("FINE-TUNED MODEL: ")
summarize_dialogues(index,dataset,ft_model,tokenizer)

#batch_size = 10  # Reduced batch size
#max_dialogues = 100
#process_and_output_summaries(original_model, ft_model, dataset, tokenizer, batch_size, max_dialogues)