# Import required libraries and modules
import os
import torch
from datetime import datetime
import pandas as pd
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
)
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
import bitsandbytes


# Configurable variables
PROJECT_NAME = "tnmLlama3_T14"
DATASET_FILE = "data/Target_Data_T14_train.csv"
BASE_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
TRAIN_TEST_SPLIT_RATIO = 0.1  # 10% for testing
# N SYSTEM="In the following text, determine the stage of N in the TNM staging. N can take on 4 integer values, 0 if there are no cancer cells in any nearby nodes or only small clusters of cancer cells less than 0.2 mm across, 1 if there are cancer cells in 1 to 3 lymph nodes, 2 if there are 4 to 9 lymph nodes in the armpit and at least one is larger than 2 mm, or 3 if there are cancer cells in 10 or more lymph nodes in the armpit and at least one is larger than 2 mm."
# M SYSTEM="In the following text, determine whether the cancer has distantly metastasized. Distant metastasis refers specifically to cancer that has spread from the original (primary) tumor to distant organs or distant lymph nodes. Return 1 strictly if it has spread distantly, otherwise return 0. This answer should reflect the M in TNM staging. Return no text beyond the integer value."
SYSTEM="In the following text, determine the stage of T in the TNM staging. T can take on 4 integer values, 1 if tumor is 2cm or less across, 2 if tumor is more than 2cm but not more than 5cm across, 3 if tumor is more than 5cm across, or 4 if tumor of any size growing into the chest wall or skin. Return no text beyond the integer value."
RESPONSE_COL="t"
PREDICTOR_COL="text"

# Setup environment variables
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# Prepare the tokenizer and model
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.pad_token = tokenizer.eos_token

# Load and process dataset
df = pd.read_csv(DATASET_FILE)
hf_dataset = Dataset.from_pandas(df)

# Define processing functions
def tokenize(prompt):
    """Tokenize input text with specific options."""
    result = tokenizer(
        prompt[PREDICTOR_COL], truncation=True, max_length=5000, padding=True
    )
    result["labels"] = result["input_ids"].copy()  # Self-supervised learning setup
    return result

def format_chat_template(example):
    """Format the input for the chat-based model."""
    convo = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": example[PREDICTOR_COL]},
        {"role": "assistant", "content": (example[RESPONSE_COL])}
    ]
    formatted_chat = tokenizer.apply_chat_template(convo, tokenize=False)
    return {"text": formatted_chat}

# Apply formatting and tokenization
hf_dataset = hf_dataset.map(format_chat_template, batched=False)
split_dataset = hf_dataset.train_test_split(test_size=TRAIN_TEST_SPLIT_RATIO)
train_dataset = split_dataset['train']
test_dataset = split_dataset['test']


train_dataset= hf_dataset


tokenized_train_dataset = train_dataset.map(tokenize)
tokenized_val_dataset = test_dataset.map(tokenize)

model.train() # put model back into training mode
model = prepare_model_for_kbit_training(model)

config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=[
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, config)


print('1', flush=True)

# Configure training arguments
training_args = TrainingArguments(
    per_device_train_batch_size=2,    
    gradient_accumulation_steps=1,     
    warmup_steps=5,
    num_train_epochs=3,
    learning_rate=3e-4,
    fp16=True,
    logging_steps=10,
    optim="adamw_torch",
    evaluation_strategy="no",         
    save_strategy="steps",
    save_steps=500,
    output_dir=PROJECT_NAME,
    group_by_length=True,
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    data_collator=DataCollatorForSeq2Seq(
        tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
    ),
)
trainer.train()
trainer.save_model(str(PROJECT_NAME) + "_model")
