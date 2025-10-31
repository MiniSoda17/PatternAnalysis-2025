from datasets import load_dataset
from utils import preprocess_function

# --- Global Constants ---
DATA_NAME = "BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track"
TASK_PREFIX = "Summarize this radiology report in plain language: "

dataset = load_dataset(DATA_NAME)

def clean_up(example):
    src = example.get("radiology_report", "")
    tgt = example.get("layman_report", "")
    return bool(src.strip()) and bool(tgt.strip())

# Filter out examples with empty or missing fields
dataset = dataset.filter(clean_up)

# Remove unused columns safely (only if they exist)
cols_to_remove = [c for c in ["source", "images_path"] if c in dataset.column_names]
dataset = dataset.remove_columns(cols_to_remove)

def preprocess_function(examples, tokenizer: T5Tokenizer, prefix: str):
    """Adds the task prefix, tokenizes the text, and sets the labels."""
    
    # Inputs: Report text + Prefix
    inputs = [prefix + doc for doc in examples["radiology_report"]]
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True)
    
    # Labels: Layman summary text
    labels = tokenizer(text_target=examples["layman_summary"], 
                       max_length=512, 
                       truncation=True)

    # Add labels for the trainer
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def load_and_tokenize_data(tokenizer):
    """
    Loads the raw dataset and applies the preprocessing and tokenization.
    """
    print(f"Loading dataset: {DATA_NAME}")
    dataset = load_dataset(DATA_NAME)

    # Use a lambda function to pass the tokenizer and prefix to the utility function
    tokenized_dataset = dataset.map(
        lambda x: preprocess_function(x, tokenizer=tokenizer, prefix=TASK_PREFIX), 
        batched=True,
        remove_columns=['radiology_report', 'layman_report', 'id']
    )
    
    print("Dataset tokenization complete.")
    return tokenized_dataset