from datasets import load_dataset
from utils import preprocess_function
from transformers import T5Tokenizer

# --- Global Constants ---
DATA_NAME = "BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track"
TASK_PREFIX = "Summarize this radiology report in plain language: "

def clean(sample):
    radiology_input = sample.get("radiology_report", "")
    layman_input = sample.get("layman_report", "")
    return bool(radiology_input.strip()) and bool(layman_input.strip())


def preprocess_function(examples, tokenizer: T5Tokenizer, prefix: str):
    """Adds the task prefix, tokenizes the text, and sets the labels."""
    
    # Inputs: Report text + Prefix
    inputs = [prefix + rad for rad in examples["radiology_report"]]
    model_inputs = tokenizer(inputs, max_length=256, truncation=True, padding="longest")
    
    # Labels: Layman summary text
    labels = tokenizer(text_target=examples["layman_report"], max_length=128, truncation=True, padding="longest")

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def load_and_tokenize_data(tokenizer):
    """
    Loads the raw dataset and applies the preprocessing and tokenization.
    """
    print(f"Loading dataset: {DATA_NAME}")
    dataset = load_dataset(DATA_NAME)
    dataset = dataset.filter(clean)

    # Use a lambda function to pass the tokenizer and prefix to the utility function
    tokenized_dataset = dataset.map(
        lambda x: preprocess_function(x, tokenizer=tokenizer, prefix=TASK_PREFIX), 
        batched=True,
        remove_columns=['radiology_report', 'layman_report', 'id']
    )
    
    print("Dataset tokenization complete.")
    return tokenized_dataset