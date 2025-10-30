from datasets import load_dataset
from utils import preprocess_function

# --- Global Constants ---
DATA_NAME = "BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track"
TASK_PREFIX = "Summarize this radiology report in plain language: "

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