# utils.py

import nltk
import numpy as np
import evaluate
from datasets import load_dataset
from transformers import T5Tokenizer

# --- Global Components ---
# Download the 'punkt' resource once for sentence segmentation
try:
    nltk.download("punkt", quiet=True)
except LookupError:
    # Handle the specific error you previously encountered
    nltk.download("punkt_tab", quiet=True) 

# Load the ROUGE metric
metric = evaluate.load("rouge")

# --- Preprocessing Function ---
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

# --- Evaluation Function ---
def compute_metrics(eval_preds, tokenizer: T5Tokenizer):
    """Computes ROUGE metrics for evaluation."""
    preds, labels = eval_preds

    # Decode predictions and labels
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # ROUGE-L expects newline after each sentence for better scoring
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    
    # Format the results nicely (e.g., to only show ROUGE-Lsum F1 score)
    # result = {k: round(v * 100, 4) for k, v in result.items()}
    
    return result