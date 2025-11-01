# train.py

import os
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer

from modules import get_model_components
from dataset import load_and_tokenize_data

OUTPUT_DIR = "./results-biolaysumn"

# --- Global Components ---
# Download the 'punkt' resource once for sentence segmentation
try:
    nltk.download("punkt", quiet=True)
except LookupError:
    # Handle the specific error you previously encountered
    nltk.download("punkt_tab", quiet=True) 

# Load the ROUGE metric
metric = evaluate.load("rouge")

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
    
    return result

def main():
    tokenizer, model, data_collator = get_model_components()

    tokenized_dataset = load_and_tokenize_data(tokenizer)

    training_args = Seq2SeqTrainingArguments(
        output_dir="./results-biolaysumm",
        eval_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=3e-4,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=3,
        predict_with_generate=True,
        push_to_hub=False,
        report_to="none"
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"].select(range(20000)),      
        eval_dataset=tokenized_dataset["validation"].select(range(5000)), 
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    print("-" * 50)
    print("Starting Training...")
    print("-" * 50)
    trainer.train()

    print("\nTraining complete! Model checkpoints are saved in:", OUTPUT_DIR)
    
    # Save the final model for prediction
    trainer.save_model(os.path.join(OUTPUT_DIR, "final_model"))
    tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final_model"))
