import pandas as pd
import matplotlib.pyplot as plt
import os
import nltk
import evaluate
import numpy as np
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import T5Tokenizer
from modules import get_model_components
from dataset import load_and_tokenize_data

OUTPUT_DIR = "./results-biolaysumn"

try:
    nltk.download("punkt", quiet=True)
except LookupError:
    # Handle the specific error you previously encountered
    nltk.download("punkt_tab", quiet=True) 

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

def plot_loss_graph(trainer):
    """ Responsible for plotting the loss graph of training and validation during training of model """
    log_history = trainer.state.log_history
    log_df = pd.DataFrame(log_history)
    train_loss_df = log_df[log_df['loss'].notna()]
    eval_loss_df = log_df[log_df['eval_loss'].notna()]

    plt.figure(figsize=(10, 6))

    plt.plot(train_loss_df['step'], train_loss_df['loss'], label='Training Loss')
    plt.plot(eval_loss_df['step'], eval_loss_df['eval_loss'], label='Validation Loss', marker='o')

    plt.title('Training and Validation Loss Over Steps')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    """ Runs the entire training loop """
    tokenizer, model, data_collator = get_model_components()

    tokenized_dataset = load_and_tokenize_data(tokenizer)

    # Training arguments to be passed into Sequence 2 Sequence
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

    # Trains the model on the given parameters
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"].select(range(20000)),     # Using subset of the full 150k training dataset 
        eval_dataset=tokenized_dataset["validation"].select(range(5000)),  # Using subset of the full 10k validation dataset
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

    plot_loss_graph(trainer);
