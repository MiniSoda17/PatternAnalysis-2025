# train.py

import os
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer

from modules import get_model_components
from dataset import load_and_tokenize_data
from utils import compute_metrics

L_RATE = 3e-4
BATCH_SIZE = 8
PER_DEVICE_EVAL_BATCH = 4
WEIGHT_DECAY = 0.01
SAVE_TOTAL_LIM = 3
NUM_EPOCHS = 3
OUTPUT_DIR = "./results-biolaysumm"

def main():
    tokenizer, model, data_collator = get_model_components()

    tokenized_dataset = load_and_tokenize_data(tokenizer)

    training_args = Seq2SeqTrainingArguments(
        output_dir="/content/drive/MyDrive/My_Project_Checkpoints/ckpt_flan_t5/",
        eval_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=L_RATE,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        weight_decay=WEIGHT_DECAY,
        save_total_limit=SAVE_TOTAL_LIM,
        num_train_epochs=NUM_EPOCHS,
        predict_with_generate=True,
        push_to_hub=False,
        report_to="none"
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"].select(range(20000)),       # Use the 'train' split
        eval_dataset=tokenized_dataset["validation"].select(range(5000)),  # Use the 'validation' split
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

if __name__ == "__main__":
    main()