from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding="longest")

DRIVE_PATH = "/content/drive/MyDrive/My_Project_Checkpoints/ckpt_flan_t5_lora"

training_args = Seq2SeqTrainingArguments(
    output_dir=DRIVE_PATH,
    learning_rate=2e-4,
    num_train_epochs=1,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,   # effective batch 32
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    predict_with_generate=True,
    generation_max_length=128,
    fp16=True,
    report_to="none"
)

import numpy as np # Make sure to import numpy at the top of your script

def compute_metrics(eval_pred):
    preds, labels = eval_pred

    # Clean predictions: Replace -100 with pad_token_id
    # preds is a numpy array. Use np.where for efficiency.
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)

    # Decode predictions
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Clean labels: Your original logic was fine, but np.where is cleaner
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    # Decode labels
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # (Assuming compute_rouge is defined elsewhere and expects two lists of strings)
    return compute_rouge(decoded_preds, decoded_labels)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"].select(range(1000)),
    eval_dataset=tokenized_dataset["validation"].select(range(100)),
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
