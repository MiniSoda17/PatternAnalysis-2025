from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForSeq2SeqLM

model_name = "google/flan-t5-base"

model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name,
    load_in_8bit=True,
    device_map="auto"
)

lora_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q","v","k","o","wi","wo"],
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


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
    predict_with_generate=True,
    generation_max_length=128,
    logging_steps=100,
    fp16=True,
    report_to="none"
)
