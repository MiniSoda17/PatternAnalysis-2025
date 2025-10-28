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
