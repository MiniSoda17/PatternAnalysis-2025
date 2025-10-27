from datasets import load_dataset
from transformers import AutoTokenizer

dataset = load_dataset("BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track")
dataset = dataset.remove_columns(["source", "images_path"])
print(dataset)

training_data = dataset["train"]
validation_data = dataset["validation"]
test_data = dataset["test"]

model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

MAX_INPUT = 1024
MAX_TARGET = 256

def preprocess_function(batch):
    inputs = [f"Summarize the following radiology report for a patient:\n{r}"
              for r in batch["radiology_report"]]
    targets = batch["layman_report"]

    model_inputs = tokenizer(inputs, max_length=MAX_INPUT, truncation=True)
    labels = tokenizer(targets, max_length=MAX_TARGET, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = dataset.map(preprocess_function,
                                batched=True,
                                remove_columns=dataset["train"].column_names)
