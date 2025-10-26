from datasets import load_dataset

dataset = load_dataset("BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track")
dataset = dataset.remove_columns(["source", "images_path"])
print(dataset)

training_data = dataset["train"]
validation_data = dataset["validation"]
test_data = dataset["test"]
