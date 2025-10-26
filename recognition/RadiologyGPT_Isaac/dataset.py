from datasets import load_dataset

dataset = load_dataset("BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track")

print(dataset)

print(dataset["train"][0])