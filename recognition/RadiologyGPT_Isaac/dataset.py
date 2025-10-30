from datasets import load_dataset

tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
DATA_NAME = "BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track"
dataset = load_dataset(DATA_NAME)

def preprocess_function(examples):
    """Add prefix to the sentences, tokenize the text, and set the labels"""
    
    # The "inputs" are the tokenized radiology reports
    # We increase max_length as reports are longer than questions
    inputs = [prefix + doc for doc in examples["radiology_report"]]
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True)
    
    # The "labels" are the tokenized layman summaries
    labels = tokenizer(text_target=examples["layman_report"], 
                       max_length=512, 
                       truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Map the preprocessing function across our dataset
tokenized_dataset = dataset.map(preprocess_function, batched=True)
