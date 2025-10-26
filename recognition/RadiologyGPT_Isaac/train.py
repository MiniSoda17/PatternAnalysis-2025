from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model)
model = AutoModelForSeq2SeqLM.from_pretrained(model, load_in_8bit=True, device_map="auto")