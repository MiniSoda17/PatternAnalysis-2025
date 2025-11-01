from transformers import T5Tokenizer, DataCollatorForSeq2Seq
from transformers import T5ForConditionalGeneration

MODEL_NAME = "google/flan-t5-base"

def get_module_components():
    """ Gets all the important components needed that are rquired for other programs within the program """
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    return tokenizer, model, data_collator



