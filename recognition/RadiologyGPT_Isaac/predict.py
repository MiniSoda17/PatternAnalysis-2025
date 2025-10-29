# save as evaluate_and_examples.py
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from peft import PeftModel
import torch, random

# IMPORTANT: SET THESE TWO VARIABLES
# 1. The ID of the original model you started with
BASE_MODEL_ID = "google/flan-t5-base"

# 2. The full path to your saved checkpoint folder on Google Drive
DRIVE_MODEL_PATH = "/content/drive/MyDrive/My_Project_Checkpoints/ckpt_flan_t5_lora/checkpoint-63"

def sample_examples(ds, n=4, seed=123):
    random.seed(seed); idx = list(range(len(ds))); random.shuffle(idx)
    return [ds[i] for i in idx[:n]]

def run_flan_t5(model_dir=DRIVE_MODEL_PATH, max_new_tokens=256):
    # --- STEP 1: Load the base model and tokenizer ---
    tok = AutoTokenizer.from_pretrained(BASE_MODEL_ID, use_fast=True)

    # Load the base model. Using float16 is common for Colab to save VRAM.
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float16,
    )

    # --- STEP 2 & 3: Load the LoRA adapter and merge it ---
    # Load the adapter weights from the specified Drive path
    model = PeftModel.from_pretrained(base_model, model_dir)

    # Merge the LoRA weights into the base model and move to the device
    model = model.merge_and_unload()
    model.to("cuda" if torch.cuda.is_available() else "cpu").eval()

    # --- Inference Loop (Your original code continues here) ---
    # print(dataset["test"])
    ds = dataset["validation"].select(range(1000))
    preds, refs = [], []
    for ex in ds:
        # ... (rest of your generation code remains the same)
        inp = tok(f"Summarize for a patient:\n{ex['radiology_report']}", return_tensors="pt", truncation=True, max_length=1024).to(model.device)
        out = model.generate(**inp, max_new_tokens=max_new_tokens)
        preds.append(tok.decode(out[0], skip_special_tokens=True))
        refs.append(ex["layman_report"])

    scores = compute_rouge(preds, refs)
    exs = sample_examples(dataset["validation"].select(range(1000)))
    return scores, preds[:len(exs)], refs[:len(exs)], exs

# if __name__ == "__main__":
#     print("FLAN-T5:", run_flan_t5()[0])

if __name__ == "__main__":
    scores, preds, refs, exs = run_flan_t5()

    print("\n=== ROUGE SCORES ===")
    for k, v in scores.items():
        print(f"{k}: {v:.4f}") 

    print("\n=== SAMPLE EXAMPLES ===\n")
    for i, ex in enumerate(exs):
        print(f"Example {i+1}:")
        print("Radiology Report:")
        print(ex["radiology_report"][:400], "...\n")  # shorten long text
        print("Model Summary:")
        print(preds[i])
        print("Reference Summary:")
        print(refs[i])
        print("-" * 60)
