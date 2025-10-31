# evaluate_and_examples.py
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch, random

# ---- CONFIG ----
BASE_MODEL_ID = "google/flan-t5-base"

DRIVE_MODEL_PATH = "./results_biolaysumm"

def sample_examples(ds, n=4, seed=123):
    random.seed(seed)
    idx = list(range(len(ds)))
    random.shuffle(idx)
    return [ds[i] for i in idx[:n]]

def run_flan_t5(model_dir=DRIVE_MODEL_PATH, max_new_tokens=256):

    # ---- Load tokenizer ----
    tok = AutoTokenizer.from_pretrained(BASE_MODEL_ID, use_fast=True)

    # ---- Load finetuned checkpoint instead of base model ----
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_dir,
        torch_dtype=torch.float16,
    )

    # Move to GPU / CPU
    model.to("cuda" if torch.cuda.is_available() else "cpu").eval()

    # Evaluate on first 1000 validation samples
    ds = dataset["validation"].select(range(3))

    preds, refs = [], []
    for ex in ds:
        inp = tok(
            f"Summarize for a patient:\n{ex['radiology_report']}",
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        ).to(model.device)

        out = model.generate(**inp, max_new_tokens=max_new_tokens)

        preds.append(tok.decode(out[0], skip_special_tokens=True))
        refs.append(ex["layman_report"])

    scores = compute_rouge(preds, refs)
    exs = sample_examples(dataset["validation"].select(range(3)))
    return scores, preds[:len(exs)], refs[:len(exs)], exs

if __name__ == "__main__":
    scores, preds, refs, exs = run_flan_t5()

    print("\n=== ROUGE SCORES ===")
    for k, v in scores.items():
        print(f"{k}: {v:.4f}")

    print("\n=== SAMPLE EXAMPLES ===\n")
    for i, ex in enumerate(exs):
        print(f"Example {i+1}:")
        print("Radiology Report:")
        print(ex["radiology_report"][:400], "...\n")
        print("Model Summary:")
        print(preds[i])
        print("Reference Summary:")
        print(refs[i])
        print("-" * 60)