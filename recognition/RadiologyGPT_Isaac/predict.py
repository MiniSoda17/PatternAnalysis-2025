from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch, random
from train import main
import evaluate
import tqdm

BASE_MODEL_ID = "google/flan-t5-base"
DRIVE_MODEL_PATH = "./results_biolaysumm"

rouge = evaluate.load("rouge")

def compute_rouge(preds, refs):
    """ Computes the rouge score by comparing model output and reference output"""
    res = rouge.compute(predictions=preds, references=refs, use_stemmer=True)

    return {k: float(res[k]) for k in ["rouge1","rouge2","rougeL","rougeLsum"]}

def sample_examples(ds, n=5, seed=123):
    """ Chooses a certain number of examples to test on trained model"""
    random.seed(seed)
    idx = list(range(len(ds)))
    random.shuffle(idx)
    return [ds[i] for i in idx[:n]]

def run_flan_t5(dataset, model_dir=DRIVE_MODEL_PATH, max_new_tokens=256):
    """ Runs the pre-trained FLAN-T5 from the environment """
    tok = AutoTokenizer.from_pretrained(BASE_MODEL_ID, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_dir,
        torch_dtype=torch.float16,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    ds = dataset["validation"]

    preds, refs = [], []

    print(f"\n🔍 Evaluating on {len(ds)} validation samples...\n")

    # Uses tqdm to show progress
    for ex in tqdm(ds, desc="Generating summaries", ncols=100):
        inp = tok(
            f"Summarize for a patient:\n{ex['radiology_report']}",
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        ).to(device)

        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=max_new_tokens)

        preds.append(tok.decode(out[0], skip_special_tokens=True))
        refs.append(ex["layman_report"])

    print("\n✅ Generation complete! Computing ROUGE scores...\n")

    scores = compute_rouge(preds, refs)

    # Shows a few random examples
    exs = sample_examples(dataset["validation"].shuffle(seed=36).select(range(6)))
    return scores, preds[:len(exs)], refs[:len(exs)], exs


if __name__ == "__main__":
    """ Responsible for running the training and prediction of pretrained model """
    main()
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