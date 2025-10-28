from datasets import load_from_disk
from transformers import AutoTokenizer
import evaluate, numpy as np
import torch

def load_biolaysumm(path="biolaysumm_hf"):
    return load_from_disk(path)

def get_tokenizer(model_name):
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tok.pad_token is None:  # GPT-2 needs a pad token
        tok.pad_token = tok.eos_token
    return tok

rouge = evaluate.load("rouge")

def compute_rouge(preds, refs):
    # preds/refs are lists[str]
    res = rouge.compute(predictions=preds, references=refs, use_stemmer=True)
    # keep the four required
    return {k: float(res[k]) for k in ["rouge1","rouge2","rougeL","rougeLsum"]}

def pretty_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_params": int(total), "trainable_params": int(trainable)}