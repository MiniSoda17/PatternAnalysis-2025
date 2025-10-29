# Radiology terminology into layperson terms
Fine tune a pretrained encoder-decoder LLM such as T5 [19] or FLAN-T5 , or decoder-only LLM
such as GPT-2 , to translate expert radiology reports into layperson summaries using the BioLaySumm dataset [Subtask 2.1 of ACL 2025 BioLaySumm workshop]. Evaluate on a held-out test split using
ROUGE Scores (rouge1, rouge2, rougeL, rougeLsum). In your report, include model and parameter count,
fine-tuning strategy (full vs parameter-efficient such as LoRA), GPU type, VRAM, epochs, and total training time. Also, include 3–5 representative input–output examples with a short error analysis paragraph.

Tasks may include:
* Image Segmentation
* Object detection
* Graph node classification
* Image super resolution
* Disease classification
* Generative modelling with StyleGAN and Stable Diffusion