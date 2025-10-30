# Radiology terminology into layperson terms

# Introduction
The problem-space was to translate radiology reports into layperson summaries. To do this, we will train a pretrained encoder-decoder LLM, specifically Flan-t5 with large amounts of data

# Flan-T5 Architecture
<img width="689" height="450" alt="Screenshot 2025-10-30 at 16 15 03" src="https://github.com/user-attachments/assets/13b49202-21db-4de1-9345-ef2f3dca95a4" />

# Task

# Python Script Structure
1. modules.py -
2. dataset.py 
3. train.py
4. predict.py - Shows an e

# Dependencies
transformers (version 4.57.1)
datasets (version 4.3.0)
accelerate (version 1.11.0)
evaluate (version 0.4.6)
bitsandbytes(version 0.48.2)
peft (version 0.17.1)
rouge_scores (version 0.1.2)
nltk (version 3.9.2)

# Installation

# Training

# Results

# Running the program

# Reference

The use of Generative AI like Gemini and ChatGPT have been used for the learning process and code inspiration of this project

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
