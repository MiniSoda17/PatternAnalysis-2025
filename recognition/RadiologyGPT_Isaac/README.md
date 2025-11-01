# Radiology terminology into layperson terms

# Introduction
The problem-space was to translate radiology reports into layperson summaries. To do this, we will train a pretrained encoder-decoder LLM, specifically Flan-t5 with large amounts of data. 

# Flan-T5 Architecture
The FLAN-T5 architecture is based off the T5 architecture shown in the image below. 
<img width="689" height="450" alt="Screenshot 2025-10-30 at 16 15 03" src="https://github.com/user-attachments/assets/13b49202-21db-4de1-9345-ef2f3dca95a4" />

The diagram demonstrates the encoder-decoder architecture used in the T5. A Text-to-Text Transformer. Essentially, converting text into other text. FLAN-t5 allows us to use the current T5 Architecture while having our own training on top of it. 

Essentailly, the encoder on the left receives the text and input and contextualises it using the self-attention and feedforward networks

# Training dataset
The training data was acquired from the BioLaySumm-2025-Layman dataset track from the https://huggingface.co/datasets/BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track from Huggingface.

The dataset contains a training, validation and test dataset. The training has 150k rows, the validation has 10k rows and the test has 10.5k rows. 

Each row contains an image_path, radiology_report, layman_report column. The radiology report is a snippet of medical literature from radiologists and the layman_report is a simplified translation of it that anyone should be able to understand. 

20k rows will be used for the training data and 10k will be used for the valuation data.

# Training


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

These were the 
| Hyperparamater | Description | Value |
|----------------|-------------|-------|
| num_epochs    | Number of training epochs   | 3 |
| learning rate | Initial learning rate for optimizer | 3e-4 |
| batch_size | Number of batches for training | 32 | 
| per device evaluation batch | Number of samples | 32 |
| weight decay | Regularisation to reduce large weights | 0.01 | 
| GPU Type | The GPU used to run and train model | NVIDIA's A100 |
| VRAM | Memory usage on app | 

# Results
The model was trained for 3 epochs with a learning rate of 3e-4 and a batch size of 32. It was trained on 20k of the dataset ()

|  Epoch  | Training loss | Validation Loss | Rogue1 | Rougue2 | Rouguel | Rougelsum |
|---------|---------------|-----------------|--------|---------|---------|-----------|
|  1  |    0.280420   | 0.280420 |    0.535829   |  0.397830 | 0.500228 | 0.514253 |
|  2  |    0.207000  | 0.263835 |    0.540682  |   0.405223  | 0.506848 | 0.520300 |
|  3  |    0.181800    | 0.258859 |    0.543166  |  0.409191  | 0.509671 | 0.522787|

<img width="853" height="545" alt="Screenshot 2025-10-31 at 22 29 23" src="https://github.com/user-attachments/assets/1fc14cc7-f1a6-4efb-921e-6c1ced01fad5" />




Final Testing


# Running the program


# References

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
