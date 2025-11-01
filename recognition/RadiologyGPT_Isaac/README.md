# Radiology terminology into layperson terms

By Isaac Arli - 47204296

# Introduction
The problem-space is to translate radiology reports into layperson summaries. To do this, we will train and fine-tune a pretrained encoder-decoder model, specifically the T5 transformer using the BioLaySumm dataset from Huggingface. The results will then be calculated using Rouge and we will test the output of our trained model

# Flan-T5 Architecture
The FLAN-T5 architecture is based off the T5 architecture shown in the image below. 
<img width="689" height="450" alt="Screenshot 2025-10-30 at 16 15 03" src="https://github.com/user-attachments/assets/13b49202-21db-4de1-9345-ef2f3dca95a4" />

The diagram demonstrates the encoder-decoder architecture used in the T5. A Text-to-Text Transformer. Essentially, converting text into other text. FLAN-t5 allows us to use the current T5 Architecture while having our own training on top of it. 

Within the T5 Architecture, it contains two components, the Encoder and Decoder. 
* Encoder: Responsible for converting input text into a sequence of contextual vectors through a series of self-attention and feed-forward networks. The result is then fed onto the Decoder for generation training. 
* Decoder: Responsible for generating the output token-by-token. 

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
transformers (version 4.57.1) \
datasets (version 4.3.0) \
accelerate (version 1.11.0) \
evaluate (version 0.4.6) \
bitsandbytes(version 0.48.2) \
peft (version 0.17.1) \
rouge_scores (version 0.1.2) \
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

Sample examples

Example 1: \
Radiology Report: \
The study is suboptimal due to poor inspiration. There are questionable faint infiltrates in the right upper and lower lobes. ...

Model Summary: \
The chest x-ray shows a lot of trapped air. There are long-term changes at the top of both lungs. The upper back is curved more than usual. There is no sign of air in the lungs.

Reference Summary: \
The chest shows a large amount of trapped air. There are long-term changes at the top of both lungs. The upper back is curved outward. There is no sign of air in the space around the lungs.

------------------------------------------------------------
Example 2: \
Radiology Report: \
Suboptimal study. No clear evidence of pneumothorax is identified. Increased density in the right base is likely related to atelectasis, consolidation, or pleural effusion. Bilateral pleural effusion is present. ...

Model Summary: \
A central venous catheter is going through the left jugular vein and its tip is in the superior vena cava. Everything else looks the same as before.

Reference Summary: \
A central venous catheter is going through the left jugular vein and its tip is in the superior vena cava. Everything else is the same as before.

--------------------------------------------------------
Example 3: \
Radiology Report: \
Technique performed: non-contrast CT of the chest with helical acquisition. Transverse reconstructions of 1mm with a lung filter and 1mm with a mediastinal filter. Patchy ground-glass opacities and small peripheral consolidations, as well as signs of organization with subpleural lines in posterior segments. Findings suggestive of evolving COVID-19 infection with signs of organization. No lymphaden ...

Model Summary: \
The patient has long-term lung changes.

Reference Summary: \
Long-term changes in the lungs are seen.

---------------------------------------------------------
Example 4:
Radiology Report:
Biapical pleural thickening. No significant radiological findings. ...

Model Summary:
The x-ray shows signs of air being trapped in the lungs, the diaphragm is flattened, and there's more space behind the breastbone. There are hardened areas on the pleura of the left lung, which are the lining around the lungs. The left lung has lost some volume and there are linear shadows below the pleura. These findings are related to long-term inflammation due to exposure to asbestos. Looking at the previous CT scan, there are no major changes compared to the scanogram dated 3/4/2009.

Reference Summary:
The X-ray shows signs of trapped air, a flattened muscle under the lungs, and more space behind the breastbone. There are also hardened areas on the lung lining on the left side. The left lung has lost some volume and has some linear shadows near the outer lining. These findings are related to long-term inflammation caused by exposure to asbestos. Looking at the previous CT scan, there are no significant changes compared to the scanogram dated 3/4/2009.

------------------------------------------------------------

Final Testing
| Rouge1 | Rouge2 | Rougel | Rougelsum |
|--------|---------|---------|-----------|
| 0.7188 | 0.5298  | 0.6652  |  0.6651   |

# Error Analysis
Overall, the model performed very well, as seen by the high Rouge scores and generated sample examples which closely matched the referenced samples. In the generated samples, there were minor differences, but these were primarily stylistic, just with different word ordering or vocab choices. It was able to convert both short complex radiology reports into longer simpler layman sentences and also convert long complex radiology reports into short and simple layman sentences, showing robust ability to handle different kinds of data. The slight variation shows there are things that could be improved, but nonetheless shows great readability. A Rougelsum of 0.6651 shows close to near accurate translation but could be slighlty improved. Future work could involve testing on larger and more varied datasets and have it actually be confirmed with radiologists to confirm translations are accurate not only in the dataset but also the model output. 


# Running the program


# References

The use of Generative AI like Gemini and ChatGPT have been used for the learning process and code inspiration of this project



Tasks may include:
* Image Segmentation
* Object detection
* Graph node classification
* Image super resolution
* Disease classification
* Generative modelling with StyleGAN and Stable Diffusion
