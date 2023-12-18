# Social Debiasing of Large Vision-Language Models with RLAIF
CPSC 488 Final Project - Wenni Fan, Anjali Gupta, Suba Ramesh

## Training from Scratch

### Download the Human Preference Classifier model
The Human Preference Classifier from "Better Aligning Text-to-Image Models with Human Preference" (Wu et al. 2023) serves as our primary comparison. Because of size, we could not upload their trained model to GitHub.
Please download the pretrained human preference classifier from the [official repo](https://github.com/tgxs002/align_sd). After downloading the model, create a folder called "Human_Preference_Classifier_model" inside the "MaxSkew_MinSkew_Evaluation" folder and place the pretrained model in it.

### Train reward model
Please navigate to Reward_model folder and run ```python logistic_regression.py``` to train our logistic regression reward model. The model will be trained and saved as reward_model_16.sav.

### Finetune CLIP
Please place reward model reward_model_16.sav inside the "Finetune_CLIP_with_Reward" folder, navigate to that folder, and run ```python finetune_clip.py``` to finetune the CLIP model using. The finetuned CLIP will be saved as clip_finetune_30epoch.pt.

### MaxSkew and MinSkew Evaluation
Please navigate to MaxSkew_MinSkew_Evaluation folder and run ```python evaluate_CLIP.py```. The default evaluation is of CLIP model aligned by Human Preference Classifier tested on bias for the gender attribute and measured by MaxSkew and MinSkew. Please change the code at the '#Evaluation' comment to test other attributes.

## Evaluation

### TODO
