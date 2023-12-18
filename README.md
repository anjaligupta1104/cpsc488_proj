# Social Debiasing of Large Vision-Language Models with RLAIF
CPSC 488 Final Project - Wenni Fan, Anjali Gupta, Suba Ramesh

# PATA Dataset
The original dataset used in this project can be found (here)[https://github.com/pata-fairness/pata_dataset]

## Training from Scratch

### Download the Human Preference Classifier model
The Human Preference Classifier from "Better Aligning Text-to-Image Models with Human Preference" (Wu et al. 2023) serves as our primary comparison. Because of size, we could not upload their trained model to GitHub.
Please download the pretrained human preference classifier from the [official repo](https://github.com/tgxs002/align_sd). After downloading the model, create a folder called "Human_Preference_Classifier_model" inside the "MaxSkew_MinSkew_Evaluation" folder and place the pretrained model in it.

### Train reward model
Please navigate to Reward_model folder and run ```python logistic_regression.py``` to train our logistic regression reward model. The model will be trained and saved as reward_model_16.sav.

In order to train the preference learning reward model, navigate to the Reward_model folder and run ```preference_learning.py``` to train and test the preference learning reward model. The model checkpoint will be save to model_1.pth

### Finetune CLIP
Please place reward model reward_model_16.sav inside the "Finetune_CLIP_with_Reward" folder, navigate to that folder, and run ```python finetune_clip.py``` to finetune the CLIP model using. The finetuned CLIP will be saved as clip_finetune_30epoch.pt.

In order to finetune clip with the preference learning reward model, change the torch load path on line 108 in ```finetune_clip_pref_model.py``` to the path to your saved checkpoint for the preference learning reward model. Then run this script and the finetuned CLIP weights will be saved as clip_finetune_30epoch.pt

### MaxSkew and MinSkew Evaluation
Please navigate to MaxSkew_MinSkew_Evaluation folder and run ```python evaluate_CLIP.py```. The default evaluation is of CLIP model aligned by Human Preference Classifier tested on bias for the gender attribute and measured by MaxSkew and MinSkew. Please change the code at the '#Evaluation' comment to test other attributes.

## Evaluation

### TODO
