# Social Debiasing of Large Vision-Language Models with RLAIF
CPSC 488 Final Project - Wenni Fan, Anjali Gupta, Suba Ramesh

### How to get MaxSkew and MinSkew for CLIP with Human Preference Classifier
Please download pretrained human preference classifier from the [official repo](https://github.com/tgxs002/align_sd) for "Better Aligning Text-to-Image Models with Human Preference" project. After downloading the model, create a folder called "Human_Preference_Classifier_model" inside "MaxSkew_MinSkew_Evaluation" folder and put the pretrained model in it.

### Train reward model
Please navigate to Reward_model folder and run python logistic_regression.py for getting the logistic regression reward model. The model will be trained and saved as reward_model_16.sav

### Finetune CLIP
Please put the reward model reward_model_16.sav inside Finetune_CLIP_with_Reward folder, navigate to that folder, and run finetune_clip.py for getting the finetuned CLIP model using logistic regression model as the reward model. The finetuned CLIP will be saved as clip_finetune_30epoch.pt

### MaxSkew and MinSkew Evaluation
Please navigate to MaxSkew_MinSkew_Evaluation folder and run python evaluate_CLIP.py. The default value is MaxSkew and MinSkew scores for CLIP model with Human Preference Classifier for gender attribute. Please change the code where has '#Evaluation' comment for testing other attributes.
