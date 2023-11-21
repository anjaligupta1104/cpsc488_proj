from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import clip
from transformers import CLIPProcessor, CLIPModel

# load pre-trained VLM to be finetuned
'''
CLIP can be used for image-text similarity and for zero-shot image classification.
'''
# TODO: generate input
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# example of inference
# # TODO: image = ___

# inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)

# outputs = model(**inputs)
# logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
# scores = logits_per_image.softmax(dim=1)  # we can take the softmax

# load pre-trained reward model
# TODO

# RLAIF
# TODO: pass to trainer and train