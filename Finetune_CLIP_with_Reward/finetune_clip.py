import json
from PIL import Image

from tqdm import tqdm
import requests


import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import clip
from transformers import CLIPProcessor, CLIPModel
import pandas as pd
import pickle

# Choose computation device
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Load pre-trained CLIP model
model, preprocess = clip.load("ViT-B/16", device=device, jit=False)

class PATADataset(Dataset):
    def __init__(self, image_paths, captions, preprocess):
        self.image_paths = image_paths
        self.captions = captions
        self.preprocess = preprocess

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx][:-1]
        caption = self.captions[idx]
        try:
            img = Image.open(requests.get(image_path, stream=True).raw)
            img = self.preprocess(img)
            text = clip.tokenize([caption]).to(device)
        except:
            print("error")
            return None
        return img, text

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

#read image_paths, caption, label from csv
unbiased = pd.read_csv('gpt_generated_for_finetune_clip.csv')
biased = pd.read_csv('biased_for_clip.csv')

unbiased_paths=list(unbiased.iloc[:, 1])
biased_paths=list(biased.iloc[:, 1])
image_paths=unbiased_paths+biased_paths
image_paths=biased_paths

unbiased_captions=list(unbiased.iloc[:, 2])
biased_captions=list(biased.iloc[:, 2])
captions=unbiased_captions+biased_captions
captions=biased_captions

dataset = PATADataset(image_paths, captions, preprocess)
images_dataloader = DataLoader(dataset, shuffle=True, collate_fn=collate_fn, batch_size=5)


if device == "cpu":
  model.float()

#load reward model
reward_model = pickle.load(open("reward_model_16.sav", 'rb'))

# Prepare the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2)

# Train the model
model.train()
num_epochs = 30
for epoch in range(num_epochs):
    model.train()
    pbar = tqdm(images_dataloader, total=len(images_dataloader))
    for batch in pbar:
        print("new round")
        optimizer.zero_grad()

        images,texts = batch

        images= images.to(device)
        texts=texts.squeeze(1)
        texts = texts.to(device)

        # Forward pass
        image_embedding=model.encode_image(images).cpu()
        text_embedding=model.encode_text(texts).cpu()
        img_text=torch.cat([image_embedding,text_embedding], dim=1)
        image_text_numpy=img_text.cpu().detach().numpy()
        # Compute loss using reward model
        reward_score=reward_model.predict_proba(image_text_numpy)[:, 1]
        total_loss=torch.tensor(reward_score, requires_grad=True)
        total_loss=total_loss.sum()

        # Backward pass
        total_loss.backward()
        optimizer.step()

        pbar.set_description(f"Epoch {epoch}/{num_epochs}, Loss: {total_loss.item():.4f}")

torch.save(model.state_dict(), 'clip_finetune_30epoch.pt')

