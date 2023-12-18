from PIL import Image

from tqdm import tqdm
import requests


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import clip
import pandas as pd

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

# Function to convert model's parameters to FP32 format
def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        p.grad.data = p.grad.data.float()


if device == "cpu":
  model.float()

"""## load reward model"""

class Net(nn.Module):
  def __init__(self, in_dim):
    super().__init__()
    self.fc1 = nn.Linear(in_dim, 512)
    self.fc2 = nn.Linear(512, 1)

  def cum_return(self, sample):
    sum_rewards = 0
    sum_abs_rewards = 0

    x = F.leaky_relu(self.fc1(sample))
    x = F.sigmoid(self.fc2(x))

    sum_rewards += torch.sum(x)
    sum_abs_rewards += torch.sum(torch.abs(x))

    return sum_rewards, sum_abs_rewards

  def forward_individual(self, sample):
    x = F.leaky_relu(self.fc1(sample))
    x = F.sigmoid(self.fc2(x))

    rewards = x
    return rewards, abs(rewards)

  def forward(self, sample_i, sample_j):
    cum_r_i, abs_r_i = self.forward_individual(sample_i)
    cum_r_j, abs_r_j = self.forward_individual(sample_j)
    rewards_0_1 = torch.cat((cum_r_i, cum_r_j), dim=1)
    return rewards_0_1, abs_r_i + abs_r_j

#load reward model
reward_model = Net(512 +512)
reward_model.load_state_dict(torch.load('model.pth'))
reward_model.eval()

# Prepare the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2) # the lr is smaller, more safe for fine tuning to new dataset


# Specify the loss function
loss_img = nn.CrossEntropyLoss()
loss_txt = nn.CrossEntropyLoss()

# Train the model
model.train()
num_epochs = 30
for epoch in range(num_epochs):
    model.train()
    pbar = tqdm(images_dataloader, total=len(images_dataloader))
    for batch in pbar:
        optimizer.zero_grad()

        images,texts = batch

        images= images.to(device)
        texts=texts.squeeze(1)
        texts = texts.to(device)

        image_embedding=model.encode_image(images).cpu()
        text_embedding=model.encode_text(texts).cpu()
        img_text=torch.cat([image_embedding,text_embedding], dim=1)
        image_text_numpy=img_text.cpu().detach().numpy()


        reward_score, abs_reward_score=reward_model.forward_individual(torch.from_numpy(image_text_numpy).to(torch.float32))

        total_loss= -1 * reward_score.sum()

        # Backward pass
        total_loss.backward()
        optimizer.step()

        pbar.set_description(f"Epoch {epoch}/{num_epochs}, Loss: {total_loss.item():.4f}")

torch.save(model.state_dict(), 'clip_finetune_30epoch.pt')

model_base, preprocess_base = clip.load("ViT-B/16", device=device, jit=False)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/16", device=device, jit=False)

params = torch.load("clip_finetune_30epoch_biased.pt", map_location=torch.device('cpu'))
model.load_state_dict(params)
print("model loading complete")
