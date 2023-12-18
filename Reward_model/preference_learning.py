import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import clip

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm
from PIL import Image


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/16", device=device, jit=False)

class PATADataset(Dataset):
    def __init__(self, image_paths, biased_captions, unbiased_captions, labels, preprocess, model):
        self.image_paths = image_paths
        self.biased_captions = biased_captions
        self.unbiased_captions = unbiased_captions
        self.labels = labels
        self.preprocess = preprocess
        self.model=model

    def encode_caption(self, caption):
        text = clip.tokenize([caption]).to(device)
        text_embedding = self.model.encode_text(text)
        return text_embedding

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx][:-1]
        biased_caption = self.biased_captions[idx]
        unbiased_caption = self.unbiased_captions[idx]
        try:
          img = Image.open(requests.get(image_path, stream=True).raw)
          img = self.preprocess(img).unsqueeze(0).to(device)
          img_embedding = self.model.encode_image(img)

          biased_embedding = self.encode_caption(biased_caption)
          unbiased_embedding = self.encode_caption(unbiased_caption)
        except:
          return None

        return img_embedding, biased_embedding, unbiased_embedding
    

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


# Load Datasets
unbiased = pd.read_csv('gpt_generated.csv', header=None)
biased = pd.read_csv('biased.csv', header=None)

dataframe = biased.merge(unbiased, left_on=1, right_on=1).drop('0_y', axis=1)
dataframe = dataframe.apply(lambda x: x.astype(str).str.lower())

# Data Normalization
dataframe['2_x'] = dataframe['2_x'].str.replace("photo of ", "")
dataframe['2_y'] = dataframe['2_y'].str.replace("photo of ", "")
dataframe['2_y'] = dataframe['2_y'].str.replace(".", "", regex=False)
d = dataframe

# Create DataLoaders
dataset = PATADataset(d[1], d['2_x'], d['2_y'], d['0_x'], preprocess, model)

training_data_size = 0.8
train_size = int(training_data_size * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=3, num_workers=0, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=3, num_workers=0, shuffle=False, collate_fn=collate_fn)

# Reward Model

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


# Train Model
def learn_reward(dataloader, model, optimizer, num_iter, checkpoint_dir):
    loss_criterion = nn.CrossEntropyLoss()

    for epoch in range(num_iter):
        cum_loss = 0.0
        abs_rewards = 0.0

        pbar = tqdm(dataloader, total=len(dataloader))
        for sample in enumerate(pbar):
            batch, (image, biased, unbiased) = sample
            batch_size = len(biased)

            optimizer.zero_grad()

            # gt tells us if 0 or 1 is preferred ground truth
            image_text_0 = torch.zeros((batch_size, 1024)).to(device)
            image_text_1 = torch.zeros((batch_size, 1024)).to(device)

            # randomize position of biased and unbiased values
            gt = np.random.choice([0,1], batch_size)
            captions = torch.cat([unbiased, biased], dim = 1)

            # if gt = 0: 0 is unbiased, else 1 is unbiased
            for idx in range(batch_size):
              image_text_0[idx] = torch.cat([image[idx][0], captions[idx][gt[idx]]], dim=0).to(torch.float32).unsqueeze(0)
              image_text_1[idx] = torch.cat([image[idx][0], captions[idx][1-gt[idx]]], dim=0).to(torch.float32).unsqueeze(0)

            outputs, abs_rewards = model.forward(image_text_0, image_text_1)
            outputs=outputs.to(device)

            # calculate loss
            labels = torch.from_numpy(gt).long().to(device)
            loss = loss_criterion(outputs, labels)
            loss = loss.to(device)

            loss.backward()
            optimizer.step()

            item_loss = loss.item()
            cum_loss += item_loss

            pbar.set_description('sample output: {}'.format(outputs[0]))

        if True:
            print("epoch {} loss {}".format(epoch, cum_loss))
            print("check pointing")
            torch.save(model.state_dict(), checkpoint_dir)

    print("finished training")


def test_network(model, dataloader):
    correct = 0.0
    total = 0.0

    pbar = tqdm(dataloader, total=len(dataloader))
    for sample in enumerate(pbar):
        batch, (image, biased, unbiased) = sample
        batch_size = len(biased)

        with torch.no_grad():
          image_text_0 = torch.zeros((batch_size, 1024)).to(device)
          image_text_1 = torch.zeros((batch_size, 1024)).to(device)

          # randomize position of biased and unbiased values
          gt = np.random.choice([0,1], batch_size)
          captions = torch.cat([unbiased, biased], dim = 1)

          # if gt = 0: 0 is unbiased, else 1 is unbiased
          for idx in range(batch_size):
            image_text_0[idx] = torch.cat([image[idx][0], captions[idx][gt[idx]]], dim=0).to(torch.float32).unsqueeze(0)
            image_text_1[idx] = torch.cat([image[idx][0], captions[idx][1-gt[idx]]], dim=0).to(torch.float32).unsqueeze(0)

          outputs, abs_rewards = model.forward(image_text_0, image_text_1)
          outputs=outputs.to(device)
          prediction = torch.argmax(outputs, dim=1)
          labels = torch.from_numpy(gt).long().to(device)

          correct += sum(prediction == labels)
          total += batch_size
    return correct/total


if __name__ == "__main__":
    # Initialize Model, embedding size is 512
    reward_network = Net(512+512).to(device)

    optimizer = optim.Adam(reward_network.parameters(), lr=1e-3)
    #Train Model
    learn_reward(train_loader, reward_network, optimizer, 2, 'model_1.pth')
    #Test Model
    test_acc = test_network(reward_network, test_loader)
    print("Test Accuracy:", test_acc)