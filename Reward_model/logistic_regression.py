from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import torch
import clip
from PIL import Image
import numpy as np
import math
import json
from transformers import CLIPProcessor, CLIPModel
import requests
from torch.utils.data import Dataset, DataLoader
import pickle
import pandas as pd

if __name__ == "__main__":
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/16", device=device)
    
    #read image_paths, caption, label from csv
    unbiased = pd.read_csv('gpt_generated.csv')
    biased = pd.read_csv('biased.csv')
    
    unbiased_paths=list(unbiased.iloc[:, 1])
    biased_paths=list(biased.iloc[:, 1])
    image_paths=unbiased_paths+biased_paths
    
    unbiased_captions=list(unbiased.iloc[:, 2])
    biased_captions=list(biased.iloc[:, 2])
    captions=unbiased_captions+biased_captions
    
    unbiased_labels = [0] * len(unbiased_paths)
    biased_labels = [1] * len(biased_paths)
    labels=unbiased_labels+biased_labels
    
    X=[]
    Y=[]
    for img_index in range(len(image_paths)):
      if (img_index%10==0):
        print(img_index)
      try:
        image_path = image_paths[img_index][:-1]
        img = Image.open(requests.get(image_path, stream=True).raw)
        img = preprocess(img).unsqueeze(0).to(device)
        img_embedding = model.encode_image(img).cpu()
        text = clip.tokenize([captions[img_index]]).to(device)
        text_embedding = model.encode_text(text).cpu()
        img_text=torch.cat([img_embedding,text_embedding], dim=1)
        image_text_numpy=img_text.cpu().view(-1).detach().numpy()
        X.append(image_text_numpy)
        Y.append(labels[img_index])
      except:
        print("error")
        continue
        
    #train test split
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    reward_model = LogisticRegression()
    
    reward_model.fit(X_train, y_train)
    
    y_pred = reward_model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    
    print(f'Accuracy: {accuracy}')
    print('Classification Report:\n', report)
    print('Confusion Matrix:\n', cm)
    
    pickle.dump(reward_model, open("reward_model_16.sav", 'wb'))
            
    
        
