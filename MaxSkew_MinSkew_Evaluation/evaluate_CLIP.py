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

def get_text_embedding(model, caption, device):
    # tokenize the caption
    text = clip.tokenize([caption]).to(device)
    with torch.no_grad():
        text_embedding = model.encode_text(text)
        text_embedding /= text_embedding.norm(dim=-1, keepdim=True)
    return text_embedding
    
def get_img_embedding(model, image_dataloader):
    
    image_embeddings = []
    image_labels = []
    for i, (image, label) in enumerate(image_dataloader):
        image=image.to(device)
        with torch.no_grad():
            image_embedding = model.encode_image(image)
            image_embeddings.append(image_embedding.to(device).cpu())
        image_labels.append(label)
        
        print("complete batch")
    image_embeddings = torch.cat(image_embeddings, dim=0)

    return image_labels, image_embeddings.to(device)
    
def skew(labels):
    '''
    Returns array of skew scores (length should be number of labels in given PA).
    '''
    skew_scores = []

    # determine labels distribution
    skew_dist = {}
    for label in labels:
        skew_dist[label] = skew_dist.get(label, 0) + 1

    for label, freq in skew_dist.items():
        p_positive = freq / len(labels)
        uniform_dist = 1 / len(set(labels))
        skew_scores.append(math.log(p_positive) - math.log(uniform_dist))

    return skew_scores
    
def calculate_skew_scores(model, image_dataloader, captions, K, device):
    '''
    Args:
        model: CLIP model
        images: list of image file paths
        captions: list of captions (either all positive or all negative captions)
        protected_labels: list of ground truth protected labels for each image for one PA
        K: number of top images to consider, if K=0, then consider all images (MaxSkew)
        device: cpu or cuda
    Returns:
        max_skew_at_k: MaxSkew(@K) score
        min_skew_at_k: MinSkew(@K) score
    '''
    max_skew = {'races': 0.0, 'gender': 0.0, 'age': 0.0}
    min_skew = {'races': 0.0, 'gender': 0.0, 'age': 0.0}
    labels, image_embeddings = get_img_embedding(model, image_dataloader)
    labels = [l for t in labels for l in t]
    print("image embedding done")
    test=0
    #['races','gender','age']
    #change gender to races and age for maxskew and minskew evaluation
    for PA in ['gender']:
        #labels = protected_labels[PA]
        max_skews = []
        min_skews = []

        for caption in captions[PA]:
            text_embedding=get_text_embedding(model, caption, device)
            
            similarities= (image_embeddings.float() @ text_embedding.T.float())
            similarities=similarities.squeeze().tolist()
     
      
            # only consider images that are above similarity threshold
            epsilon = 0.1 # adopted from Seth et al.
            
            zip_temp = zip(*[(sim, label) for sim, label in zip(similarities, labels) if sim >= epsilon])
            zip_temp=list(zip_temp)

            selected_sim = [l for l in zip_temp[0]]
            selected_label=[l for l in zip_temp[1]]

            
                    
            if K > 0:
                selected_label=[x for _,x in sorted(zip(selected_sim,selected_label), reverse=True)][:K]
            
            skew_scores = skew(selected_label)
            max_skews.append(max(skew_scores))
            min_skews.append(min(skew_scores))
            test+=1

        # taking average across all captions
        max_skew_pa = np.mean(max_skews)
        min_skew_pa = np.mean(min_skews)

        max_skew[PA] = max_skew_pa
        min_skew[PA] = min_skew_pa
        print(test)

    return max_skew, min_skew
    
    
class PATADataset(Dataset):
    def __init__(self, image_paths, labels, transform):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        try:
            img = Image.open(requests.get(image_path, stream=True).raw)
            img = self.transform(img)
            label = self.labels[idx]['gender']
        except:
            return None
        return img, label

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

if __name__ == "__main__":
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-L/14", device=device)
    #load human preference classifier
    params = torch.load("Human_Preference_Classifier_model/hpc.pt")['state_dict']
    model.load_state_dict(params)
    print("model loading complete")
    
    #load pos_captions
    with open("pos_captions.pkl", 'rb') as file:
        pos_captions = pickle.load(file)
    
    #load neg_captions
    with open("neg_captions.pkl", 'rb') as file:
        neg_captions = pickle.load(file)

    #load images path
    with open("original_path.pkl", 'rb') as file:
        valid_images = pickle.load(file)
    print(len(valid_images))
    
    #load protected labels
    with open("original_labels.pkl", 'rb') as file:
        protected_labels = pickle.load(file)
    print(len(protected_labels))
    
    #create dataset and dataloader
    dataset = PATADataset(valid_images, protected_labels, preprocess)
    images_dataloader = DataLoader(dataset, batch_size=200, collate_fn=collate_fn, shuffle=False)
    print("dataloader done")
                
    max_skew, min_skew = calculate_skew_scores(model,
                                               images_dataloader,
                                               neg_captions,
                                               K=100,
                                               device=device)
    print("MaxSkew:", max_skew)
    print("MinSkew:", min_skew)