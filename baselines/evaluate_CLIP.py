import torch
import clip
from PIL import Image
import numpy as np
import math
import json
from transformers import CLIPProcessor, CLIPModel
import requests

def get_text_embedding(model, caption, device="cpu"):
    # tokenize the caption
    text = clip.tokenize([caption]).to(device)
    with torch.no_grad():
        text_embedding = model.encode_text(text)
        text_embedding /= text_embedding.norm(dim=-1, keepdim=True) # not sure why they do this, in measuring_bias.py
    return text_embedding

# TODO: modify this according to measuring_bias.py once we start using Dataloaders and batches
def get_img_embedding(model, image):
    with torch.no_grad():
        image_embedding = model.encode_image(image)

    return image_embedding

def calculate_similarity_scores(model, image_paths, caption, device="cpu"):
    '''
    Given model, images, caption, calculate similarity scores between each image and caption.
    '''

    # NOTE: wrote this before Anjali realized that CLIP automatically returns similarity scores
    # similarities = []
    # for image_path in image_paths:

    #     # preprocess should resize image appropriately, but if not, use 224x224
    #     image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

    #     with torch.no_grad():
    #         image_embedding = get_img_embedding(model, image)
    #         text_embedding = get_text_embedding(model, caption, device=device)
    #         similarity = torch.cosine_similarity(image_embedding, text_embedding).cpu().numpy()

    #     similarities.append(similarity[0])
    
    # return similarities

    #images = [Image.open(requests.get(path, stream=True).raw) for path in image_paths]
    images=image_paths
    inputs = processor(text=caption, images=images, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    similarities = outputs.logits_per_image

    return similarities

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

def calculate_skew_scores(model, images, captions, protected_labels, K=0, device="cpu"):
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

    for PA in ['races','gender','age']:
        labels = protected_labels[PA]
        max_skews = []
        min_skews = []

        for caption in captions[PA]:
            similarities = calculate_similarity_scores(model, images, caption, device=device)

            # only consider images that are above similarity threshold
            epsilon = 0.1 # adopted from Seth et al. 
            similarities, labels = zip(*[(sim, label) for sim, label in zip(similarities, labels) if sim >= epsilon])

            # @K
            if K > 0:
                similarities, labels = sorted(zip(similarities, protected_labels), reverse=True)[:K]

            # compute skew scores for labels, for each image w.r.t. caption
            skew_scores = skew(labels)

            # taking max across protected labels for each caption
            max_skews.append(max(skew_scores))
            min_skews.append(min(skew_scores))

        # taking average across all captions
        max_skew_pa = np.mean(max_skews)
        min_skew_pa = -1.0 * np.mean(min_skews)

        max_skew[PA] = max_skew_pa
        min_skew[PA] = min_skew_pa

    return max_skew, min_skew

if __name__ == "__main__":
    # Load the model
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Load dataset and captions
    #dataset_path = '../pata_dataset/'
    #with open(dataset_path.join('pata_fairness.captions.json'), 'r', encoding='utf-8') as f:
        #scene_captions = json.load(f)
    #(Wenni)changed this part of code because files could not be found
    with open('pata_dataset/pata_fairness.captions.json', 'r', encoding='utf-8') as f:
        scene_captions = json.load(f)

    pos_captions = {'races': [], 'gender': [], 'age': []}
    neg_captions = {'races': [], 'gender': [], 'age': []}
    for scene in scene_captions:
        for k in ['races', 'gender', 'age']:
            pos_captions[k].extend(scene['pos'][k])
            neg_captions[k].extend(scene['neg'][k])
    '''
    protected_labels = {'races': [], 'gender': [], 'age': []}
    image_paths = []
    with open('pata_dataset/pata_fairness.files.lst', 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            all_label, image_path = line.split('|')
            labels = all_label.split('_')

            image_paths.append(image_path)
            for k, v in zip(['races', 'gender', 'age'], labels[1:]): 
                protected_labels[k].append(v)
    '''
    num_images_for_test = 200
    #images = image_paths[:num_images_for_test]
    #protected_labels = protected_labels[:num_images_for_test]
    #(Wenni)Changed this part because protected_labels[:num_images_for_test] does not work for dictionary object
    protected_labels = {'races': [], 'gender': [], 'age': []}
    image_paths = []
    images_opened = []
    number_images=0
    with open('pata_dataset/pata_fairness.files.lst', 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            all_label, image_path = line.split('|')
            labels = all_label.split('_')
            try:
                temp=Image.open(requests.get(image_path, stream=True).raw)
                images_opened.append(temp)
                image_paths.append(image_path)
                for k, v in zip(['races', 'gender', 'age'], labels[1:]): 
                    protected_labels[k].append(v)
                number_images+=1
                if (number_images==num_images_for_test):
                    break
            except:
                continue
    images=image_paths
    print("The length is:")
    print(len(images))
                

    max_skew, min_skew = calculate_skew_scores(model,
                                               images_opened,
                                               pos_captions,
                                               protected_labels,
                                               K=50,
                                               device=device)
    print("MaxSkew:", max_skew)
    print("MinSkew:", min_skew)
