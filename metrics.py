import math
from collections import Counter, defaultdict
from typing import Union, Tuple, Callable, List

import numpy as np
import pandas as pd
import torch
import torch.utils.data
from torch.utils.data import DataLoader
from tqdm import tqdm

from debias_clip import PROMPT_DATA_PATH
from debias_clip.datasets import IATDataset, FairFace
from debias_clip.model.model import ClipLike, model_loader

# NDKL and Skew metrics introduced in https://arxiv.org/pdf/1905.01989.pdf

# TODO: modify NDKL to work with same pipeline as MaxSkew
def normalized_discounted_KL(df: pd.DataFrame, top_n=25) -> dict:
    '''
    Implementation of NDKL metric.
    From debias-vision-lang/debias_clip/measuring_bias.py
        See for implementation of equality of opportunity and demographic parity. 
        Here we assume uniform distribution (equality of opportunity).

    Arg(s):
        df: pandas dataframe with columns "score" and "label"
            "score" is the cosine similarity between image and text embeddings
            "label" is label of image, one of the labels of a Protected Attribute
        top_n: number of top scores to consider [int]
    Returns:
        result_metric: NDKL score [float]
    '''

    def KL_divergence(p, q):
        return np.sum(np.where(p != 0, p * (np.log(p) - np.log(q)), 0))

    result_metric = 0.0

    # sort label counts by label
    _, label_counts = zip(*sorted(Counter(df.label).items()))  

    # if label count is 0, set it to 1 to avoid degeneracy
    if len(label_counts) == 0:
        desired_dist = np.array([1])
    else:
        desired_dist = np.array([1 / len(label_counts) for _ in label_counts]) # uniform distribution

    top_n_scores = df.nlargest(top_n, columns="score", keep="all")
    top_n_label_counts = np.zeros(len(label_counts))

    for index, (_, row) in enumerate(top_n_scores.iterrows(), start=1):
        label = int(row["label"])
        top_n_label_counts[label] += 1
        kl_div = KL_divergence(top_n_label_counts / index, desired_dist)
        result_metric += (kl_div / math.log2(index + 1))

    # normalize by Z
    Z = sum(1 / math.log2(i + 1) for i in range(1, top_n + 1))
    result_metric /= Z

    return result_metric

# TODO: harmful zero-shot image misclassification