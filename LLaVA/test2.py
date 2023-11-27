'''
import debias_clip
from debias_clip.model.model import model_loader
model, img_preproc, tokenizer, alias_name = model_loader("openai/CLIP/RN50", "cuda")
model.eval()

model2, img_preproc2, tokenizer2, alias_name2 = model_loader("openai/CLIP/ViT-B/16", "cuda")
model2.eval()

# measure bias, lower == less biased
print(debias_clip.measure_bias(model, img_preproc, tokenizer, attribute="race"))
print(debias_clip.measure_bias(model2, img_preproc2, tokenizer2, attribute="race"))
'''
import torch
import clip
import debias_clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
deb_clip_model, preprocess = debias_clip.load("ViT-B/16-gender", device=device)
print(debias_clip.measure_bias(deb_clip_model, preprocess, clip.tokenize, attribute="gender"))

clip_model, preprocess = clip.load("RN50", device=device)
print(debias_clip.measure_bias(clip_model, preprocess, clip.tokenize, attribute="gender"))