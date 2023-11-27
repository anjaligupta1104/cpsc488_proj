#!/bin/bash

#SBATCH --job-name=debias_eval
#SBATCH --output=debias_job.txt
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH -p gpu --gpus=1 --reservation=cpsc488
#SBATCH --partition=gpu
#SBATCH --time=50:00
module load Python/3.10.8-GCCcore-12.2.0
pip install torch
pip install torchvision
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
pip install git+https://github.com/oxai/debias-vision-lang
python evaluate_CLIP.py