#!/bin/bash

#SBATCH --job-name=llava_rlhf
#SBATCH --output=llava_job.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH -p gpu --gpus=4 --reservation=cpsc488
#SBATCH --partition=gpu
#SBATCH --time=50:00
module load Python/3.10.8-GCCcore-12.2.0
pip install --upgrade pip
pip install -e .
pip install clip
pip install gdown
python test.py