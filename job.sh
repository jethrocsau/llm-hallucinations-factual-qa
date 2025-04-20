#!/bin/bash

#SBATCH -J llm-hallucination 
#SBATCH -t 48:00:00 
#SBATCH --mail-user=csauac@connect.ust.hk
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH -p normal
#SBATCH -N 1 -n 4 --gres=gpu:2
#SBATCH --account=mscbdt2024

# Setup runtime environment
cd msbd5002

# If need to install env
#module avail
#module load Anaconda3
#conda init
#conda create -y -n hallucination python=3.10 numpy scipy ipykernel pandas scikit-learn
#source activate hallucination
#pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
##pip install git+https://github.com/huggingface/transformers.git
#pip install matplotlib seaborn accelerate sentencepiece evaluate einops rouge-score gputil captum
#pip install selfcheckgpt spacy
#python -m spacy download en_core_web_sm

# Execute applications
source activate hallucination
srun python generation.py
