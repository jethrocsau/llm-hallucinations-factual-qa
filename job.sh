#!/bin/bash

#SBATCH -J slurm_job #Slurm job name
##SBATCH -t 48:00:00 #Maximum runtime of 48 hours
##SBATCH --mail-user=csauac@ust.hk
##SBATCH --mail-type=begin
##SBATCH --mail-type=end
#SBATCH -p normal
#SBATCH -N 1 -n 4 --gres=gpu:2
#SBATCH --account=mscbdt2024

# Setup runtime environment
cd home/csauac/msbd5002
conda create -y -n hallucination python=3.10 numpy scipy ipykernel pandas scikit-learn

# Install dependencies
source activate hallucination
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install git+https://github.com/huggingface/transformers.git
pip install matplotlib seaborn accelerate sentencepiece evaluate einops rouge-score gputil captum
pip install selfcheckgpt spacy
python -m spacy download en_core_web_sm

# Execute applications in parallel
srun python generation.py
