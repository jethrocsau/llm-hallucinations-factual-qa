#!/bin/bash

#SBATCH -J hallucination
#SBATCH -t 12:00:00
#SBATCH --mail-user=csauac@connect.ust.hk
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH -p normal
#SBATCH --nodes=1 --gpus-per-node=1
#SBATCH --account=mscbdt2024

# Setup runtime environment
cd msbd5002

# If need to install env
#module avail
#module load Anaconda3
#conda init
#conda create -y -n hallucination_slurm python=3.10 numpy scipy ipykernel pandas
#source activate hallucination_slurm
#pip install scikit-learn
#pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
#pip install git+https://github.com/huggingface/transformers.git
#pip install matplotlib seaborn accelerate sentencepiece evaluate einops rouge-score gputil captum
#pip install selfcheckgpt spacy
#python -m spacy download en_core_web_sm

# Execute applications
source activate hallucination_slurm
srun python generate.py --start 0 --end 1000 --n 3
