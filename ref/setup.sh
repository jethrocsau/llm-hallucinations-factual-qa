#!/bin/bash

# Create and activate conda environment
conda create -y -n hallucination python=3.10
conda activate hallucination

# Load CUDA module - adjust version as needed for your cluster
module load cuda11.8/toolkit/11.8.0
echo "Loaded CUDA module for GPU support"

echo "Setting up hallucination detection environment for GPU cluster..."

# Install basic dependencies first
pip install certifi urllib3 distro>=1.7.0 
pip install packaging>=20.9 pyyaml>=5.1 pytz>=2020.1 six>=1.14.0 jinja2

# Install numpy with the right version and fsspec
pip install "numpy>=1.22.4,<2.0.0"  # Compatible with captum but newer for pandas
pip install "fsspec[http]>=2023.1.0,<=2024.12.0"  # Version compatible with datasets

# Install scientific packages that depend on numpy
pip install scipy matplotlib!=3.6.1,>=3.4 
pip install scikit-learn  # Explicitly install scikit-learn

# Install pandas and related visualization
pip install pandas==2.2.3  # Specific version
pip install seaborn==0.13.2  # Specific version

# GPU setup for cluster environment
echo "Setting up GPU environment for cluster..."

# Set visible CUDA devices - if needed for your cluster
# Uncomment and modify the line below if you need to specify certain GPUs
# export CUDA_VISIBLE_DEVICES=0,1,2,3

# Install CUDA toolkit using conda
conda install -y cudatoolkit=11.8
conda install -y -c conda-forge cudnn=9.0

# Set up library paths for CUDA - important for cluster environments
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export CUDA_HOME=$CONDA_PREFIX

# Install PyTorch with specific CUDA version for cluster compatibility
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

# Verify GPU availability
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count()); print('GPU name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"

# First install huggingface-hub with compatible version to avoid dependency conflicts
pip install huggingface-hub>=0.24.0

# Install transformers with specific version for compatibility
pip install transformers==4.35.0

# Verify numpy version after transformers installation
python -c "import numpy; print(f'NumPy version after transformers install: {numpy.__version__}')"

# Install dependencies that depend on torch/transformers
pip install captum==0.8.0  # Version that needs numpy<2.0
pip install bert-score==0.3.13
pip install accelerate sentencepiece evaluate rouge-score gputil einops

# Install NLP tools with specific version of spacy to avoid thinc conflict
pip install spacy==3.6.1  # Use older spacy version to avoid thinc 8.3.6 (needs numpy>=2.0)
python -m spacy download en_core_web_sm

# Install selfcheckgpt with compatible version
pip install selfcheckgpt==0.0.9  # Earlier version compatible with our dependencies

# Install datasets with fixed version
pip install datasets==3.5.0

# Verify dependencies after installation
python -c "import huggingface_hub; print(f'huggingface-hub version: {huggingface_hub.__version__}'); import datasets; print(f'datasets version: {datasets.__version__}'); import accelerate; print(f'accelerate version: {accelerate.__version__}')"

# Create GPU test script
cat > test_gpu.py << 'EOL'
#!/usr/bin/env python
import os
import sys
import time
import torch
import numpy as np

def test_gpu():
    print("\n=== GPU Environment Test ===")
    
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"✅ CUDA is available")
        print(f"   CUDA version: {torch.version.cuda}")
        device_count = torch.cuda.device_count()
        print(f"   GPU count: {device_count}")
        
        # Show details for each GPU
        for i in range(device_count):
            print(f"\n   GPU {i} details:")
            print(f"   - Name: {torch.cuda.get_device_name(i)}")
            print(f"   - Memory: {torch.cuda.get_device_properties(i).total_memory / (1024**3):.2f} GB")
            
        # Test GPU compute capability
        print("\n   Running GPU compute test...")
        
        # Create test tensors
        start = time.time()
        size = 5000
        a = torch.randn(size, size, device="cuda")
        b = torch.randn(size, size, device="cuda")
        torch.cuda.synchronize()
        init_time = time.time() - start
        
        # Matrix multiply test
        start = time.time()
        c = torch.matmul(a, b)
        torch.cuda.synchronize()
        matmul_time = time.time() - start
        
        # Memory bandwidth test
        start = time.time()
        d = torch.sin(a) + torch.cos(b)
        torch.cuda.synchronize()
        elementwise_time = time.time() - start
        
        print(f"   - Tensor creation time: {init_time:.2f}s")
        print(f"   - Matrix multiply time ({size}x{size}): {matmul_time:.2f}s")
        print(f"   - Elementwise ops time: {elementwise_time:.2f}s")
        
        # Check transformer performance
        print("\n   Testing transformer model initialization...")
        try:
            start = time.time()
            from transformers import BertModel
            model = BertModel.from_pretrained("bert-base-uncased")
            model = model.to("cuda")
            torch.cuda.synchronize()
            model_time = time.time() - start
            print(f"   - Transformer model load time: {model_time:.2f}s")
            print(f"   ✅ Transformer model loaded successfully on GPU")
        except Exception as e:
            print(f"   ❌ Error loading transformer model: {e}")
    else:
        print("❌ CUDA is NOT available. GPU tests skipped.")
        print("   Please check your CUDA installation or GPU drivers.")
        
    print("\n=== Environment Variables ===")
    cuda_vars = [var for var in os.environ.keys() if "CUDA" in var]
    if cuda_vars:
        for var in cuda_vars:
            print(f"   {var}={os.environ.get(var)}")
    else:
        print("   No CUDA environment variables found.")
    
    print("\n=== GPU Test Complete ===")

if __name__ == "__main__":
    test_gpu()
EOL

chmod +x test_gpu.py

# Create a SLURM job submission script template
cat > run_gpu_job.sh << 'EOL'
#!/bin/bash
#SBATCH --job-name=hallucination
#SBATCH --output=hallucination_%j.out
#SBATCH --error=hallucination_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=gpu

# Load modules (adjust for your cluster)
module purge
module load cuda11.8/toolkit/11.8.0

# Set up environment
source ~/.bashrc
conda activate hallucination

# Set environment variables
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export CUDA_HOME=$CONDA_PREFIX

# Print job info
echo "Running on $(hostname)"
echo "Current working directory: $(pwd)"
echo "Starting run at: $(date)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# Run the application
python "$@"

echo "Job finished at: $(date)"
EOL

chmod +x run_gpu_job.sh

# Run GPU tests to verify installation
echo "Testing GPU setup - this may take a moment..."
python test_gpu.py

echo "Setup complete! Your GPU environment is ready."
echo "To submit a job, use: sbatch run_gpu_job.sh your_script.py"
echo "To run GPU tests directly: python test_gpu.py"