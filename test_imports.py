#!/usr/bin/env python
# Test imports for hallucination detection project
import sys
import os
import time

def test_imports():
    print("Testing imports for hallucination detection project...")
    print(f"Python version: {sys.version}")
    
    # Basic data science packages
    print("\n--- Testing basic data science packages ---")
    packages = [
        "numpy", "scipy", "pandas", "matplotlib", "seaborn", 
        "sklearn"
    ]
    
    for package in packages:
        try:
            start = time.time()
            exec(f"import {package}")
            end = time.time()
            print(f"✅ {package} imported successfully ({(end-start):.2f}s)")
        except ImportError as e:
            print(f"❌ Failed to import {package}: {str(e)}")
    
    # Deep learning frameworks
    print("\n--- Testing deep learning frameworks ---")
    try:
        start = time.time()
        import torch
        end = time.time()
        print(f"✅ PyTorch {torch.__version__} imported successfully ({(end-start):.2f}s)")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   CUDA version: {torch.version.cuda}")
            print(f"   GPU device: {torch.cuda.get_device_name(0)}")
    except ImportError as e:
        print(f"❌ Failed to import torch: {str(e)}")
    
    # Hugging Face libraries
    print("\n--- Testing Hugging Face libraries ---")
    hf_packages = [
        "transformers", "accelerate", "evaluate"
    ]
    
    for package in hf_packages:
        try:
            start = time.time()
            exec(f"import {package}")
            end = time.time()
            print(f"✅ {package} imported successfully ({(end-start):.2f}s)")
        except ImportError as e:
            print(f"❌ Failed to import {package}: {str(e)}")
    
    # NLP tools
    print("\n--- Testing NLP tools ---")
    nlp_packages = [
        "spacy", "sentencepiece", "rouge_score", "selfcheckgpt"
    ]
    
    for package in nlp_packages:
        try:
            start = time.time()
            exec(f"import {package}")
            end = time.time()
            print(f"✅ {package} imported successfully ({(end-start):.2f}s)")
        except ImportError as e:
            print(f"❌ Failed to import {package}: {str(e)}")
    
    # Check spacy model
    try:
        import spacy
        start = time.time()
        nlp = spacy.load("en_core_web_sm")
        end = time.time()
        print(f"✅ spacy model 'en_core_web_sm' loaded successfully ({(end-start):.2f}s)")
    except Exception as e:
        print(f"❌ Failed to load spacy model: {str(e)}")
    
    # Additional utilities
    print("\n--- Testing additional utilities ---")
    util_packages = [
        "einops", "gputil", "captum"
    ]
    
    for package in util_packages:
        try:
            start = time.time()
            exec(f"import {package}")
            end = time.time()
            print(f"✅ {package} imported successfully ({(end-start):.2f}s)")
        except ImportError as e:
            print(f"❌ Failed to import {package}: {str(e)}")
    
    # Project-specific modules
    print("\n--- Testing project-specific modules ---")
    try:
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        
        start = time.time()
        from utils import classifier, data_utils, model_utils
        end = time.time()
        print(f"✅ Project utils modules imported successfully ({(end-start):.2f}s)")
        
        start = time.time()
        from models import load_model, classifer_softmax, ig
        end = time.time()
        print(f"✅ Project models modules imported successfully ({(end-start):.2f}s)")
        
    except ImportError as e:
        print(f"❌ Failed to import project modules: {str(e)}")
    
    print("\nImport test complete!")

if __name__ == "__main__":
    test_imports()