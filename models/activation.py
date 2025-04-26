import numpy as np
import torch
from keras.models import load_model



def load_model(model_path):
    """
    Load a Keras model from the specified path.
    """
    return load_model(model_path)
