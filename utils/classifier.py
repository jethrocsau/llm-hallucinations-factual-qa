import random
import os
import numpy as np
from models.load_model import load
from models.classifer_softmax import SoftMaxClassifier
from models.ig import RNNHallucinationClassifier
import torch

# get the current directory
utils_path = os.path.dirname(os.path.abspath(__file__))
cwd = os.path.dirname(utils_path)
model_dir = os.path.join(cwd, 'models')
trained_dir = os.path.join(model_dir, 'trained')

# device
gpu = "0"
device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")

def activations_MLP_predict(model, results):
    # load results_activations
    result_attention_np = np.stack([x[-1] for x in results['first_attention']])
    return model.predict(result_activation_np)

def softmax_predict(model, results):
    # load results_activations
    first_logits = np.stack([sp.special.softmax(i[j]) for i, j in zip(results['logits'], results['start_pos'])])
    return model(first_logits)

def RNN_predict(model, results):
    # load results_activations
    x = results['attributes_first']
    return model(torch.tensor(x).view(1, -1, 1).to(torch.float).to(device))

def attention_MLP_predict(model, results):
    # load results_activations
    result_attention_np = np.stack([x[-1] for x in results['first_attention']])
    return model.predict(result_attention_np)

# models
MODELS = {
    'activation_MLP': activations_MLP_predict,
    'attention_MLP': attention_MLP_predict,
    'RNN': RNN_predict,
    'softmax': softmax_predict
}

def run_classifier(model, results, model_name, val_fix = -1, rand = True):
    if rand == True and val_fix == -1:
        return random.randint(0, 1)
    elif rand == False and val_fix == -1:
        return MODELS[model_name](model, results)
    elif val_fix ==1:
        return 1
    elif val_fix == 0:
        return 0

def aggregate_pred(preds, agg):
    if agg == 'mean':
        return np.mean(preds)
    elif agg == 'max':
        return np.max(preds)
    elif agg == 'min':
        return np.min(preds)
    elif agg == 'median':
        return np.median(preds)
    elif agg == 'sum':
        return np.sum(preds)
    else:
        raise ValueError(f"Unknown aggregation method: {agg}")
