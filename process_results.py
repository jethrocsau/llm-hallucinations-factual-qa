#
import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import seaborn as sns
from sklearn.decomposition import PCA
from tqdm import tqdm

from utils.data_utils import read_pickle

dir = os.getcwd()
results_dir = os.path.join(dir, 'result')
results = {}
dataset = []
target_index = '1000-1500'

#import all results in folder
for file in os.listdir(results_dir):
    if file.endswith('.pkl') and target_index in file:
        file_path = os.path.join(results_dir, file)
        data = read_pickle(file_path)
        results.append(data)

# Flatten the list of results
dataset = [item for sublist in results for item in sublist]

# process results
for i in range(len(dataset)):
    # remove the first element of each result
    for j in range(len(dataset[i])):
        results[i] = dataset[i][j]
        try:
            results_data = {}
            num_layers = results['first_fully_connected'][0].shape[0]
            layer_pos = num_layers-2
            first_attribute_entropy = np.array([sp.stats.entropy(i) for i in results['attributes_first']])
            first_logits_entropy = np.array([sp.stats.entropy(sp.special.softmax(i[j])) for i,j in zip(results['logits'], results['start_pos'])])
            last_logits_entropy = np.array([sp.stats.entropy(sp.special.softmax(i[-1])) for i in results['logits']])
            first_logit_decomp = PCA(n_components=2).fit_transform(np.array([i[j] for i,j in zip(results['logits'], results['start_pos'])]))
            last_logit_decomp = PCA(n_components=2).fit_transform(np.array([i[-1] for i in results['logits']]))
            first_token_layer_activations = np.array([i[layer_pos] for i in results['first_fully_connected']])
            final_token_layer_activations = np.array([i[layer_pos] for i in results['final_fully_connected']])
            first_token_layer_attention = np.array([i[layer_pos] for i in results['first_attention']])
            final_token_layer_attention = np.array([i[layer_pos] for i in results['final_attention']])
            correct = np.array(results['correct'])
            results_data['rephrase'] = {'first_attribute_entropy': first_attribute_entropy,
                                                    'correct': correct,
                                                    'first_logits_entropy': first_logits_entropy,
                                                    'last_logits_entropy': last_logits_entropy,
                                                    'first_logit_decomp': first_logit_decomp,
                                                    'last_logit_decomp': last_logit_decomp,
                                                    'first_token_layer_activations': first_token_layer_activations,
                                                    'final_token_layer_activations': final_token_layer_activations,
                                                    'first_token_layer_attention': first_token_layer_attention,
                                                    'final_token_layer_attention': final_token_layer_attention,}
            del results
        except:
            print("Error processing results")
