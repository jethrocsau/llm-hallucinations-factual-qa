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
results = []
target_index = '500-1500'

def get_results(results_dir, target_index):
    """
    Get results from the result directory
    """
    results = []

    for file in os.listdir(results_dir):
        if file.endswith('.pkl') and target_index in file:
            file_path = os.path.join(results_dir, file)
            data = read_pickle(file_path)
            results.append(data)

    dataset = []
    for q in data:
        question = q[0]['question']
        for dict in q:
            row_val = []
            row_val.append(question)
            if dict is not None:
                for key in dict:
                        if key == 'correct':
                            row_val.append(dict[key][0])       
                        else:
                            row_val.append(dict[key])
                dataset.append(row_val)

    columns = data[0][0].keys()
    columns_list = ['trivia_qa']
    columns_list.extend(list(columns))
    df = pd.DataFrame(dataset,columns = columns_list)
    return df


if __name__ == "__main__":


    # this is still in dev

    # get dir
    dir = os.path.dirname(__file__)
    results_dir = os.path.join(dir, 'result')
    results = []
    target_index = '500-1500'

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
