import gc
import os
import pickle
from datetime import datetime

import numpy as np
import torch
from datasets import load_dataset

from utils.classifier import run_classifier
from utils.data_utils import (format_prompt, format_result, load_data,
                              load_prompts)
from utils.model_utils import generate_attributes, load_model

#Flags
debug = False
data_mining = True

# global variables
cwd = os.path.dirname(__file__)
results_dir = os.path.join(cwd, 'result')
# create directory if not exists
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
template = load_prompts()

# generate N-turns
def generate_multiturn_attributes(model, tokenizer, embedder, start_template, question, aliases, n,p_thresh):
    if debug: print("Generating N-turns...")
    n_gen_result = []
    prompt = start_template.substitute(question=question)
    for i in range(n):
        # generate attributes
        with torch.no_grad():
            results = generate_attributes(model, tokenizer, embedder, prompt, aliases)
        results['turn'] = i

        # call classifier inference
        template_name = 'sys-wrong'
        prob = run_classifier(results,val_fix = 1,debug = debug)
        if prob > p_thresh:
            results['hallucination'] = True
            prompt = format_prompt(results['question'][0], question,  results['str_response'][0], template_name , num_answer_str = 10)
            if data_mining:
                n_gen_result.append(format_result(results, save_all=True))
            else:
                n_gen_result.append(format_result(results, save_all=False))
        else:
            results['hallucination'] = False
            if data_mining:
                n_gen_result.append(format_result(results, save_all=True))
            else:
                n_gen_result.append(format_result(results, save_all=False))
            break

        del results
        gc.collect()

    if debug:
        for i in range(len(n_gen_result)):
            print(f"Turn {i}:")
            print(f"Question: {n_gen_result[i]['question']}")
            print(f"Response: {n_gen_result[i]['str_response']}")
            print(f"Hallucination: {n_gen_result[i]['hallucination']}")
            print(f"Correct: {n_gen_result[i]['correct']}")
    return n_gen_result


# attribute N-generations
if __name__ == "__main__":

    #load configurations
    start = 0
    end = 2

    #load configurations
    if debug: print("Loading configurations...")
    dataset = load_data()[start:end]
    model, tokenizer, embedder = load_model()
    template_name = 'default'
    default_p = template[template_name]

    #generate multiturn
    if debug: print("Generating multiturn attributes...")
    multi_turn = []
    n = 3
    p_thresh = 0.5
    for i, (question, aliases) in enumerate(dataset):
        if debug and i > 10:
            break
        turn_data = generate_multiturn_attributes(model, tokenizer, embedder, default_p,question, aliases, n,p_thresh)
        multi_turn.append(turn_data)

        del turn_data
        gc.collect()

    # debug
    if debug:
        for i in range(len(multi_turn)):
            print(f"Question: {multi_turn[i][0]['question']}")
            print(f"Final Response: {multi_turn[i][-1]['str_response']}")
            print(f"Hallucination: {multi_turn[i][-1]['hallucination']}")
            print(f"Correct: {multi_turn[i][-1]['correct']}")

    # save the results
    if debug: print("Saving results...")
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d_%H-%M-%S")
    save_path = os.path.join(results_dir, f"multiturn_{template_name}_{dt_string}_{start}-{end}.pkl")

    #create file if not exists
    with open(save_path, 'wb') as f:
        pickle.dump(multi_turn, f)
