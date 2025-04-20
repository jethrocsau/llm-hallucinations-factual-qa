import os
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from string import Template
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from tqdm import tqdm

# Global variable
debug = True

# Prompt templates
prompts = {
    'default': Template(
        "Q: $question\nA: "
    ),
    'sys-wrong': Template(
        "<<SYS>> Your answer was wrong. Retry answering the question.<</SYS>>\nQ: $question\nA: "
    ),
    'inst-wrong': Template(
        "<<INST>> Your answer was wrong. Retry answering the question.<</INST>>\nQ: $question\nA: "
    )
}

def load_prompts():
    return prompts

#Load Datasets
def load_data(start=0, end=-1):
    cwd = os.path.dirname(__file__)
    data_dir = os.path.join(os.path.dirname(cwd), 'data')
    trivia_dataset = load_dataset('trivia_qa', data_dir='rc.nocontext', cache_dir=data_dir)
    full_dataset = []
    for obs in tqdm(trivia_dataset['train']):
        aliases = []
        aliases.extend(obs['answer']['aliases'])
        aliases.extend(obs['answer']['normalized_aliases'])
        aliases.append(obs['answer']['value'])
        aliases.append(obs['answer']['normalized_value'])
        full_dataset.append((obs['question'], aliases))
    dataset = full_dataset[start: end]

    del trivia_dataset
    return dataset

# format prompt
def format_prompt(original_prompt: str, question:str,  answer:str, template_name:str, num_answer_str = 20):
    template = prompts[template_name]
    if num_answer_str > 0:
        gen_len = min(len(answer), num_answer_str)
        answer = answer[:gen_len]
    prompt = original_prompt + answer + '\n' + template.substitute(
        question=question
    )
    return prompt


# format results generated from generate attributes
def format_result(result,save_all = False):

    if save_all:
        result['question'] = result['question'][0]
        result['str_response'] = result['str_response'][0]
        return result
    else:
        save_result = {}
        save_result['question'] = result['question'][0]
        save_result['str_response'] = result['str_response'][0]
        save_result['hallucination'] = result['hallucination']
        save_result['correct'] = result['correct']
        save_result['turn'] = result['turn']
        return save_result