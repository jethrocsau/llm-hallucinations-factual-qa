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
        "<<SYS>> Your answer was wrong. You answered started with $prev_answer. Retry answering the question.<</SYS>>\nQ: $question\nA: "
    ),
    'inst-wrong': Template(
        "<<INST>> Your answer was wrong. You answered started with $prev_answer. Retry answering the question.<</INST>>\nQ: $question\nA: "
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
    return dataset

# format prompt
def format_prompt(original_prompt, question,  answer, template_name, num_answer_string = 20):
    template = prompts[template_name]
    if num_answer_string > 0:
        gen_len = min(len(answer), num_answer_string)
        answer = answer[:gen_len]
    prompt = original_prompt + '\n' + template.substitute(
        question=question,
        prev_answer=answer
    )
    return prompt

