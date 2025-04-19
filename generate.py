from datasets import load_dataset

from utils.classifier import run_classifier
from utils.data_utils import format_prompt, load_data, load_prompts
from utils.model_utils import generate_attributes, load_model

# global variables
debug = True

# generate N-turns
def generate_multiturn_attributes(model, tokenizer, embedder, start_template, question, aliases, n,p_thresh):
    if debug: print("Generating N-turns...")
    n_gen_result = []
    prompt = start_template.substitute(question=question)
    for i in range(n):
        # generate attributes
        results['turn'] = i
        results = generate_attributes(model, tokenizer, embedder, prompt, aliases)

        # call classifier inference
        template = 'sys-wrong'
        prob = run_classifier(results)
        if prob > p_thresh:
            results['hallucination'] = True
            format_prompt(results['question'], question,  results['response'], template , num_answer_string = 20)
            n_gen_result.append(results)
        else:
            results['hallucination'] = False
            n_gen_result.append(results)
            break
    if debug:
        for i in range(len(n_gen_result)):
            print(f"Turn {i}:")
            print(f"Question: {n_gen_result[i]['question']}")
            print(f"Response: {n_gen_result[i]['response']}")
            print(f"Hallucination: {n_gen_result[i]['hallucination']}")
            print(f"Correct: {n_gen_result[i]['correct']}")

    return n_gen_result

# load model
model, tokenizer, embedder = load_model()

# attribute N-generations
def __main__():
    #load triviaQA
    if debug: print("Loading data...")
    dataset = load_data()

    #load prompts
    if debug: print("Loading prompts...")
    prompts = load_prompts()
    default_p = template['default']

    #generate multiturn
    if debug: print("Generating multiturn attributes...")
    multi_turn = []
    n = 5
    p_thresh = 0.5
    for i, (question, aliases) in enumerate(dataset):
        if debug and i > 10:
            break
        turn_data = generate_multiturn_attributes(model, tokenizer, embedder, default_p,question, aliases, n,p_thresh)
        multi_turn.append(turn_data)

    # debug
    if debug:
        for i in range(len(multi_turn)):
            print(f"Question: {multi_turn[i][0]['question']}")
            print(f"Final Response: {multi_turn[i][-1]['response']}")
            print(f"Hallucination: {multi_turn[i][-1]['hallucination']}")
            print(f"Correct: {multi_turn[i][-1]['correct']}")
