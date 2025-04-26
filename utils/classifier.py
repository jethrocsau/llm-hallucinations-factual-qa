import random
import os

# Global vars
MODELS = {

}

# get the current directory
utils_path = os.path.dirname(os.path.abspath(__file__))


def run_classifier(results, val_fix = -1, debug = True):
    if debug == True and val_fix == -1:
        return random.randint(0, 1)
    elif val_fix ==1:
        return 1
    elif val_fix == 0:
        return 0


def load_clasifier(model_name):
    model_path = 
    return model