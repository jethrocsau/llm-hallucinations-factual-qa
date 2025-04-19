import random


def run_classifier(results, val_fix = -1, debug = True):
    if debug == True and val_fix == -1:
        return random.randint(0, 1)
    elif val_fix ==1:
        return 1
    elif val_fix == 0:
        return 0

