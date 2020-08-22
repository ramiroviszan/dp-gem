import sys
import os
import time
import copy
import random
from os import listdir
import numpy as np
import tensorflow as tf

from runner.base_runner import BaseRunner


def main(argv):
    if len(argv) == 0:
        sure = input("No exp passed, run all !skipped experiments (Y/N)")
        if sure != "Y":
            print("Bye")
            return

    experiments = discover_experiments(argv)
    for exp_name, exp in experiments.items():
        skip = exp.get('skip', False)
        if skip:
            print('Skipping experiment:', exp_name)
        else:
            print('\n\n\n\nExperiment:', exp_name)
            exp_path = create_exp_folder(exp_name)

            for module_name, module_desc in exp.items():
                if type(module_desc) == dict: 
                    print('\n\n\n\n', module_name)
                    set_random(exp)
                    runner = BaseRunner(module_name, exp_name, exp_path)
                    runner.run_module(module_desc)

def discover_experiments(filtered):
    if len(filtered) == 0:
        files_in_folder = [(name.split('.py')[0], 'experiments.' + name.split('.py')[0]) for name in listdir('experiments') if name.endswith('.py') and name != '__init__.py']
    else:
        files_in_folder = [(name.split('.py')[0], 'experiments.' + name.split('.py')[0]) for name in listdir('experiments') if name[:-3] in filtered and name.endswith('.py') and name != '__init__.py']

    experiments = {}
    for name, exp in files_in_folder:
        module = __import__(exp, globals(), locals(), ['experiment'], 0)
        experiments[name] = getattr(module, 'experiment')
    return experiments

def create_exp_folder(key):
    exp_path = 'results/' + str(key)
    try:
        os.mkdir('results')
    except OSError:
        print('Directory creation failed: /results')
    else:
        print('Successfully created directory: /results')

    try:
        os.mkdir(exp_path)
    except OSError:
        print('Directory creation failed:', exp_path)
    else:
        print('Successfully created directory:', exp_path)

    return exp_path

def set_random(exp):
    random_seed = exp.get('random_seed', int(time.time()))
    print('\n\n\n\nSetting random seed to:', random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

if __name__ == '__main__':
    physical_devices = tf.config.list_physical_devices('GPU') 
    for d in physical_devices:
        tf.config.experimental.set_memory_growth(d, True)
    main(sys.argv[1:])
