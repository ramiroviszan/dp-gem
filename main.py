import sys
import os
import time
import copy
import random
import numpy as np
import tensorflow as tf


from os import listdir


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
            set_random(exp)

            print('\n\n\n\nData Preparation')
            data_preparation_desc = exp.get('data_preparation', None)
            run_data_preparation(data_preparation_desc, exp_name, exp_path)

            print('\n\n\n\nControl Tests')
            control_test_desc = exp.get('control_test', None)
            run_control_tests(control_test_desc, exp_name, exp_path)

            print('\n\n\n\nGenerator')
            dp_gen_desc = exp.get('dp_gen', None)
            run_generator(dp_gen_desc, exp_name, exp_path)

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

def run_data_preparation(data_preparation_desc, exp_name, exp_path):
    if data_preparation_desc != None:
        data_skip = data_preparation_desc.get('skip', True) #By default will skip
        if data_skip:
            print('\n\nSkipping Data Preparation')
        else:
            preparation_params = list(data_preparation_desc.values())[1:] #[1:] removes skip param
            preparation = hot_new(exp_path, *preparation_params)
            preparation.run()
    else:
        print('\n\nWarning, no data_preparation found in', exp_name, '- Skipping.')

def run_control_tests(control_test_desc, exp_name, exp_path):
    if control_test_desc != None:
        control_iters = control_test_desc.get('run_iterations', 0)
        if control_iters == 0:
            print('\n\nSkipping Control Test')
        else:
            for i in range(0, control_iters):
                print('\n\nControl Test Iteration:', i)
                control_test_copy = copy.deepcopy(control_test_desc)
                control_params = list(control_test_copy.values())[1:] #[1:] removes run_iterations param
                control = hot_new((exp_path, {}, i), *control_params)
                control.run_test()
    else:
        print('\n\nWarning, no control_test found in', exp_name, '- Skipping.')

def run_generator(dp_gen_desc, exp_name, exp_path):
    if dp_gen_desc != None:
        trials = dp_gen_desc.get('trials', [])
        print('\n\nTrials:', trials)
        gen_iters = dp_gen_desc.get('run_iterations', 0)
        print('\nRun Iterations:', gen_iters)

        mode = dp_gen_desc.get('mode', None)
        if mode == 'all' or mode == 'tests_only':
            utility_tests_desc = dp_gen_desc.get('utility_tests', None)

        if len(trials) == 0:
            print('\n\nWarning: No trials found in', exp_name, '- Skipping.')
        elif gen_iters == 0:
            print('\n\nWarning: Skipping Generation, run_iterations is 0 or missing key in', exp_name, '-Skipping.')
        elif mode == 'all':
            print('\n\nGenerator and tests.')
            gen_params = list(dp_gen_desc.values())[3:-1]#1:-1 remove run_iterations, trials, mode and utility_tests from dp_gen_desc
            gen = hot_new(exp_path, *gen_params)
            for trial in trials:
                for i in range(0, gen_iters):
                    print('\n\nGenerator with trial =', trial, 'iter =', i, 'total iters =', gen_iters)
                    gen.generate(trial, i)
                    run_tests(utility_tests_desc, trial, i, exp_name, exp_path)
        elif mode == 'gen_only':
            print('\n\nGenerator only: skipping tests.')
            gen_params = list(dp_gen_desc.values())[3:-1]#1:-1 remove run_iterations, trials, mode and utility_tests from dp_gen_desc
            gen = hot_new(exp_path, *gen_params)
            for trial in trials:
                for i in range(0, gen_iters):
                    print('\n\nGenerator with trial =', trial, 'iter =', i, 'total iters =', gen_iters)
                    gen.generate(trial, i)
        elif mode == 'tests_only':
            print('\n\nTests only: running iterations with previously generated files! Hope they exist :). Skipping Generator.')
            for trial in trials:
                for i in range(0, gen_iters):
                    run_tests(utility_tests_desc, trial, i, exp_name, exp_path)
        else:
            print('\n\nWarning: No (mode = all, gen_only, tests_only) found in:', exp_name, '- Skipping.')
    else:
        print('\n\nWarning: No dp_gen_desc found in', exp_name, '- Skipping.')
            
def run_tests(tests_desc, trial, i, exp_name, exp_path):
    if tests_desc != None:
        tests_desc = copy.deepcopy(tests_desc)
        for test_name, test_desc in tests_desc.items():
            skip = test_desc.get('skip', False) #By default won't skip
            if skip:
                print('\n\nSkipping Test:', test_name, 'with trial =', trial, 'iter =', i)
            else:
                print('\n\nTest:', test_name, 'with trial =', trial, 'iter =', i)
                test_params = list(test_desc.values())[1:] #[1:] removes skip param
                test = hot_new((exp_path, trial, i), *test_params)
                test.run_test()
    else:
        print('Warning: no utility_tests found in', exp_name, 'but still trying to run in mode= all or tests_only.')

def hot_new(experiment_info, module_name, class_name, params):
    module = __import__(module_name, globals(), locals(), [class_name], 0)
    class_ = getattr(module, class_name)
    instance = class_(experiment_info, *params.values())
    return instance


if __name__ == '__main__':
    physical_devices = tf.config.list_physical_devices('GPU') 
    for d in physical_devices:
        tf.config.experimental.set_memory_growth(d, True)
    main(sys.argv[1:])
