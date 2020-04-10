import os
import time
import copy
import random
import numpy as np
import tensorflow as tf
import keras.backend as K

from os import listdir

def discover_experiments():
    files_in_folder = [(name.split(".py")[0], "study_cases." + name.split(".py")[0]) for name in listdir("study_cases") if name.endswith(".py") and name != '__init__.py']
    experiments = {}
    for name, exp in files_in_folder:
        module = __import__(exp, globals(), locals(), ['experiment'], 0)
        experiments[name] = getattr(module, 'experiment')
    return experiments

def main():

    experiments = discover_experiments()

    for key in experiments:
        print('\n\nExperiment:', key)
        exp = experiments[key]
        exp_path = create_exp_folder(key)

        if 'random_seed' in exp:
            seed = exp['random_seed']
            print('\n\nSetting randoms to:', seed)
            np.random.seed(seed)
            random.seed(seed)
        else:
            seed = int(time.time())
            np.random.seed(seed)
            random.seed(seed)
            print('Warning, no random_seed in ' + key + '. Executing with: ' + seed)


        print('\n\nData Preparation')
        if 'data_preparation' in exp:
            data_preparation = exp['data_preparation']
            data_skip = data_preparation.get('skip', True) #By default will skip
            if data_skip:
                print('\nSkipping Data Preparation')
            else:
                preparation_params = list(data_preparation.values())[1:] #[1:] removes skip param
                preparation = hot_new(exp_path, *preparation_params)
                preparation.run()

        print('\n\nControl Tests')
        if 'control_test' in exp:
            control_test = exp['control_test']
            control_iters = control_test.get('run_iterations', 0)
            if control_iters == 0:
                print('\nSkipping Control Test')
            else:
                for i in range(0, control_iters):
                    print('\nControl Test Iteration:', i)
                    control_test_copy = copy.deepcopy(control_test)
                    control_params = list(control_test_copy.values())[1:] #[1:] removes run_iterations param
                    control = hot_new((exp_path, 0, i), *control_params)
                    control.run_test()
        else:
            print('\nWarning, no control_test found in', key, '- Skipping.')

        print('\n\nGenerator')
        if 'dp_gen' in exp:  
            dp_gen = exp['dp_gen']
            gen_iters = dp_gen.get('run_iterations', 0)
            epsilon_tries = dp_gen.get('epsilon_tries', [])
            print('Epsilon tries:', epsilon_tries)
            if len(epsilon_tries) == 0:
                print('\nWarning, no epsilon_tries found in', key, '- Skipping Generator and Utility tests.')
            else:
                utility_test = dp_gen['utility_test']   
                utility_skip = utility_test.get('skip', False) #By default won't skip
                if gen_iters == 0:
                    print('\nSkipping Generation')
                elif gen_iters > 0:
                    gen_params = list(dp_gen.values())[2:-1]#1:-1 remove run_iterations, epsilons and utility_tests from gen_params
                    gen = hot_new(exp_path, *gen_params)
                    for epsilon in epsilon_tries:
                        for i in range(0, gen_iters):
                            print('\nGenerator Test eps=', epsilon, 'iter=', i)
                            gen.generate(epsilon, i)
                            if utility_skip:
                                print('\nSkipping Utility Test', i)
                            else:
                                print('\nUtility Test eps=', epsilon, 'iter=', i)
                                utility_test_copy = copy.deepcopy(utility_test)
                                utility_params = list(utility_test_copy.values())[1:] #[1:] removes skip param
                                utility = hot_new((exp_path, epsilon, i), *utility_params)
                                utility.run_test()
                elif gen_iters < 0:
                    print('\nSkipping Generation. Running Utility Test only:', abs(gen_iters), 'iterations with previously generated files! Hope they exist :)')
                    for epsilon in epsilon_tries:
                        for i in range(0, abs(gen_iters)):
                            print('\nUtility Test eps=', epsilon, 'iter=', i)
                            utility_test_copy = copy.deepcopy(utility_test)
                            utility_params = list(utility_test_copy.values())[1:] #[1:] removes skip param
                            utility = hot_new((exp_path, epsilon, i), *utility_params)
                            utility.run_test()
        else:
            print('\nWarning, no dp_gen found in', key, '- Skipping.')

def hot_new(experiment_info, module_name, class_name, params):
    module = __import__(module_name, globals(), locals(), [class_name], 0)
    class_ = getattr(module, class_name)
    instance = class_(experiment_info, *params.values())
    return instance

def create_exp_folder(key):
    exp_path = "results/" + str(key)
    try:
        os.mkdir("results")
    except OSError:
        print("Creation of the directory 'results' failed")
    else:
        print("Successfully created the directory 'results'")

    try:
        os.mkdir(exp_path)
    except OSError:
        print("Creation of the directory %s failed" % exp_path)
    else:
        print("Successfully created the directory %s " % exp_path)

    return exp_path


if __name__ == '__main__':

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)
    main()
