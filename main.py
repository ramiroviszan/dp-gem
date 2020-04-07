import keras.backend as K
import tensorflow as tf
import os
import copy
from os import listdir


def discover_experiments():
    files_in_folder = [(name.split(".py")[0], "study_cases." + name.split(".py")[0]) for name in listdir("study_cases") if name.endswith(".py") and name != '__init__.py']
    experiments = {}
    for name, exp in files_in_folder:
        module = __import__(exp, globals(), locals(), ['experiment'], 0)
        experiments[name] = getattr(module, 'experiment')
    print(experiments)
    return experiments

def main():

    experiments = discover_experiments()

    for key in experiments:
        print('\n\nExperiment:', key)
        exp = experiments[key]
        exp_path = create_exp_folder(key)

        run_interations = exp['run_iterations']

        print('\n\nControl Tests')
        for i in range(0, run_interations['control_test']):
            print('\nControl Test Iteration:', i)
            params = copy.deepcopy(exp['control_test'])
            control_test = hot_new(exp_path, i, *params.values())
            control_test.run_test()

        print('\n\nGenerator')
        dp_gen_params = copy.deepcopy(exp['dp_gen'])
        utility_test_params = copy.deepcopy(dp_gen_params['utility_test'])
        dp_gen_params.pop('utility_test', None)
        dp_gen = hot_new(exp_path, 0, *dp_gen_params.values())
        for i in range(0, run_interations['dp_gen']):
            print('\nGenerator Test Iteration:', i)
            dp_gen.generate(i)
            utility_params = copy.deepcopy(utility_test_params)
            utility_test = hot_new(exp_path, i, *utility_params.values())
            utility_test.run_test()


def hot_new(experiment, iteration, module_name, class_name, params):
    module = __import__(module_name, globals(), locals(), [class_name], 0)
    class_ = getattr(module, class_name)
    instance = class_(experiment, iteration, *params.values())
    return instance


def create_exp_folder(key):
    exp_path = "experiments/" + str(key)
    try:
        os.mkdir("experiments")
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
