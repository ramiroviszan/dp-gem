import keras.backend as K
import tensorflow as tf
import os
import copy


def main():

    experiments = {
        'exp_1': {
            'random_seed': 1,
            'run_iterations': {
                'control_test': 1,
                'dp_gen': 2,
            },
            'control_test': {
                'module_name': 'study_cases.deeplog.deeplog_lm',
                'class_name': 'DeepLogLMClassifier',
                'params': {
                    'datasets_params': {
                        'train': {
                            'fullpath': 'data/normal_train.txt',
                            'to_read': -1
                        },
                        'test_normal': {
                            'fullpath': 'data/normal.txt',
                            'to_read': 1000
                        },
                        'test_abnormal': {
                            'fullpath': 'data/abnormal.txt',
                            'to_read': 1000
                        }
                    },
                    'model_fullpath': '{exp_name}/deeplog.h5',
                    'train_params': {
                        'window_size': 10,
                        'vocab_size': 29,
                        'train_sessions': {
                            'first': {
                                'epochs': 1,
                                'batch_size': 30,
                                'lr': 0.001
                            }
                        }
                    },
                    'classifier_params': {
                        'use_top_k': 0,
                        'thresholds': [0.00005, 0.0001],
                        'recalulate_probas': False,
                        'probas_fullpath': '{exp_name}/control_probas_topk_{topk}.npy',
                        'results_fullpath': '{exp_name}/control_results.csv',
                        'plots_fullpath': '{exp_name}/plot_{{uuid}}.png'
                    }
                }
            },
            'dp_gen': {
                'module_name': 'study_cases.deeplog.deeplog_dp_gen_emb',
                'class_name': 'DeepLogDPGen',
                'params': {
                    'datasets_params': {
                        'train': [('data/normal.txt', 4000), ('data/abnormal.txt', 4000)],
                        'to_privatize': [('normal', 'data/normal.txt', 4000), ('abnormal', 'data/abnormal.txt', 4000)]
                    },
                    'model_fullpath': '{exp_name}/deeplog_dp_gen_emb.h5',
                    'train_params': {
                        'context_size': 4,
                        'emb_size': 16,
                        'vocab_size': 29,
                        'train_sessions': {
                            'first': {
                                'epochs': 10,
                                'batch_size': 500,
                                'lr': 0.0001
                            },
                            'second': {
                                'epochs': 10,
                                'batch_size': 100,
                                'lr': 0.0001
                            }
                        }
                    },
                    'pre_proba_matrix_fullpath': '{exp_name}/pre_proba_matrix.npy',
                    'generation_params':{
                        'eps': 3,
                        'delta': 0
                    },
                    'to_privatize_output_fullpath': '{exp_name}/fake_data_{{to_privatize_name}}_{{iteration}}.txt'
                },
                'utility_test': {
                    'module_name': 'deeplog',
                    'class_name': 'DeepLogLMClassifier',
                    'params': {
                        'datasets_params': {
                            'train': {
                                'fullpath': 'data/fake_train_{iteration}.txt',
                                'to_read': -1
                            },
                            'test_normal': {
                                'fullpath': 'data/fake_normal_{iteration}.txt',
                                'to_read': 1000
                            },
                            'test_abnormal': {
                                'fullpath': 'data/fake_abnormal_{iteration}.txt',
                                'to_read': 1000
                            }
                        },
                        'model_fullpath': '{exp_name}/deeplog_{iteration}.h5',
                        'train_params': {
                            'window_size': 10,
                            'vocab_size': 29,
                            'epochs': 1,
                            'batch_size': 30,
                            'lr': 0.001
                        },
                        'classifier_params': {
                            'use_top_k': 0,
                            'thresholds': [0.00005, 0.0001],
                            'recalulate_probas': False,
                            'probas_fullpath': '{exp_name}/control_probas_topk_{topk}.npy',
                            'results_fullpath': '{exp_name}/control_results.csv',
                            'plots_fullpath': '{exp_name}/plot_{{uuid}}.png'
                        }
                    }
                }
            }
        }
    }

    for key in experiments.keys():
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
        utility_test_params = dp_gen_params['utility_test']
        dp_gen_params.pop('utility_test', None)
        dp_gen = hot_new(exp_path, 0, *dp_gen_params.values())
        for i in range(0, run_interations['dp_gen']):
            print('\nGenerator Test Iteration:', i)
            dp_gen.generate(i)
            #utility_test = hot_new(exp_path, i, *utility_test_params.values())
            #utility_test.run_test()


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
