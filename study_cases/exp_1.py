experiment = {
    'random_seed': 27,
    'data_preparation': {
        'skip': False,
        'module_name': 'study_cases.deeplog.data_splitter',
        'class_name': 'DataSplitter',
        'params': {
            'datasets': {
                'normal': {
                    'original': 'data/deeplog/all_normal.txt',
                    'to_read': 4000,
                    'train_output_fullpath': '{exp_name}/normal_train.txt',
                    'val_output_fullpath': '{exp_name}/normal_val.txt',
                    'test_output_fullpath': '{exp_name}/normal_test.txt',
                    'splits': {
                        'train_test': 0.3,
                        'train_val': 0.3
                    }
                },
                'abnormal': {
                    'original': 'data/deeplog/all_abnormal.txt',
                    'to_read': 4000,
                    'train_output_fullpath': '{exp_name}/abnormal_train.txt',
                    'val_output_fullpath': '{exp_name}/abnormal_val.txt',
                    'test_output_fullpath': '{exp_name}/abnormal_test.txt',
                    'splits': {
                        'train_test': 0.3,
                        'train_val': 0.3
                    }
                }
            }
        }
    },
    'control_test': {
        'run_iterations': 1,
        'module_name': 'study_cases.deeplog.deeplog_lm',
        'class_name': 'DeepLogLMClassifier',
        'params': {
            'datasets_params': {
                'train': {
                    'normal': {
                        'fullpath': '{exp_name}/normal_train.txt',
                        'to_read': -1
                    }
                },   
                'val': {
                    'normal': {
                        'fullpath': '{exp_name}/normal_val.txt',
                        'to_read': -1,
                        'class': 1
                    },
                    'abnormal': {
                        'fullpath': '{exp_name}/abnormal_val.txt',
                        'to_read': -1,
                        'class': 0
                    }
                },
                'test': {
                    'normal': {
                        'fullpath': '{exp_name}/normal_test.txt',
                        'to_read': -1,
                        'class': 1
                    },
                    'abnormal': {
                        'fullpath': '{exp_name}/abnormal_test.txt',
                        'to_read': -1,
                        'class': 0
                    }
                }
            },
            'network_fullpath': '{exp_name}/deeplog_control.h5',
            'network_params': {
                'model_type': 'control',
                'window_size': 10,
                'vocab_size': 29,
                'train_sessions': {
                    'first': {
                        'epochs': 10,
                        'batch_size': 30,
                        'lr': 0.001,
                        'loss': 'categorical_crossentropy',
                        'validation_split': 0.3
                    }
                }
            },
            'classifier_params': {
                'use_top_k': 0,#if top == 0 threasholds will be evaluated
                'roc_thresholds': True,
                'custom_thresholds': [0.00005, 0.0001, 0.0004],
                'recalulate_probas': False,
                'probas_fullpath': '{exp_name}/control_probas_{{dataset_type}}_topk_{topk}.npy',
            }, 
            'outputs': {
                'results_fullpath': '{exp_name}/control_{dataset_type}_results.csv'
            }
        }
    },
    'dp_gen': {
        'run_iterations': 2, #0 means skip, n > 0 generate n interations and run utility_tests for each iterations (if no skip in utility), < 0 run utility_tests n times with old generated files
        'epsilon_tries': [100, 500, 1000], #for each epsilon will generate 'run_iterations' privatizations
        'module_name': 'study_cases.deeplog.deeplog_dp_gen_emb',
        'class_name': 'DeepLogDPGen',
        'params': {
            'datasets_params': {
                'train': {
                    'normal': {
                        'fullpath': '{exp_name}/normal_train.txt',
                        'to_read': -1
                    },
                    'abnormal': {
                        'fullpath': '{exp_name}/abnormal_train.txt',
                        'to_read': -1
                    }
                },
                'val': {
                    'normal': {
                        'fullpath': '{exp_name}/normal_val.txt',
                        'to_read': -1,
                    },
                    'abnormal': {
                        'fullpath': '{exp_name}/abnormal_val.txt',
                        'to_read': -1
                    }
                },
                'to_privatize': {
                    'normal_train': {
                        'fullpath': '{exp_name}/normal_train.txt',
                        'to_read': -1
                    },
                    'abnormal_train': {
                        'fullpath': '{exp_name}/abnormal_train.txt',
                        'to_read': -1
                    },
                    'normal_val': {
                        'fullpath': '{exp_name}/normal_val.txt',
                        'to_read': -1
                    },
                    'abnormal_val': {
                        'fullpath': '{exp_name}/abnormal_val.txt',
                        'to_read': -1
                    },
                    'normal_test': {
                        'fullpath': '{exp_name}/normal_test.txt',
                        'to_read': -1
                    },
                    'abnormal_test': {
                        'fullpath': '{exp_name}/abnormal_test.txt',
                        'to_read': -1
                    }
                }
            },
            'network_fullpath': '{exp_name}/deeplog_dp_gen_emb.h5',
            'network_params': {
                'model_type': 'gen',
                'vocab_size': 29,
                'emb_size': 4,
                'context_size': 4,
                'train_sessions': {
                    'first': {
                        'epochs': 10,
                        'batch_size': 500,
                        'lr': 0.0001,
                        'loss': 'categorical_crossentropy',
                        'validation_split': 0.3
                    },
                    'second': {
                        'epochs': 15,
                        'batch_size': 100,
                        'lr': 0.0001,
                        'loss': 'categorical_crossentropy',
                        'validation_split': 0.3
                    }
                }
            },
            'pre_proba_matrix_fullpath': '{exp_name}/pre_proba_matrix.npy',
            'to_privatize_output_fullpath': '{exp_name}/fake_{{to_privatize_name}}_eps_{{epsilon}}_{{iteration}}.txt'
        },
        'utility_test': {
            'skip': False, #the iterations are given by dp_gen iterations
            'module_name': 'study_cases.deeplog.deeplog_lm',
            'class_name': 'DeepLogLMClassifier',
            'params': {
                'datasets_params': {
                    'train': {
                        'normal': {
                            'fullpath': '{exp_name}/fake_normal_train_eps_{epsilon}_{iteration}.txt',
                            'to_read': -1
                        }
                    },
                    'val': {
                        'normal': {
                            'fullpath': '{exp_name}/fake_normal_val_eps_{epsilon}_{iteration}.txt',
                            'to_read': -1,
                            'class': 1
                        },
                        'abnormal': {
                            'fullpath': '{exp_name}/fake_abnormal_val_eps_{epsilon}_{iteration}.txt',
                            'to_read': -1,
                            'class': 0
                        }
                    },      
                    'test': {
                        'normal': {
                            'fullpath': '{exp_name}/fake_normal_test_eps_{epsilon}_{iteration}.txt',
                            'to_read': -1,
                            'class': 1
                        },
                        'abnormal': {
                            'fullpath': '{exp_name}/fake_abnormal_test_eps_{epsilon}_{iteration}.txt',
                            'to_read': -1,
                            'class': 0
                        }
                    }
                },
                'network_fullpath': '{exp_name}/deeplog_utility_eps_{epsilon}_{iteration}.h5',
                'network_params': {
                    'model_type': 'utility',
                    'window_size': 10,
                    'vocab_size': 29,
                    'train_sessions': {
                        'first': {
                            'epochs': 20,
                            'batch_size': 100,
                            'lr': 0.001,
                            'loss': 'categorical_crossentropy',
                            'validation_split': 0.3
                        }
                    }
                },
                'classifier_params': {
                    'use_top_k': 0,#if top == 0 threasholds will be evaluated
                    'roc_thresholds': True,
                    'custom_thresholds': [0.000009, 0.00005, 0.0001, 0.0002, 0.0004],
                    'recalulate_probas': False,
                    'probas_fullpath': '{exp_name}/utility_probas_{{dataset_type}}_topk_{topk}_eps_{epsilon}_{iteration}.npy',
                }, 
                'outputs': {
                    'results_fullpath': '{exp_name}/utility_{dataset_type}_results.csv'
                }
            }
        }
    }
}
