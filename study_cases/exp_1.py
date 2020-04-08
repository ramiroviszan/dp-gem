experiment = {
    'random_seed': 27,
    'data_preparation': {
        'skip': True,
        'module_name': 'study_cases.deeplog.data_splitter',
        'class_name': 'DataSplitter',
        'data': {
            'normal': {
                'original': 'data/deeplog/normal.txt',
                'to_read': 16000,
                'train_output_fullpath' = '{exp_name}/normal_train.txt',
                'val_output_fullpath' = '{exp_name}/normal_val.txt',
                'test_output_fullpath' = '{exp_name}/normal_test.txt',
                'splits': {
                    'train_test': 0.3,
                    'train_val': 0.3
                }
            },
            'abnormal': {
                'original': 'data/deeplog/abnormal.txt',
                'to_read': 16000,
                'train_output_fullpath' = '{exp_name}/abnormal_train.txt',
                'val_output_fullpath' = '{exp_name}/abnormal_val.txt',
                'test_output_fullpath' = '{exp_name}/abnormal_test.txt',
                'splits': {
                    'train_test': 0.3,
                    'train_val': 1.0 #as abnormal train is not required for training, use all for validation
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
                        'epochs': 1,
                        'batch_size': 30,
                        'lr': 0.001,
                        'loss': 'categorical_crossentropy',
                        'validation_split': 0.3,
                        'show_history': True
                    }
                }
            },
            'classifier_params': {
                'use_top_k': 0,#if top == 0 threasholds will be evaluated
                'roc_thresholds': True,
                'custom_thresholds': [0.00005, 0.0001],
                'recalulate_probas': False,
                'probas_fullpath': '{exp_name}/control_probas_topk_{topk}.npy',
                'results_fullpath': '{exp_name}/control_results.csv',
                'plots_fullpath': '{exp_name}/plot_{{uuid}}.png'
            }
        }
    },
    'dp_gen': {
        'run_iterations': 2, #0 means skip, n > 0 generate n interations and run utility_tests for each iterations (if no skip in utility), < 0 run utility_tests n times with old generated files
        'module_name': 'study_cases.deeplog.deeplog_dp_gen_emb',
        'class_name': 'DeepLogDPGen',
        'params': {
            'datasets_params': {
                'train': {
                    'normal': {
                        'fullpath': '{exp_name}/normal_train.txt',
                        'to_read': -1
                    }
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
                    }
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
                'emb_size': 16,
                'context_size': 4,
                'train_sessions': {
                    'first': {
                        'epochs': 1,
                        'batch_size': 500,
                        'lr': 0.0001,
                        'loss': 'categorical_crossentropy',
                        'validation_split': 0.3,
                        'show_history': True
                    },
                    'second': {
                        'epochs': 1,
                        'batch_size': 100,
                        'lr': 0.0001,
                        'loss': 'categorical_crossentropy',
                        'validation_split': 0.3,
                        'show_history': True
                    }
                }
            },
            'pre_proba_matrix_fullpath': '{exp_name}/pre_proba_matrix.npy',
            'generation_params':{
                'epsilon': 3,
                'delta': 0
            },
            'to_privatize_output_fullpath': '{exp_name}/fake_{{to_privatize_name}}_{{iteration}}.txt'
        },
        'utility_test': {
            'skip': False, #the iterations are given by dp_gen iterations
            'module_name': 'study_cases.deeplog.deeplog_lm',
            'class_name': 'DeepLogLMClassifier',
            'params': {
                'datasets_params': {
                    'train': {
                        'normal': {
                            'fullpath': '{exp_name}/fake_normal_train_{iteration}.txt',
                            'to_read': -1
                        }
                    },
                    'val': {
                        'normal': {
                            'fullpath': '{exp_name}/fake_normal_val_{iteration}.txt',
                            'to_read': -1,
                            'class': 1
                        },
                        'abnormal': {
                            'fullpath': '{exp_name}/abnormal_val_{iteration}.txt',
                            'to_read': -1,
                            'class': 0
                        }
                    },      
                    'test': {
                        'normal': {
                            'fullpath': '{exp_name}/fake_normal_test_{iteration}.txt',
                            'to_read': -1,
                            'class': 1
                        },
                        'abnormal': {
                            'fullpath': '{exp_name}/fake_abnormal_test_{iteration}.txt',
                            'to_read': -1,
                            'class': 0
                        }
                    }
                },
                'network_fullpath': '{exp_name}/deeplog_utility_{iteration}.h5',
                'network_params': {
                    'model_type': 'utility',
                    'window_size': 10,
                    'vocab_size': 29,
                    'train_sessions': {
                        'first': {
                            'epochs': 1,
                            'batch_size': 30,
                            'lr': 0.001,
                            'loss': 'categorical_crossentropy',
                            'validation_split': 0.3,
                            'show_history': True
                        }
                    }
                },
                'classifier_params': {
                    'use_top_k': 0,#if top == 0 threasholds will be evaluated
                    'roc_thresholds': True,
                    'custom_thresholds': [0.000009, 0.00005, 0.0001],
                    'recalulate_probas': True,
                    'probas_fullpath': '{exp_name}/utility_probas_topk_{topk}_{iteration}.npy',
                    'results_fullpath': '{exp_name}/utility_results.csv',
                    'plots_fullpath': '{exp_name}/plot_{{uuid}}.png'
                }
            }
        }
    }
}
