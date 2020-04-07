experiment = {
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
                    'normal': {
                        'fullpath': '{exp_name}/normal_train.txt',
                        'to_read': -1
                    }
                },      
                'test': {
                    'normal': {
                        'fullpath': '{exp_name}/normal.txt',
                        'to_read': 1000,
                        'class': 1
                    },
                    'abnormal': {
                        'fullpath': '{exp_name}/abnormal.txt',
                        'to_read': 1000,
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
                        'validation_split': 0.3
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
                'train': {
                    'normal': {
                        'fullpath': '{exp_name}/normal.txt',
                        'to_read': 4000
                    },
                    'abnormal': {
                        'fullpath': '{exp_name}/abnormal.txt',
                        'to_read': 4000
                    }
                },      
                'to_privatize': {
                    'normal': {
                        'fullpath': '{exp_name}/normal.txt',
                        'to_read': 4000
                    },
                    'abnormal': {
                        'fullpath': '{exp_name}/abnormal.txt',
                        'to_read': 4000
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
                        'epochs': 10,
                        'batch_size': 500,
                        'lr': 0.0001,
                        'loss': 'categorical_crossentropy',
                        'validation_split': 0.3
                    },
                    'second': {
                        'epochs': 10,
                        'batch_size': 100,
                        'lr': 0.0001,
                        'loss': 'categorical_crossentropy',
                        'validation_split': 0.3
                    }
                }
            },
            'pre_proba_matrix_fullpath': '{exp_name}/pre_proba_matrix.npy',
            'generation_params':{
                'epsilon': 3,
                'delta': 0
            },
            'to_privatize_output_fullpath': '{exp_name}/fake_data_{{to_privatize_name}}_{{iteration}}.txt'
        },
        'utility_test': {
            'module_name': 'study_cases.deeplog.deeplog_lm',
            'class_name': 'DeepLogLMClassifier',
            'params': {
                'datasets_params': {
                    'train': {
                        'normal': {
                            'fullpath': '{exp_name}/fake_data_normal_{iteration}.txt',
                            'to_read': -1
                        }
                    },      
                    'test': {
                        'normal': {
                            'fullpath': '{exp_name}/fake_data_normal_{iteration}.txt',
                            'to_read': 1000,
                            'class': 1
                        },
                        'abnormal': {
                            'fullpath': '{exp_name}/fake_data_abnormal_{iteration}.txt',
                            'to_read': 1000,
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
                            'validation_split': 0.3
                        }
                    }
                },
                'classifier_params': {
                    'use_top_k': 0,
                    'thresholds': [0.0000009, 0.00005, 0.0001],
                    'recalulate_probas': True,
                    'probas_fullpath': '{exp_name}/utility_probas_topk_{topk}_{iteration}.npy',
                    'results_fullpath': '{exp_name}/utility_results.csv',
                    'plots_fullpath': '{exp_name}/plot_{{uuid}}.png'
                }
            }
        }
    }
}
