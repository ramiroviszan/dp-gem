experiment = {
    'skip': True,
    'random_seed': 27,
    'data_preparation': {
        'skip': False,
        'module_name': 'study_cases.deeplog.deeplog_data_splitter',
        'class_name': 'DeepLogDataSplitter',
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
        'run_iterations': 0,
        'module_name': 'study_cases.deeplog.language_model_classifier',
        'class_name': 'LMClassifier',
        'params': {
            'datasets_params': {
                'train': {
                    'normal': {
                        'fullpath': '{exp_name}/normal_train.txt',
                        'to_read': 0
                    }
                },   
                'val': {
                    'normal': {
                        'fullpath': '{exp_name}/normal_val.txt',
                        'to_read': 0,
                        'class': 1
                    },
                    'abnormal': {
                        'fullpath': '{exp_name}/abnormal_val.txt',
                        'to_read': 0,
                        'class': 0
                    }
                },
                'test': {
                    'normal': {
                        'fullpath': '{exp_name}/normal_test.txt',
                        'to_read': 0,
                        'class': 1
                    },
                    'abnormal': {
                        'fullpath': '{exp_name}/abnormal_test.txt',
                        'to_read': 0,
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
                        'epochs': 100,
                        'batch_size': 100,
                        'lr': 0.001,
                        'loss': 'categorical_crossentropy',
                        'validation_split': 0.3,
                        'patience': 10,
                        'save_model': False
                    },
                    'second': {
                        'epochs': 50,
                        'batch_size': 30,
                        'lr': 0.001,
                        'loss': 'categorical_crossentropy',
                        'validation_split': 0.3,
                        'patience': 10,
                        'save_model': True
                    }
                }
            },
            'classifier_params': {
                'use_top_k': 0,#if top == 0 threasholds will be evaluated
                'roc_thresholds': True,
                'custom_thresholds': [],
                'recalulate_probas': False,
                'probas_fullpath': '{exp_name}/control_probas_{{dataset_type}}_topk_{topk}.npy',
            }, 
            'results_fullpath': '{exp_name}/control_{dataset_type}_results.csv'
        }
    },
    'dp_gen': {
        'run_iterations': 1,
        'epsilon_trials': [10, 20, 30, 40, 100], #for each epsilon will generate 'run_iterations' privatizations
        'mode': 'all', #all, gen_only, tests_only, skip
        'module_name': 'study_cases.deeplog.dp_gen_exponential_emb',
        'class_name': 'DPGenExponentialEmbedding',
        'params': {
            'datasets_params': {
                'train': {
                    'normal': {
                        'fullpath': '{exp_name}/normal_train.txt',
                        'to_read': 0
                    },
                    'abnormal': {
                        'fullpath': '{exp_name}/abnormal_train.txt',
                        'to_read': 0
                    }
                },
                'val': {
                    'normal': {
                        'fullpath': '{exp_name}/normal_val.txt',
                        'to_read': 0,
                    },
                    'abnormal': {
                        'fullpath': '{exp_name}/abnormal_val.txt',
                        'to_read': 0
                    }
                },
                'to_privatize': {
                    'normal_train': {
                        'fullpath': '{exp_name}/normal_train.txt',
                        'to_read': 0
                    },
                    'abnormal_train': {
                        'fullpath': '{exp_name}/abnormal_train.txt',
                        'to_read': 0
                    },
                    'normal_val': {
                        'fullpath': '{exp_name}/normal_val.txt',
                        'to_read': 0
                    },
                    'abnormal_val': {
                        'fullpath': '{exp_name}/abnormal_val.txt',
                        'to_read': 0
                    },
                    'normal_test': {
                        'fullpath': '{exp_name}/normal_test.txt',
                        'to_read': 0
                    },
                    'abnormal_test': {
                        'fullpath': '{exp_name}/abnormal_test.txt',
                        'to_read': 0
                    }
                }
            },
            'network_fullpath': '{exp_name}/deeplog_dp_gen_emb.h5',
            'network_params': {
                'model_type': 'gen',
                'vocab_size': 29,
                'emb_size': 4,
                'context_size': 10,
                'train_sessions': {
                    'first': {
                        'epochs': 100,
                        'batch_size': 500,
                        'lr': 0.0001,
                        'loss': 'categorical_crossentropy',
                        'validation_split': 0.3,
                        'patience': 10,
                        'save_model': False
                    },
                    'second': {
                        'epochs': 50,
                        'batch_size': 100,
                        'lr': 0.0001,
                        'loss': 'categorical_crossentropy',
                        'validation_split': 0.3,
                        'patience': 10,
                        'save_model': True
                    }
                }
            },
            'pre_proba_matrix_fullpath': '{exp_name}/pre_proba_matrix.npy',
            'to_privatize_output_fullpath': '{exp_name}/fake_{{to_privatize_name}}_eps_{{epsilon}}_{{iteration}}.txt'
        },
        'utility_tests': {
            'classifier': {
                'skip': False, #the iterations are given by dp_gen iterations
                'module_name': 'study_cases.deeplog.language_model_classifier',
                'class_name': 'LMClassifier',
                'params': {
                    'datasets_params': {
                        'train': {
                            'normal': {
                                'fullpath': '{exp_name}/fake_normal_train_eps_{epsilon}_{iteration}.txt',
                                'to_read': 0
                            }
                        },
                        'val': {
                            'normal': {
                                'fullpath': '{exp_name}/fake_normal_val_eps_{epsilon}_{iteration}.txt',
                                'to_read': 0,
                                'class': 1
                            },
                            'abnormal': {
                                'fullpath': '{exp_name}/fake_abnormal_val_eps_{epsilon}_{iteration}.txt',
                                'to_read': 0,
                                'class': 0
                            }
                        },      
                        'test': {
                            'normal': {
                                'fullpath': '{exp_name}/fake_normal_test_eps_{epsilon}_{iteration}.txt',
                                'to_read': 0,
                                'class': 1
                            },
                            'abnormal': {
                                'fullpath': '{exp_name}/fake_abnormal_test_eps_{epsilon}_{iteration}.txt',
                                'to_read': 0,
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
                                'epochs': 100,
                                'batch_size': 100,
                                'lr': 0.001,
                                'loss': 'categorical_crossentropy',
                                'validation_split': 0.3,
                                'patience': 10,
                                'save_model': False
                            },
                            'second': {
                                'epochs': 50,
                                'batch_size': 30,
                                'lr': 0.001,
                                'loss': 'categorical_crossentropy',
                                'validation_split': 0.3,
                                'patience': 10,
                                'save_model': True
                            }
                        }
                    },
                    'classifier_params': {
                        'use_top_k': 0,#if top == 0 threasholds will be evaluated
                        'roc_thresholds': True,
                        'custom_thresholds': [],
                        'recalulate_probas': False,
                        'probas_fullpath': '{exp_name}/utility_probas_{{dataset_type}}_topk_{topk}_eps_{epsilon}_{iteration}.npy',
                    }, 
                    'results_fullpath': '{exp_name}/utility_classifier_{dataset_type}_results.csv'
                }
            },
            'similarity':{
                'skip': False, #the iterations are given by dp_gen iterations
                'module_name': 'study_cases.deeplog.data_similarity',
                'class_name': 'DataSimilarity',
                'params': {
                    'metrics': ['hamming', 'hamming_wise', 'cosine'],
                    'datasets_params': {
                        'normal': {
                            'orig_fullpath': '{exp_name}/normal_test.txt',
                            'privatized_fullpath': '{exp_name}/fake_normal_test_eps_{epsilon}_{iteration}.txt',
                            'to_read': 0
                        },
                        'abnormal':{
                            'orig_fullpath': '{exp_name}/abnormal_test.txt',
                            'privatized_fullpath': '{exp_name}/fake_abnormal_test_eps_{epsilon}_{iteration}.txt',
                            'to_read': 0
                        }
                    },
                    'results_fullpath': '{exp_name}/utility_similarity_test_results.csv'
                }
            }
        }
    }
}
