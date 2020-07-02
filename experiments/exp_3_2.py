experiment = {
    'skip': False,
    'random_seed': 27,
    'data_preparation': {
        'skip': True,
        'module_name': 'study_cases.deeplog.deeplog_final_token',
        'class_name': 'DeepLogDataSplitter',
        'params': {
            'datasets': {
                'normal': {
                    'original': {
                        'fullpath': 'data/deeplog/all_normal.txt',
                        'to_read': 4000,
                        'shuffle': True,
                        'max_len': 50,
                        'dtype': int,
                        'split_token': '',
                        'encoding': 'ascii',
                        'errors': 'strict'
                    },
                    'train_output_fullpath': '{exp_name}/normal_train.txt',
                    'val_output_fullpath': '{exp_name}/normal_val.txt',
                    'test_output_fullpath': '{exp_name}/normal_test.txt',
                    'splits': {
                        'train_test': 0.3,
                        'train_val': 0.3
                    }
                },
                'abnormal': {
                    'original': {
                        'fullpath': 'data/deeplog/all_abnormal.txt',
                        'to_read': 4000,
                        'shuffle': True,
                        'max_len': 50,
                        'dtype': int,
                        'split_token': '',
                        'encoding': 'ascii',
                        'errors': 'strict'
                    },
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
        'module_name': 'study_cases.deeplog.lm_classifier',
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
            'network_fullpath': '{exp_name}/control.h5',
            'network_params': {
                'model_type': 'control_fixed_window',
                'window_size': 10,
                'vocab_size': 31,  # this value considers padding, 30 without
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
                'use_top_k': 0,  # if top == 0 threasholds will be evaluated
                'roc_thresholds': True,
                'custom_thresholds': [],
                'recalulate_probas': False,
                'probas_fullpath': '{exp_name}/control_probas_{{dataset_type}}_topk_{topk}.npy',
            },
            'results_fullpath': '{exp_name}/control_{dataset_type}_results.csv'
        }
    },
    'dp_gen': {
        'run_iterations': 1,  # for each trial bellow will generate 'run_iterations' privatizations
        'trials': [
            {'eps': 20},
            {'eps': 30},
            {'eps': 40},
            {'eps': 50},
            {'eps': 60},
            {'eps': 100}],
        'mode': 'all',  # all, gen_only, tests_only, skip
        'module_name': 'study_cases.deeplog.dp_gen_exponential_class_emb',
        'class_name': 'DPGen',
        'params': {
            'datasets_params': {
                'train': {
                    'normal': {
                        'fullpath': '{exp_name}/normal_train.txt',
                        'to_read': 0,
                        'class': 1
                    },
                    'abnormal': {
                        'fullpath': '{exp_name}/abnormal_train.txt',
                        'to_read': 0,
                        'class': 0
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
            'network_fullpath': '{exp_name}/gen.h5',
            'network_params': {
                'model_type': 'gen_class',
                'vocab_size': 31,
                'emb_size': 8,
                'train_sessions': {
                    'first': {
                        'epochs': 500,
                        'batch_size': 500,
                        'lr': 0.0001,
                        'loss': 'binary_crossentropy',
                        'validation_split': 0.3,
                        'patience': 10,
                        'save_model': False
                    },
                    'second': {
                        'epochs': 500,
                        'batch_size': 100,
                        'lr': 0.00001,
                        'loss': 'binary_crossentropy',
                        'validation_split': 0.3,
                        'patience': 5,
                        'save_model': True
                    }
                }
            },
            'pre_proba_matrix_fullpath': '{exp_name}/pre_proba_matrix.npy',
            'to_privatize_output_fullpath': '{exp_name}/fake_{{to_privatize_name}}_{{eps}}_{{iteration}}.txt'
        },
        'utility_tests': {
            'classifier': {
                'skip': False,  # the iterations are given by dp_gen iterations
                'module_name': 'study_cases.deeplog.lm_classifier',
                'class_name': 'LMClassifier',
                'params': {
                    'datasets_params': {
                        'train': {
                            'normal': {
                                'fullpath': '{exp_name}/fake_normal_train_{eps}_{iteration}.txt',
                                'to_read': 0
                            }
                        },
                        'val': {
                            'normal': {
                                'fullpath': '{exp_name}/fake_normal_val_{eps}_{iteration}.txt',
                                'to_read': 0,
                                'class': 1
                            },
                            'abnormal': {
                                'fullpath': '{exp_name}/fake_abnormal_val_{eps}_{iteration}.txt',
                                'to_read': 0,
                                'class': 0
                            }
                        },
                        'test': {
                            'normal': {
                                'fullpath': '{exp_name}/fake_normal_test_{eps}_{iteration}.txt',
                                'to_read': 0,
                                'class': 1
                            },
                            'abnormal': {
                                'fullpath': '{exp_name}/fake_abnormal_test_{eps}_{iteration}.txt',
                                'to_read': 0,
                                'class': 0
                            }
                        }
                    },
                    'network_fullpath': '{exp_name}/deeplog_utility_{eps}_{iteration}.h5',
                    'network_params': {
                        'model_type': 'control_fixed_window',
                        'window_size': 10,
                        'vocab_size': 31,
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
                        'use_top_k': 0,  # if top == 0 threasholds will be evaluated
                        'roc_thresholds': True,
                        'custom_thresholds': [],
                        'recalulate_probas': False,
                        'probas_fullpath': '{exp_name}/utility_probas_{{dataset_type}}_topk_{topk}_{eps}_{iteration}.npy',
                    },
                    'results_fullpath': '{exp_name}/utility_classifier_{dataset_type}_results.csv'
                }
            },
            'similarity': {
                'skip': False,  # the iterations are given by dp_gen iterations
                'module_name': 'common.data_similarity',
                'class_name': 'DataSimilarity',
                'params': {
                    'metrics': ['hamming', 'hamming_wise', 'cosine'],
                    'datasets_params': {
                        'normal': {
                            'orig_fullpath': '{exp_name}/normal_test.txt',
                            'privatized_fullpath': '{exp_name}/fake_normal_test_{eps}_{iteration}.txt',
                            'to_read': 0,
                            'dtype': int
                        },
                        'abnormal': {
                            'orig_fullpath': '{exp_name}/abnormal_test.txt',
                            'privatized_fullpath': '{exp_name}/fake_abnormal_test_{eps}_{iteration}.txt',
                            'to_read': 0,
                            'dtype': int
                        }
                    },
                    'results_fullpath': '{exp_name}/utility_similarity_test_results.csv'
                }
            }
        }
    }
}
