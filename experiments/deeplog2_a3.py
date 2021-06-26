experiment = {
    'skip': 0,
    'random_seed': 27,
    'data_preparation': {
        'skip': 0,
        'module_name': 'study_cases.deeplog2.final_token',
        'class_name': 'CustomDataSplitter',
        'build_params': {
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
                    'train_output_fullpath': '{exp_path}/normal_train.txt',
                    'val_output_fullpath': '{exp_path}/normal_val.txt',
                    'test_output_fullpath': '{exp_path}/normal_test.txt',
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
                    'train_output_fullpath': '{exp_path}/abnormal_train.txt',
                    'val_output_fullpath': '{exp_path}/abnormal_val.txt',
                    'test_output_fullpath': '{exp_path}/abnormal_test.txt',
                    'splits': {
                        'train_test': 0.3,
                        'train_val': 0.3
                    }
                }
            }
        },
        'trials_params': [{}]
    },
    'control_test': {
        'skip':0,
        'module_name': 'study_cases.deeplog2.classifier',
        'class_name': 'Classifier',
        'build_params': {
            'datasets_params': {
                'train': {
                    'normal': {
                        'fullpath': '{exp_path}/normal_train.txt',
                        'to_read': 0
                    }
                },
                'val': {
                    'normal': {
                        'fullpath': '{exp_path}/normal_val.txt',
                        'to_read': 0,
                        'class': 1
                    },
                    'abnormal': {
                        'fullpath': '{exp_path}/abnormal_val.txt',
                        'to_read': 0,
                        'class': 0
                    }
                },
                'test': {
                    'normal': {
                        'fullpath': '{exp_path}/normal_test.txt',
                        'to_read': 0,
                        'class': 1
                    },
                    'abnormal': {
                        'fullpath': '{exp_path}/abnormal_test.txt',
                        'to_read': 0,
                        'class': 0
                    }
                }
            },
            'network_fullpath': '{exp_path}/control.h5',
            'network_params': {
                'model_type': 'control_model',
                'model_params': {
                    'vocab_size': 31, #this value considers padding, 30 without
                    'window_size': 10,
                    'hidden_layers': [256]
                },
                'train_sessions': {
                    'first': {
                        'epochs': 1000,
                        'batch_size': 100,
                        'lr': 0.001,
                        'loss': 'categorical_crossentropy',
                        'validation_split': 0.3,
                        'patience': 5,
                        'save_model': False
                    },
                    'second': {
                        'epochs': 500,
                        'batch_size': 30,
                        'lr': 0.001,
                        'loss': 'categorical_crossentropy',
                        'validation_split': 0.3,
                        'patience': 5,
                        'save_model': True
                    }
                }
            },
            'results_fullpath': '{exp_path}/control_{dataset_type}_results.csv'
        },
        'trials_params': [{
            'recalulate_probas': False,
            'probas_fullpath': '{exp_path}/control_probas_{{dataset_type}}.npy'
        }]
    },
    'dp_gen': {
        'skip': 0,  
        'module_name': 'study_cases.deeplog2.dp_gen_exponential_class_emb',
        'class_name': 'Gen',
        'mode': 'all',  # all, main_only, submodules_only
        'build_params': {
            'datasets_params': {
                'train': {
                    'normal': {
                        'fullpath': '{exp_path}/normal_train.txt',
                        'to_read': 0,
                        'class': 1
                    },
                    'abnormal': {
                        'fullpath': '{exp_path}/abnormal_train.txt',
                        'to_read': 0,
                        'class': 0
                    }
                },
                'val': {
                    'normal': {
                        'fullpath': '{exp_path}/normal_val.txt',
                        'to_read': 0,
                    },
                    'abnormal': {
                        'fullpath': '{exp_path}/abnormal_val.txt',
                        'to_read': 0
                    }
                },
                'to_privatize': {
                    'normal_train': {
                        'fullpath': '{exp_path}/normal_train.txt',
                        'to_read': 0
                    },
                    'abnormal_train': {
                        'fullpath': '{exp_path}/abnormal_train.txt',
                        'to_read': 0
                    },
                    'normal_val': {
                        'fullpath': '{exp_path}/normal_val.txt',
                        'to_read': 0
                    },
                    'abnormal_val': {
                        'fullpath': '{exp_path}/abnormal_val.txt',
                        'to_read': 0
                    },
                    'normal_test': {
                        'fullpath': '{exp_path}/normal_test.txt',
                        'to_read': 0
                    },
                    'abnormal_test': {
                        'fullpath': '{exp_path}/abnormal_test.txt',
                        'to_read': 0
                    }
                }
            },
            'network_fullpath': '{exp_path}/gen.h5',
            'network_params': {
                'model_type': 'dp_gen_emb_classifier',
                'model_params': {
                    'vocab_size': 31, #this value considers padding, 30 without
                    'emb_size': 8,
                    'max_len': 0, 
                    'hidden_layers': [512, 256, 128]
                },
                'train_sessions': {
                    'first': {
                        'epochs': 1000,
                        'batch_size': 500,
                        'lr': 0.0001,
                        'loss': 'binary_crossentropy',
                        'validation_split': 0.3,
                        'patience': 5,
                        'save_model': False
                    },
                    'second': {
                        'epochs': 1000,
                        'batch_size': 100,
                        'lr': 0.00001,
                        'loss': 'binary_crossentropy',
                        'validation_split': 0.3,
                        'patience': 5,
                        'save_model': True
                    }
                }
            },
            'pre_proba_matrix_fullpath': '{exp_path}/pre_proba_matrix.npy',
            'to_privatize_output_fullpath': '{exp_path}/fake_{{to_privatize_name}}_{{trial}}.txt'
        },
        'trials_params': [
            {'iter': 0, 'eps': 0.05},
            {'iter': 1, 'eps': 0.5},
            {'iter': 2, 'eps': 1},
            {'iter': 3, 'eps': 10},
            {'iter': 4, 'eps': 50},
            {'iter': 5, 'eps': 100},
        ],
        'submodules': {
            'classifier': {
                'skip': 0, 
                'module_name': 'study_cases.deeplog2.classifier',
                'class_name': 'Classifier',
                'build_params': {
                    'datasets_params': {
                        'train': {
                            'normal': {
                                'fullpath': '{exp_path}/fake_normal_train_{parent_trial}.txt',
                                'to_read': 0
                            }
                        },
                        'val': {
                            'normal': {
                                'fullpath': '{exp_path}/fake_normal_val_{parent_trial}.txt',
                                'to_read': 0,
                                'class': 1
                            },
                            'abnormal': {
                                'fullpath': '{exp_path}/fake_abnormal_val_{parent_trial}.txt',
                                'to_read': 0,
                                'class': 0
                            }
                        },
                        'test': {
                            'normal': {
                                'fullpath': '{exp_path}/fake_normal_test_{parent_trial}.txt',
                                'to_read': 0,
                                'class': 1
                            },
                            'abnormal': {
                                'fullpath': '{exp_path}/fake_abnormal_test_{parent_trial}.txt',
                                'to_read': 0,
                                'class': 0
                            }
                        }
                    },
                    'network_fullpath': '{exp_path}/utility_{parent_trial}.h5',
                    'network_params': {
                        'model_type': 'control_model',
                        'model_params': {
                            'vocab_size': 31, #this value considers padding, 30 without
                            'window_size': 10,
                            'hidden_layers': [256]
                        },
                        'train_sessions': {
                            'first': {
                                'epochs': 1000,
                                'batch_size': 100,
                                'lr': 0.001,
                                'loss': 'categorical_crossentropy',
                                'validation_split': 0.3,
                                'patience': 5,
                                'save_model': False
                            },
                            'second': {
                                'epochs': 500,
                                'batch_size': 30,
                                'lr': 0.001,
                                'loss': 'categorical_crossentropy',
                                'validation_split': 0.3,
                                'patience': 5,
                                'save_model': True
                            }
                        }
                    },
                    'results_fullpath': '{exp_path}/utility_classifier_{dataset_type}_results.csv'
                },
                'trials_params': [{
                    'recalulate_probas': False,
                    'probas_fullpath': '{exp_path}/utility_probas_{{dataset_type}}_{parent_trial}.npy',
                }]
            },
            'similarity': {
                'skip': 0,  
                'module_name': 'common.data_similarity',
                'class_name': 'DataSimilarity',
                'build_params': {
                    'metrics': ['hamming', 'hamming_wise', 'cosine'],
                    'datasets_params': {
                        'normal': {
                            'orig_fullpath': '{exp_path}/normal_test.txt',
                            'privatized_fullpath': '{exp_path}/fake_normal_test_{parent_trial}.txt',
                            'to_read': 0,
                            'dtype': int
                        },
                        'abnormal': {
                            'orig_fullpath': '{exp_path}/abnormal_test.txt',
                            'privatized_fullpath': '{exp_path}/fake_abnormal_test_{parent_trial}.txt',
                            'to_read': 0,
                            'dtype': int
                        }
                    },
                    'results_fullpath': '{exp_path}/utility_similarity_test_results.csv'
                },
                'trials_params': [{}]
            }
        }
    }
}
