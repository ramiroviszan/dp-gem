experiment = {
    'skip': 0,
    'random_seed': 27,
    'data_preparation': {
        'iterations': 0,
        "mode": "all",
        'module_name': 'common.data_splitter',
        'class_name': 'DataSplitter',
        'params': {
            'datasets': {
                'normal': {
                    'original': {
                        'fullpath': 'data/waf/token_all_normal.txt',
                        'to_read': 0.2, #0 = 100%, (0, 1) = %, >= 1 count
                        'shuffle': True,
                        'max_len': 200,
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
                        'fullpath': 'data/waf/token_all_abnormal.txt',
                        'to_read': 0.2,
                        'shuffle': True,
                        'max_len': 200,
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
        'iterations': 1,
        "mode": "all",
        'module_name': 'study_cases.waf.lm_classifier',
        'class_name': 'LMClassifier',
        'params': {
            'datasets_params': {
                'train': {
                    'normal': {
                        'fullpath': '{exp_name}/abnormal_train.txt',
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
                'model_type': 'control',
                'model_params': {
                    'vocab_size': 257,  #Real vocab goes from 1-256, 0-257 with padding
                    'window_size': 10,
                    'emb_size': 4,
                    'dropout': 0.1,
                    'hidden_layers': [512, 512, 512]
                },
                'train_sessions': {
                    'first': {
                        'epochs': 50,
                        'batch_size': 100,
                        'lr': 0.01,
                        'loss': 'categorical_crossentropy',
                        'validation_split': 0.3,
                        'patience': 5,
                        'save_model': False
                    },
                    'second': {
                        'epochs': 50,
                        'batch_size': 30,
                        'lr': 0.001,
                        'loss': 'categorical_crossentropy',
                        'validation_split': 0.3,
                        'patience': 5,
                        'save_model': True
                    }
                }
            },
            'classifier_params': {
                'use_top_k': 0,  # if top == 0 threasholds will be evaluated
                'roc_thresholds': True,
                'custom_thresholds': [],
                'recalulate_probas': False,
                'probas_fullpath': '{exp_name}/control_probas_{{dataset_type}}_topk_{topk}.npy'
            },
            'results_fullpath': '{exp_name}/control_{dataset_type}_results.csv'
        }
    },
    'dp_gen': {
        'interations': 1,
        'trials': [
            {'eps': 'no_dp', 'maxdelta':0},#no dp
            {'eps': 0.5, 'maxdelta':1},
            {'eps': 1, 'maxdelta':1},
            {'eps': 10, 'maxdelta':1},
            {'eps': 20, 'maxdelta':1},
            {'eps': 30, 'maxdelta':1},
            {'eps': 40, 'maxdelta':1}],
        'mode': 'all',  # all, main_only, submodules_only, skip
        'module_name': 'study_cases.waf.dp_gen_lap_autoencoder',
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
                'model_type': 'gen_lap_autoencoder',
                'vocab_size': 257,
                'window_size': 150,
                'emb_size': 4,
                'hidden_state_size': 1024,
                'train_sessions': {
                    'first': {
                        'epochs': 1000,
                        'batch_size': 500,
                        'lr': 0.0001,
                        'loss': 'binary_crossentropy',
                        'validation_split': 0.3,
                        'patience': 20,
                        'save_model': False
                    },
                    'second': {
                        'epochs': 1000,
                        'batch_size': 100,
                        'lr': 0.00001,
                        'loss': 'binary_crossentropy',
                        'validation_split': 0.3,
                        'patience': 10,
                        'save_model': True
                    }
                }
            },
            'to_privatize_output_fullpath': '{exp_name}/fake_{{to_privatize_name}}_{{eps}}_{{iteration}}.txt'
        },
        'submodules': {
            'classifier': {
                'iterations': 1,  # the iterations are given by dp_gen iterations
                'mode': 'all',
                'module_name': 'study_cases.waf.lm_classifier',
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
                        'vocab_size': 257,
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
                'iterations': 1,  # the iterations are given by dp_gen iterations
                'mode': 'all',
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