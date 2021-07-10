experiment = {
    'skip': 0,
    'random_seed': 27,
    'control_test': {
        'skip': 0,
        'module_name': 'study_cases.deeplog2.classifier_e3',
        'class_name': 'Classifier',
        'build_params': {
            'datasets_params': {
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
            'network_fullpath': '{exp_path}/gen.h5',
            'results_fullpath': '{exp_path}/control_{dataset_type}_results.csv'
        },
        'trials_params': [{"eps": 0.5, "maxdelta":1, "hidden_state_size": 512}]
    }
}
