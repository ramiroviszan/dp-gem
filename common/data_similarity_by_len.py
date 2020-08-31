import numpy as np
from sklearn.metrics.pairwise import pairwise_distances

import common.data_utils as data_utils
from common.csv_result import CSVResult

from common.logger_utils import get_logger

class DataSimilarity:

    def __init__(self, experiment, metrics, datasets_params, results_fullpath):
        self.exp_name, self.exp_path, self.parent_trial = experiment
        self.logger = get_logger('similarity_len', self.exp_name, self.parent_trial) #TODO: Finish this
        self.metrics = metrics
        self.datasets_params = datasets_params

        results_header = list(self.parent_trial.keys()) + ['metric', 'len', 'count', 'mean_all']
        #for dataset_type in self.datasets_params:
        #    results_header.append('mean_' + dataset_type)
        #results_header.append('mean_all')

        self.result_line = dict()
        result_line = list(self.parent_trial.values()) + [self.iteration]
        for metric_name in metrics:
            self.result_line[metric_name] = result_line + [metric_name]

        print(self.result_line)

        results_fullpath = results_fullpath.format(exp_path=self.exp_path)
        self.results = CSVResult(results_fullpath, results_header)

    def run_test(self):
        datasets = []
        for dataset_type in self.datasets_params:
            first, second = self._load_test(*self.datasets_params[dataset_type].values())
            datasets.append((first, second))
            #Per dataset mean
            #for metric_name in self.metrics:
            #    mean = self._calculate_mean_metric(metric_name, first, second)
            #    self.result_line[metric_name].append(mean)

        #All data mean 
        all_data_first = []
        all_data_second = []
        for first, second in datasets:
            all_data_first = data_utils.stack_datasets(all_data_first, first)
            all_data_second = data_utils.stack_datasets(all_data_second, second)

        first_data_by_len = {}
        second_data_by_len = {}
        for i, d in enumerate(all_data_first):
            _len = len(d)
            if first_data_by_len.get(_len) is None:
                first_data_by_len[_len] = [d]
                second_data_by_len[_len] = [all_data_second[i]]
            else:
                first_data_by_len[_len].append(d)
                second_data_by_len[_len].append(all_data_second[i])

        lens = list(first_data_by_len.keys())
        lens.sort()
        for metric_name in self.metrics:
            for l in lens:
                all_mean = self._calculate_mean_metric(metric_name, first_data_by_len[l], second_data_by_len[l])
                res = self.result_line[metric_name]  + [l] + [len(first_data_by_len[l])] + [all_mean]
                self.results.save_results(res)
                
    def _load_test(self, first_fullpath, second_fullpath, to_read, dtype):
        first_fullpath = first_fullpath.format(exp_path=self.exp_path, parent_trial=flat_trial(self.parent_trial))
        second_fullpath = second_fullpath.format(exp_path=self.exp_path, parent_trial=flat_trial(self.parent_trial))
        
        first = data_utils.load_file(first_fullpath, to_read, shuffle=False, dtype=dtype)
        second= data_utils.load_file(second_fullpath, to_read, shuffle=False, dtype=dtype)

        return first, second
        
    def _calculate_mean_metric(self, metric_name, first, second):
        if metric_name == 'hamming_wise':
            mean = np.mean([pairwise_distances(first[i].reshape(1, -1), second[i].reshape(1, -1), metric='hamming') * len(first[i]) for i in range(0, len(first))])
        else: 
            mean = np.mean([pairwise_distances(first[i].reshape(1, -1), second[i].reshape(1, -1), metric=metric_name) for i in range(0, len(first))])
        return mean

