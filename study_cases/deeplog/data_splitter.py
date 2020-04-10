import random
import numpy as np

from sklearn.model_selection import train_test_split

import data_utils

class DataSplitter:

    def __init__(self, experiment, datasets):
        self.exp_name = experiment
        self.datasets = datasets
    
    def run(self):
        for dataset_name, dataset_desc in self.datasets.items():
            self._process_dataset(dataset_name, *dataset_desc.values())

    def _process_dataset(self, dataset_name, original, to_read, train_output_fullpath, val_output_fullpath, test_output_fullpath, splits):
        train_output_fullpath = train_output_fullpath.format(exp_name=self.exp_name)
        val_output_fullpath = val_output_fullpath.format(exp_name=self.exp_name)
        test_output_fullpath = test_output_fullpath.format(exp_name=self.exp_name)
        train_test_ratio, train_val_ratio = splits.values()

        all_data = data_utils.load_file(original, to_read, int)
        all_data = self._tweak_data(all_data)

        train, test = train_test_split(all_data, test_size=float(train_test_ratio))
        if train_val_ratio == 1.0:
            val = train
            train = np.array([])
        else:
            train, val = train_test_split(train, test_size=float(train_val_ratio))

        data_utils.write_file(train, train_output_fullpath)
        data_utils.write_file(val, val_output_fullpath)
        data_utils.write_file(test, test_output_fullpath)


    def _tweak_data(self, all_data):
        all_data = all_data - 1 
        all_data = np.array([np.append(seq, 1) for seq in all_data]) 
        return all_data



    
