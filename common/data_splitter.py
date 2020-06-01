import random
import numpy as np
from sklearn.model_selection import train_test_split

import common.data_utils as data_utils


class DataSplitter:

    def __init__(self, experiment, datasets):
        self.exp_name = experiment
        self.datasets = datasets

    def run(self):
        loaded = dict()
        for dataset_name, dataset_desc in self.datasets.items():
            loaded[dataset_name] = data_utils.load_file(*dataset_desc['original'].values())

        loaded = self.tweak_data_before_split(loaded)

        for dataset_name, dataset_desc in self.datasets.items():
            self._split_dataset(loaded[dataset_name],
                                *list(dataset_desc.values())[1:])  #1=removes 'original'

    def tweak_data_before_split(self, data):
        return data

    def _split_dataset(self, data, train_output_fullpath, val_output_fullpath, test_output_fullpath, splits):
        train_output_fullpath = train_output_fullpath.format(
            exp_name=self.exp_name)
        val_output_fullpath = val_output_fullpath.format(
            exp_name=self.exp_name)
        test_output_fullpath = test_output_fullpath.format(
            exp_name=self.exp_name)
        train_test_ratio, train_val_ratio = splits.values()

        train, test = train_test_split(data, test_size=float(train_test_ratio))
        if train_val_ratio == 1.0:
            val = train
            train = []
        else:
            train, val = train_test_split(
                train, test_size=float(train_val_ratio))

        data_utils.write_file(train, train_output_fullpath)
        data_utils.write_file(val, val_output_fullpath)
        data_utils.write_file(test, test_output_fullpath)
