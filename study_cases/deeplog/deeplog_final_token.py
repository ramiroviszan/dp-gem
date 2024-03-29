import random
import numpy as np

from sklearn.model_selection import train_test_split

import common.data_utils as data_utils
from common.data_splitter import DataSplitter

class DeepLogDataSplitter(DataSplitter):

    def __init__(self, experiment, datasets):
        super().__init__(experiment, datasets)

    def tweak_data_before_split(self, data):
        _, _, vocab_size = data_utils.dataset_vocab(data_utils.combine_datasets(data.values()))

        for dataset_name, dataset in data.items():
            data[dataset_name] = [np.append(seq, vocab_size) for seq in dataset] #Add vocab_size at the end for as ending token

        return data