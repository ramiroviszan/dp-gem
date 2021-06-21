import random
import numpy as np

import common.data_utils as data_utils
from common.data_splitter import DataSplitter

class CustomDataSplitter(DataSplitter):

    def __init__(self, experiment, datasets):
        super().__init__(experiment, datasets)

    def tweak_data_before_split(self, data):
        _, _, vocab_size = data_utils.dataset_vocab(data_utils.combine_datasets(data.values()))

        for dataset_name, dataset in data.items():
            data[dataset_name] = [np.append(seq, vocab_size) for seq in dataset] #Add vocab_size at the end for as ending token

        return data