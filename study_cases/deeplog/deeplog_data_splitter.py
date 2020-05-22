import random
import numpy as np

from sklearn.model_selection import train_test_split

from common.data_splitter import DataSplitter

class DeepLogDataSplitter(DataSplitter):

    def __init__(self, experiment, datasets):
        super().__init__(experiment, datasets)

    def tweak_data_before_split(self, data):
        for dataset_name, dataset in data.items():
            data[dataset_name] = [np.append(seq -1, 0) for seq in dataset]  #-1 to optimize vocab and add 0 at the end as ending token

        return data