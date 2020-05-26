
import random
import numpy as np

from itertools import chain
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences


def stack_datasets(first, second, axis=0, verbose=0):
    if len(first) == 0:
        return second

    if len(second) == 0:
        return first

    if type(first) == list:
        return first + second
    else:
        if axis == 0:
            if verbose > 0:
                print("Vertical stacking")
            return np.vstack((first, second))
        else:
            if verbose > 0:
                print("Horizontal stacking")
            return np.hstack((first, second))


def shuffle_dataset(a):
    p = np.random.permutation(len(a))
    if type(a) == list:
        return [a[i] for i in p]
    else:
        return a[p]


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))

    if type(a) == list or type(b) == list:
        return [a[i] for i in p], [b[i] for i in p]
    else:
        return a[p], b[p]


def dataset_vocab(data):
    present_vocab = list(set().union(*data))
    present_vocab_count = len(present_vocab)
    vocab_size = np.max(present_vocab) + 1
    return present_vocab, present_vocab_count, vocab_size


def dataset_longest_seq(data):
    max_len_seq = max(data, key=len)
    return len(max_len_seq), max_len_seq


def to_onehot(data, vocab_size):
    return to_categorical(data, num_classes=vocab_size)


def pad_dataset(data, max_length, padding):
    return list(pad_sequences(data, maxlen=max_length, padding=padding))


def generate_windows_from_dataset(data, window_size, padding):
    #Input:
    # data  = [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6]]
    # ws = get_windows_from_dataset(data, 4, 'pre')
    # display(data)
    # display(ws)
    #Output:
    # [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6]]
    #
    # [array([0, 0, 0, 1]),
    #  array([0, 0, 1, 2]),
    #  ...
    #  array([2, 3, 4, 5]),
    #  array([0, 0, 0, 1]),
    #  array([0, 0, 1, 2]),
    #  ...
    #  array([3, 4, 5, 6])]
    prefixes = get_dataset_prefixes(data)
    return pad_dataset(prefixes, window_size, padding)

def get_dataset_prefixes(sequences):
    #Input:
    # data  = [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6]]
    # prefixes = get_dataset_prefixes(data)
    # display(data)
    # display(prefixes)
    #Output:
    # [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6]]
    #
    # [[1],
    #  [1, 2],
    #  ...
    #  [1, 2, 3, 4, 5],
    #  [1],
    #  [1, 2],
    #  ...
    #  [1, 2, 3, 4, 5, 6]]
    return list(chain.from_iterable([get_seq_prefixes(seq) for seq in sequences]))


def get_seq_prefixes(seq):
    #Input:
    # data  = [1, 2, 3, 4, 5]
    # p = get_seq_prefixes(data)
    # display(data)
    # display(p)
    #Output:
    # [1, 2, 3, 4, 5]
    # [[1], [1, 2], [1, 2, 3], [1, 2, 3, 4], [1, 2, 3, 4, 5]]
    return [seq[:i+1] for i in range(0, len(seq))]


def shift_windows(data):
    #Ex1:
    #Input:
    # data = [[1, 2, 3, 4],[5, 6, 7, 8], [9, 10, 11, 12]]
    # X, Y = shift_windows(data)
    # display(X)
    # display(Y)
    #Output
    # X= [[1, 2, 3, 4], [5, 6, 7, 8]]
    # Y= [[5, 6, 7, 8], [9, 10, 11, 12]]
    #
    #Ex2:
    #Input:
    # data = get_windows_from_dataset([[1, 2, 3]], 4, 'pre')
    # #data: => [[0, 0, 0, 1], [0, 0, 1, 2], [0, 1, 2, 3]]
    # X, Y = shift_windows(data)
    # display(X)
    # display(Y)
    #Output
    # X= [[0, 0, 0, 1], [0, 0, 1, 2]]
    # Y= [[0, 0, 1, 2], [0, 1, 2, 3]]
    return data[:len(data)-1], data[1:len(data)]


def combine_datasets(datasets):
    #Input:
    # [[1,2], [3,4,5]]
    #Output:
    #[1,2,3,4,5]
    return list(chain.from_iterable(datasets))


def load_file(original_path, to_read=0, shuffle=False, _dtype=None, max_len = 0):
    # Reads a file from path, shuffles if needed,
    # Filters out max_len seqs (if max_len = 0, then reads all)
    # Takes a sample of size to_read (if to_read = 0, then uses all of above) 
    # Applies a split to each file line
    # Return a list of arrays of _dtype, one array for each file line
    with open(str(original_path), 'r') as f:
        sample = list(f)

        if shuffle:
            random.shuffle(sample)
        

        sample = [np.array(s.split(), dtype=_dtype) for s in sample]

        if max_len > 0:
            sample = [x for x in sample if len(x) <= max_len]

        if to_read <= 0:
            to_read = len(sample)

        sample = sample[0: to_read]

    return sample


def load_multiple_files(files_dict, shuffle=False, _dtype=int, max_len=0,  **path_params):
    #Input:
    #files_dict = {
    # 'data1' : {
    #   'fullpath': 'hola.txt', 'to_read': 20
    # },
    # 'data2' : {
    #   'fullpath': 'chau.txt', 'to_read': 10
    # }
    #}
    #**path_params= exp_name='exp_1', epsilon=2, iteration=1...
    #Output:
    #[[1,2,3], [1,2]...]

    all_data = []
    for dataset in files_dict.values():
        print(dataset)
        path = dataset["fullpath"].format(**path_params)
        data = load_file(path, to_read=dataset["to_read"], shuffle=shuffle, _dtype=_dtype, max_len=max_len)
        all_data.append(data)
    all_data = combine_datasets(all_data)
    if shuffle:
        random.shuffle(all_data)
    return all_data


def load_multiple_files_with_class(files_dict, shuffle=False, _dtype=int, max_len=0, **path_params):
    #Input:
    #files_dict = {
    # 'data1' : {
    #   'fullpath': 'hola.txt', 'to_read': 20, 'class': 0
    # },
    # 'data2' : {
    #   'fullpath': 'chau.txt', 'to_read': 10, 'class': 1
    # }
    #}
    #**path_params= exp_name='exp_1', epsilon=2, iteration=1...
    #Output:
    #[[1,2,3], [1,2]...], [0, 1,...]
    all_x = []
    all_y = []
    for dataset in files_dict.values():
        path = dataset["fullpath"].format(**path_params)
        x = load_file(path, to_read=dataset["to_read"], shuffle=shuffle, _dtype=_dtype, max_len=max_len)
        y = [dataset['class']]*len(x)
        all_x.append(x)
        all_y.append(y)

    all_x = combine_datasets(all_x)
    all_y = combine_datasets(all_y)

    if shuffle:
        all_x, all_y = unison_shuffled_copies(all_x, all_y)
    
    return all_x, all_y

def write_file(dataset, filename_out):
    file = open(str(filename_out), 'w')
    for element in dataset:
        file.write(' '.join(map(str, np.array(element))))
        file.write('\n')
    file.close()
