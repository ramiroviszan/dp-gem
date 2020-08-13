
import random
import numpy as np

from itertools import chain
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences


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


def generate_windows_from_dataset(data, window_size, remove_shorter=False, padding='pre'):
    """_
    #Input:
    \ndata  = [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6]]
    \nws = get_windows_from_dataset(data, 4, 'pre')
    
    #Output:
    \ndata = [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6]]
    \nws = [array([0, 0, 0, 1]),
     array([0, 0, 1, 2]),
     ...
     array([2, 3, 4, 5]),
     array([0, 0, 0, 1]),
     array([0, 0, 1, 2]),
     ...
     array([3, 4, 5, 6])]"""
    prefixes = get_dataset_prefixes(data, window_size, remove_shorter)
    return pad_dataset(prefixes, window_size, padding)

def get_dataset_prefixes(sequences, window_size, remove_shorter):
    """_
    #Input:
    \ndata  = [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6]]
    \nprefixes = get_dataset_prefixes(data)
    
    #Output:
    \ndata = [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6]]
    \nprefixes = [[1],
     [1, 2],
     ...
     [1, 2, 3, 4, 5],
     [1],
     [1, 2],
     ...
     [1, 2, 3, 4, 5, 6]]"""
    result = list(chain.from_iterable([get_seq_prefixes(seq) for seq in sequences])) 
    if remove_shorter:
        result = [seq for seq in result if len(seq) >= window_size]
    return result


def get_seq_prefixes(seq):
    """_
    #Input:
    \ndata  = [1, 2, 3, 4, 5]
    \np = get_seq_prefixes(data)
    #Output:
    \ndata = [1, 2, 3, 4, 5]
    \np = [[1], [1, 2], [1, 2, 3], [1, 2, 3, 4], [1, 2, 3, 4, 5]]"""
    return [seq[:i+1] for i in range(0, len(seq))]


def shift_windows(data):
    """_
    \n### Ex1:
    #Input:
    \ndata = [[1, 2, 3, 4],[5, 6, 7, 8], [9, 10, 11, 12]]
    \nX, Y = shift_windows(data)
    #Output:
    \nX= [[1, 2, 3, 4], [5, 6, 7, 8]]
    \nY= [[5, 6, 7, 8], [9, 10, 11, 12]]
    
    \n### Ex2:
    #Input:
    \ndata = get_windows_from_dataset([[1, 2, 3]], 4, 'pre')
    #data: => [[0, 0, 0, 1], [0, 0, 1, 2], [0, 1, 2, 3]]
    \nX, Y = shift_windows(data)
    #Output:
    \nX= [[0, 0, 0, 1], [0, 0, 1, 2]]
    \nY= [[0, 0, 1, 2], [0, 1, 2, 3]]"""
    return data[:len(data)-1], data[1:len(data)]


def combine_datasets(datasets):
    """_
    #Input:
    \n[[1,2], [3,4,5]]
    #Output:
    \n[1,2,3,4,5]"""
    return list(chain.from_iterable(datasets))


def load_file(fullpath, to_read=0, shuffle=False, max_len=0, dtype=None, split_token='', encoding='utf-8', errors='strict'):
    """_
    #Reads a file from path, shuffles if needed,
    \nFilters out max_len seqs (if max_len = 0, then reads all)
    \nTakes a sample of size to_read (if to_read = 0, then uses all of above) 
    \nApplies a split to each file line
    #Returns a list of arrays of dtype, one array for each file line if split_token is != None
    #Returns a list of strings one for each file line otherwise"""
    with open(str(fullpath), 'r', encoding=encoding, errors=errors) as f:
        sample = list(f)

        if shuffle:
            random.shuffle(sample)
        
        if split_token != None:
            if split_token == '':
                sample = [np.array(s.split(), dtype=dtype) for s in sample]
            else:
                sample = [np.array(s.split(split_token), dtype=dtype) for s in sample] 

        if max_len > 0:
            sample = [x for x in sample if len(x) <= max_len]

        if to_read <= 0:
            to_read = len(sample)

        sample = sample[0: to_read]

    return sample


def load_multiple_files(files_dict, shuffle=False, max_len=0, dtype=None, split_token='', encoding='ascii', errors='strict', **path_params):
    """_
    #Input:
    \nfiles_dict = {
        'data1' : {
            'fullpath': 'hola.txt', 'to_read': 20
        },
        'data2' : {
            'fullpath': 'chau.txt', 'to_read': 10
        }
    },
    \n**path_params= exp_name='exp_1', epsilon=2, iteration=1...
    #Output:
    \n[[1,2,3], [1,2]...]"""

    all_data = []
    for dataset in files_dict.values():
        print(dataset)
        path = dataset["fullpath"].format(**path_params)
        data = load_file(path, to_read=dataset["to_read"], shuffle=shuffle, max_len=max_len, dtype=dtype, split_token=split_token, encoding=encoding, errors=errors)
        all_data.append(data)
    all_data = combine_datasets(all_data)
    if shuffle:
        random.shuffle(all_data)
    return all_data


def load_multiple_files_with_class(files_dict, shuffle=False, max_len=0, dtype=None, split_token='', encoding='ascii', errors='strict', **path_params):
    """_
    #Input:
    \nfiles_dict = {
        'data1' : {
            'fullpath': 'hola.txt', 'to_read': 20, 'class': 0
        },
        'data2' : {
            'fullpath': 'chau.txt', 'to_read': 10, 'class': 1
        }
    },
    \n**path_params= exp_name='exp_1', epsilon=2, iteration=1...
    #Output:
    \n[[1,2,3], [1,2]...], [0, 1,...]"""
    all_x = []
    all_y = []
    for dataset in files_dict.values():
        path = dataset["fullpath"].format(**path_params)
        x = load_file(path, to_read=dataset["to_read"], shuffle=shuffle, max_len=max_len, dtype=dtype, split_token=split_token, encoding=encoding, errors=errors)
        y = [dataset['class']]*len(x)
        all_x.append(x)
        all_y.append(y)

    all_x = combine_datasets(all_x)
    all_y = combine_datasets(all_y)

    if shuffle:
        all_x, all_y = unison_shuffled_copies(all_x, all_y)
    
    return all_x, all_y

def write_file(dataset, filename_out, split_token=' ', encoding='utf-8', errors='strict'):
    with open(str(filename_out), 'w', encoding=encoding, errors=errors) as file:
        for element in dataset:
            file.write(split_token.join(map(str, np.array(element))))
            file.write('\n')
