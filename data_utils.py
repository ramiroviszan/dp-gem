
import random
import numpy as np

from keras.utils import to_categorical 

def stack_datasets(first, second, axis=0, verbose=0):
    if len(first) == 0:
        return second

    if len(second) == 0:
        return first

    if type(first) ==  list:
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
    present_vocab = np.array(list(set().union(*data)))
    present_vocab_count = len(present_vocab)
    vocab_size = np.max(present_vocab) + 1
    return present_vocab, present_vocab_count, vocab_size
    
def dataset_longest_seq(data):
    max_len_seq = max(data, key=len)
    return len(max_len_seq), max_len_seq

def to_onehot(data, vocab_size):
    return to_categorical(data, num_classes=vocab_size)

def load_file(original_path, to_read = 0, _dtype=None, shuffle=True):
    with open(str(original_path), 'r') as f:
        sample = list(f)
        if to_read <= 0:
            to_read = len(sample)

        if shuffle:
            if to_read > 0:
                sample = random.sample(sample, to_read)  
            else:
                random.shuffle(sample)
        else:
            sample = sample[0: to_read]  

    splits = [np.array(s.split(), dtype=_dtype) for s in sample]    
    return splits
    
def write_file(dataset, filename_out):
    file = open(str(filename_out), 'w')
    for element in dataset:
        file.write(' '.join(map(str, np.array(element))))
        file.write('\n')
    file.close()
