#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils, to_categorical 


# In[ ]:


data_path = ''


# In[ ]:


def stack_datasets(first, second, axis=0, verbose=0):
    if len(first) == 0:
        return second

    if len(second) == 0:
        return first

    if axis == 0:
        if verbose > 0:
            print("Vertical stacking")
        return np.vstack((first, second))
    else:
        if verbose > 0:
            print("Horizontal stacking")
        return np.hstack((first, second))


# In[ ]:


def shuffle_dataset(a):
    p = np.random.permutation(len(a))
    return a[p]


# In[ ]:


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


# In[ ]:


def dataset_vocab(data):
    present_vocab = np.array(list(set().union(*data)))
    present_vocab_count = len(present_vocab)
    vocab_size = np.max(present_vocab) + 1
    return present_vocab, present_vocab_count, vocab_size
    


# In[ ]:


def dataset_longest_seq(data):
    max_len_seq = max(data, key=len)
    return len(max_len_seq), max_len_seq


# In[ ]:


def to_onehot(data, vocab_size):
    return to_categorical(data, num_classes=vocab_size)


# In[ ]:


def load_dataset_for_cbow(filename, amount_to_load = -1, window_size=-1):
    num_sessions = 0
    seqs = []
    with open(data_path + str(filename),  'r') as f:
        for index, line in enumerate(f.readlines()):
            if amount_to_load > 0 and index == amount_to_load:
                break
            num_sessions += 1
            line = line.strip()
            if len(line) > 0: 
                line = np.array(line.split() + [1], dtype=np.int32) - 1
                for i in range(len(line) - window_size + 1):
                    seq = line[i:i + window_size]
                    if len(seq) != 0:
                        seqs.append(seq)
                        #print(num_sessions, i, '\n', seq, '\n', line, '\n')
    print('Number of seqs({}): {}'.format(filename, len(seqs)))
    return np.array(seqs)


# In[ ]:


def load_dataset_for_lm(filename, window_size=-1, amount_to_load = -1):
    num_sessions = 0
    seqs = []
    nexts = []
    with open(data_path + str(filename),  'r') as f:
        for index, line in enumerate(f.readlines()):
            if amount_to_load > 0 and index == amount_to_load:
                break
            num_sessions += 1
            line = line.strip()
            if len(line) > 0: 
                line = np.array(line.split()  + [1], dtype=np.int32) - 1
                for i in range(len(line) - window_size):
                    seq = line[i:i + window_size]
                    next_symbols = line[i+1:i + window_size + 1]
                    if len(seq) != 0:
                        seqs.append(seq)
                        nexts.append(next_symbols)
                        #print(num_sessions, i, '\n', seq, '\n',next_symbols,'\n',line, '\n')
    print('Number of seqs({}): {}'.format(filename, len(seqs)))
    return np.array(seqs), np.array(nexts)


# In[ ]:


def load_test(filename, amount_to_load = 0):
    dataset = []
    with open(data_path + str(filename),  'r') as f:
        for index, line in enumerate(f.readlines()):
            if amount_to_load > 0 and index == amount_to_load:
                break
            line = line.strip()
            if len(line) > 0:  
                line = np.array(line.split()  + [1], dtype=np.int32) - 1
                dataset.append(line)
    return dataset


# In[ ]:


def generate_batch_for_cbow(data, window_size = 2):
    x_batch = []
    y_batch = []
    for entry in data:
        for index, _ in enumerate(entry):
            filter_array = np.ones(len(entry), dtype=bool)
            filter_array[index] = False;
            start = index - window_size
            start = 0 if start < 0 else start
            end = index + window_size
            #print(start, end, len(entry))
            #print(entry)
            x = entry[filter_array]
            #print(x)
            x = x[start:end]
            #print(x)
            if len(x) <  end:
                x = pad_sequences([x], window_size*2, padding='post')[0]
            else:
                x = pad_sequences([x], window_size*2)[0]
            y = entry[index]
            x_batch.append(x)
            y_batch.append(y)
    return np.array(x_batch), np.array(y_batch)


# In[ ]:


def write_file(dataset, filename_out):
    file = open(data_path + str(filename_out),'w')
    for element in dataset:
        file.write(' '.join(map(str, np.array(element) + 1)))
        file.write('\n')
    file.close()

