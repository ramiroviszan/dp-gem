import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical 

def load_dataset_for_cbow(filename, amount_to_load = -1, window_size=-1):
    num_sessions = 0
    seqs = []
    with open(str(filename),  'r') as f:
        for index, line in enumerate(f.readlines()):
            if amount_to_load > 0 and index == amount_to_load:
                break
            num_sessions += 1
            line = line.strip()
            if len(line) > 0: 
                #line = np.array(line.split() + [1], dtype=np.int32) - 1
                line = np.array(line.split(), dtype=int)
                for i in range(len(line) - window_size + 1):
                    seq = line[i:i + window_size]
                    if len(seq) != 0:
                        seqs.append(seq)
                        #print(num_sessions, i, '\n', seq, '\n', line, '\n')
    print('Number of seqs({}): {}'.format(filename, len(seqs)))
    return np.array(seqs)

def load_dataset_for_lm(filename, window_size=-1, amount_to_load = -1):
    num_sessions = 0
    seqs = []
    nexts = []
    with open(str(filename),  'r') as f:
        for index, line in enumerate(f.readlines()):
            if amount_to_load > 0 and index == amount_to_load:
                break
            num_sessions += 1
            line = line.strip()
            if len(line) > 0: 
                #line = np.array(line.split() + [1], dtype=np.int32) - 1
                line = np.array(line.split(), dtype=int)
                for i in range(len(line) - window_size):
                    seq = line[i:i + window_size]
                    next_symbols = line[i+1:i + window_size + 1]
                    if len(seq) != 0:
                        seqs.append(seq)
                        nexts.append(next_symbols)
                        #print(num_sessions, i, '\n', seq, '\n',next_symbols,'\n',line, '\n')
    print('Number of seqs({}): {}'.format(filename, len(seqs)))
    return np.array(seqs), np.array(nexts)

def load_file(filename, amount_to_load = 0):
    dataset = []
    with open(str(filename),  'r') as f:
        for index, line in enumerate(f.readlines()):
            if amount_to_load > 0 and index == amount_to_load:
                break
            line = line.strip()
            if len(line) > 0:  
                #line = np.array(line.split() + [1], dtype=np.int32) - 1
                line = np.array(line.split(), dtype=int)
                dataset.append(line)
    return dataset

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



