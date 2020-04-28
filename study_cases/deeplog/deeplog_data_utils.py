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
                line = np.array(line.split(), dtype=int)
                for i in range(len(line) - window_size + 1):
                    seq = line[i:i + window_size]
                    if len(seq) != 0:
                        seqs.append(seq)
                        #print(num_sessions, i, '\n', seq, '\n', line, '\n')
    print('Number of seqs({}): {}'.format(filename, len(seqs)))
    print('CBOW', seqs, np.array(seqs), np.array(seqs).shape)
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

def generate_batch_for_cbow(data, context_size = 4):
    x_batch = []
    y_batch = []
    side_context = int(context_size / 2)
    for entry in data:
        #print(entry)
        entry = pad_sequences([entry], len(entry) + side_context, padding='pre')[0]
        entry = pad_sequences([entry], len(entry) + side_context, padding='post')[0]
        #print(entry)
        for index in range(0, len(entry)-context_size):
            x_before = entry[index:index+side_context] 
            x_after = entry[index+side_context+1:index+context_size+1]
            x_n = np.hstack((x_before, x_after))
            y = entry[index+side_context]
            x_batch.append(x_n)
            y_batch.append(y)
          #  print(x_n, y)
    return np.array(x_batch), np.array(y_batch)
    



