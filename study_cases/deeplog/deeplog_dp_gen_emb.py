
from keras.layers import Embedding, LSTM, Dense, TimeDistributed, Flatten, Lambda
from keras.models import Sequential, load_model
from keras.optimizers import Adam

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.special import softmax

import plot_utils
from data_utils import load_dataset_for_cbow, shuffle_dataset, unison_shuffled_copies, load_test, write_file
from data_utils import stack_datasets, generate_batch_for_cbow, dataset_vocab, dataset_longest_seq, to_onehot

class DeepLogDPGen:

    def __init__(self, experiment, iteration, datasets_params, model_fullpath, train_params, pre_proba_matrix_fullpath, generation_params, to_privatize_output_fullpath):
        self.datasets_params = datasets_params
        self.model_fullpath = model_fullpath.format(exp_name=experiment)
        self.train_params = train_params
        self.pre_proba_matrix_fullpath = pre_proba_matrix_fullpath.format(exp_name=experiment)
        self.epsilon, _ = generation_params.values()
        self.to_privatize_output_fullpath = to_privatize_output_fullpath.format(exp_name=experiment)
        self._get_model()
        self._get_pre_proba_matrix()
        self._load_data_to_privatize()

    def _get_model(self):
        try:
            self.model = load_model(self.model_fullpath)
        except:
            print("\nModel", self.model_fullpath, "not found. Training started...")
            self.model = self._train_model(*self.train_params.values())

        self.embedding = self.model.layers[0].get_weights()[0]
        self.vocab_size = self.embedding.shape[0]

    def _train_model(self, context_size, emb_size, vocab_size, train_sessions):
        train = np.array([])
        for dataset_fullpath, to_read in self.datasets_params['train']:
            dataset = load_dataset_for_cbow(dataset_fullpath, to_read)
            train = stack_datasets(train, dataset, 1)
        train = shuffle_dataset(train)
        
        x_batch, y_batch = generate_batch_for_cbow(train, window_size = int(context_size*0.5))
        y_batch_oh = to_onehot(y_batch, vocab_size)

        model = Sequential()
        model.add(Embedding(input_dim=vocab_size, output_dim=emb_size, input_length=context_size, mask_zero=False))
        model.add(Flatten())
        model.add(Dense(vocab_size, activation='softmax', kernel_regularizer='l2'))

        for key in train_sessions.keys():
            print("\nTrain Session:", key)
            session_info = train_sessions[key]
            epochs, batch_size, lr = session_info.values()

            model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=lr))

            history = model.fit(x_batch, y_batch_oh, epochs = epochs, batch_size = batch_size, validation_split=0.2, verbose=1)
            #plot_utils.plot(history)
        
        model.save(self.model_fullpath)
        return model

    def _get_pre_proba_matrix(self):
        try:
            self.pre_proba_matrix = np.load(self.pre_proba_matrix_fullpath, allow_pickle=True)
        except:
            print("\nProba Matrix", self.pre_proba_matrix_fullpath, "not found. Training started...")
            self.pre_proba_matrix = self._compute_pre_proba_matrix()
    
    def _compute_pre_proba_matrix(self):
   
        u_matrix = np.zeros(shape=(self.vocab_size,self.vocab_size))
        for i in range(0, self.vocab_size):
            for j in range(i, self.vocab_size):
                u_matrix[i][j] = cosine_similarity(self.embedding[i].reshape(1,-1), self.embedding[j].reshape(1,-1))[0][0]
                u_matrix[j][i] = u_matrix[i][j]   

        values = []
        for i in range(0, self.vocab_size):
            for j in range(i, self.vocab_size):
                values.append(abs(u_matrix[i] - u_matrix[j]))
        delta_u = np.max(values)

        pre_proba_matrix = np.zeros(shape=(self.vocab_size, self.vocab_size))
        for i in range(0, self.vocab_size):
            for j in range(i, self.vocab_size):
                pre_proba_matrix[i][j] = (u_matrix[i][j]*0.5)/delta_u

        np.save(self.pre_proba_matrix_fullpath, pre_proba_matrix, allow_pickle=True)
        return pre_proba_matrix

    def _load_data_to_privatize(self):
        self.datasets_to_privatize = {}
        for dataset_name, dataset_fullpath, to_read in self.datasets_params['to_privatize']:
            dataset = np.array(load_test(dataset_fullpath, to_read))
            self.datasets_to_privatize[dataset_name] = shuffle_dataset(dataset)

    def generate(self, iteration):
        for dataset_name in self.datasets_to_privatize.keys():
            self._generate_synthetic(iteration, dataset_name, self.datasets_to_privatize[dataset_name])

    def _generate_synthetic(self, iteration, dataset_name, dataset):
        fake_data = []
        for seq in dataset:
            private_seq = []
            for i in seq:
                #print("Proba:", i, "-", pre_proba_matrix[i], "\n")
                #softmax_divider = np.sum(np.exp(self.epsilon  * self.pre_proba_matrix[i]))
                #proba_vector = np.exp(self.epsilon  * self.pre_proba_matrix[i]) / softmax_divider
                proba_vector = softmax(self.epsilon * self.pre_proba_matrix[i])
                private_symbol = np.random.choice(np.arange(0, self.vocab_size), p=proba_vector)
                private_seq.append(private_symbol)
            fake_data.append(private_seq)
            #print("\nOriginal:", seq, "\nPrivate:", np.array(private_seq))

        filename_output = self.to_privatize_output_fullpath.format(to_privatize_name = dataset_name, iteration = iteration)   
        write_file(fake_data, filename_output)
