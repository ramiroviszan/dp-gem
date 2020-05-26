from keras.models import load_model

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.special import softmax

import common.plot_utils as plot_utils
import common.data_utils as data_utils
import study_cases.deeplog.models as models
import study_cases.deeplog.deeplog_data_utils as d_utils
from common.nn_trainer import NNTrainer

class DPGen:

    def __init__(self, experiment, datasets_params, network_fullpath, network_params, pre_proba_matrix_fullpath, to_privatize_output_fullpath):
        self.exp_name = experiment

        self.datasets_params = datasets_params
        self.network_fullpath = network_fullpath.format(exp_name=self.exp_name)
        self.network_params = network_params

        self.pre_proba_matrix_fullpath = pre_proba_matrix_fullpath.format(
            exp_name=self.exp_name)
        self.to_privatize_output_fullpath = to_privatize_output_fullpath.format(
            exp_name=self.exp_name)

        self._get_model()
        #self._get_pre_proba_matrix()
        self._load_data_to_privatize()

    def _get_model(self):
        try:
            self.model = load_model(self.network_fullpath)
        except:
            print("\nModel", self.network_fullpath,
                  "not found. Training started...")
            self.model = self._train_model(*self.network_params.values())

        self.max_len = self.model.layers[0].input_shape[1]
        self.embedding = self.model.layers[0].get_weights()[0]
        self.vocab_size = self.embedding.shape[0]
        return



        print(max_len)
        all_val = data_utils.load_multiple_files(self.datasets_params['val'], shuffle=True, _dtype=int, max_len=self.max_len, exp_name=self.exp_name)
        val_x = np.array(data_utils.pad_dataset(all_val, self.max_len, 'pre'))
        #val_x = np.expand_dims(val_x, axis=2)
        probas  = self.model.predict(val_x)
        y_hat = np.argmax(probas, axis=2)
        import sys
        np.set_printoptions(threshold=sys.maxsize)
        for i, x in enumerate(val_x):
            print(x)
            print(y_hat[i])
            print("\n")
            #print(preds[i][len(x)-5:len(x)])
        print(np.mean(np.sum(y_hat == val_x, axis= 1))/50)
      
        #self.embedding = self.model.layers[0].get_weights()[0]
        #self.vocab_size = self.embedding.shape[0]

    def _train_model(self, model_type, vocab_size, window_size, train_sessions):

        all_data = data_utils.load_multiple_files(self.datasets_params['train'], shuffle=True, _dtype=int, max_len=window_size, exp_name=self.exp_name)
        
        if window_size == 0:
            max_len, _ = data_utils.dataset_longest_seq(all_data)
            window_size = max_len

        train_x = np.array(data_utils.pad_dataset(all_data, window_size, 'pre'))
        #train_x = data_utils.pad_dataset(all_data, window_size, 'pre')
        train_y_oh = data_utils.to_onehot(train_x, vocab_size)
        #train_x = np.expand_dims(train_x, axis=2)

        model = models.create_model(model_type, [window_size, vocab_size, int(vocab_size**(1/4))])
        #model = models.create_model(model_type, [window_size, vocab_size])

        trainer = NNTrainer()
        model = trainer.train(model, self.network_fullpath, train_x, train_y_oh, train_sessions)

        return model

    def _get_pre_proba_matrix(self):
        try:
            self.pre_proba_matrix = np.load(
                self.pre_proba_matrix_fullpath, allow_pickle=True)
        except:
            print("\nProba Matrix", self.pre_proba_matrix_fullpath,
                  "not found. Training started...")
            self.pre_proba_matrix = self._compute_pre_proba_matrix()

    def _compute_pre_proba_matrix(self):

        u_matrix = np.zeros(shape=(self.vocab_size, self.vocab_size))
        for i in range(0, self.vocab_size):
            for j in range(i, self.vocab_size):
                u_matrix[i][j] = cosine_similarity(self.embedding[i].reshape(
                    1, -1), self.embedding[j].reshape(1, -1))[0][0]
                u_matrix[j][i] = u_matrix[i][j]

        values = []
        for i in range(0, self.vocab_size):
            for j in range(i, self.vocab_size):
                values.append(abs(u_matrix[i] - u_matrix[j]))
        delta_u = np.max(values)
        print('delta_U:', delta_u)

        pre_proba_matrix = np.zeros(shape=(self.vocab_size, self.vocab_size))
        for i in range(0, self.vocab_size):
            for j in range(i, self.vocab_size):
                pre_proba_matrix[i][j] = (u_matrix[i][j]*0.5)/delta_u

        np.save(self.pre_proba_matrix_fullpath,
                pre_proba_matrix, allow_pickle=True)
        return pre_proba_matrix

    def _load_data_to_privatize(self):
        t_sets = self.datasets_params['to_privatize']
        self.datasets_to_privatize = {}
        for dataset_name, dataset in t_sets.items():
            path = dataset["fullpath"].format(exp_name=self.exp_name)
            self.datasets_to_privatize[dataset_name] = data_utils.load_file(
                path, to_read=dataset["to_read"], shuffle=False, _dtype=int, max_len=self.max_len)

    def generate(self, epsilon, iteration):
        for dataset_name, dataset in self.datasets_to_privatize.items():
            print('\n\nGenerating dataset:', dataset_name, '- Num seqs:', len(dataset))
            self._generate_synthetic(epsilon, iteration, dataset_name, dataset)

    def _generate_synthetic(self, epsilon, iteration, dataset_name, dataset):
        seq_x = np.array(data_utils.pad_dataset(dataset, self.max_len, 'pre'))
        probas = self.model.predict(seq_x) * epsilon * 0.5

        fake_data = []
        for seq_i, seq in enumerate(seq_x):
            private_seq = []
            last_index = len(seq) - 1
            padding = 0
            for index, real_symbol in enumerate(seq):
                if real_symbol != 0:#do not include padding
                    if index == last_index:#do not privatize end token
                        private_symbol = real_symbol
                    else:
                        #print("Proba:", i, "-", pre_proba_matrix[i], "\n")
                        proba_vector = softmax(probas[seq_i][index])
                        private_symbol = np.random.choice(np.arange(0, self.vocab_size), p=proba_vector)
                    private_seq.append(private_symbol)
            fake_data.append(private_seq)
            #print("\nOriginal:", seq, "\nPrivate:", np.array(private_seq))

        filename_fullpath = self.to_privatize_output_fullpath.format(
            to_privatize_name=dataset_name, epsilon=epsilon, iteration=iteration)
        data_utils.write_file(fake_data, filename_fullpath)