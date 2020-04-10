from keras.models import load_model

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.special import softmax

import plot_utils
import data_utils
import study_cases.deeplog.models as models
import study_cases.deeplog.deeplog_data_utils as d_utils
from nn_trainer import NNTrainer


class DeepLogDPGen:

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
        self._get_pre_proba_matrix()
        self._load_data_to_privatize()

    def _get_model(self):
        try:
            self.model = load_model(self.network_fullpath)
        except:
            print("\nModel", self.network_fullpath,
                  "not found. Training started...")
            self.model = self._train_model(*self.network_params.values())

        self.embedding = self.model.layers[0].get_weights()[0]
        self.vocab_size = self.embedding.shape[0]

    def _train_model(self, model_type, vocab_size, emb_size, context_size, train_sessions):

        t_sets = self.datasets_params['train']
        train = np.array([])
        for dataset_name, t_set in t_sets.items():
            path = t_set["fullpath"].format(exp_name=self.exp_name)
            temp = d_utils.load_dataset_for_cbow(path, amount_to_load=t_set["to_read"])
            train = data_utils.stack_datasets(train, temp, 1)
        train = data_utils.shuffle_dataset(train)

        train_x, train_y = d_utils.generate_batch_for_cbow(train, window_size=int(context_size*0.5))
        train_y_oh = data_utils.to_onehot(train_y, vocab_size)

        trainer = NNTrainer()
        model = models.create_model(
            model_type, [vocab_size, emb_size, context_size])
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
        for dataset_name in t_sets:
            t_set = t_sets[dataset_name]
            path = t_set["fullpath"].format(exp_name=self.exp_name)
            self.datasets_to_privatize[dataset_name] = data_utils.load_file(
                path, to_read=t_set["to_read"], _dtype=int, shuffle=False)

    def generate(self, epsilon, iteration):
        for dataset_name, dataset in self.datasets_to_privatize.items():
            self._generate_synthetic(epsilon, iteration, dataset_name, dataset)

    def _generate_synthetic(self, epsilon, iteration, dataset_name, dataset):
        fake_data = []
        for seq in dataset:
            private_seq = []
            for i in seq:
                #print("Proba:", i, "-", pre_proba_matrix[i], "\n")
                #softmax_divider = np.sum(np.exp(self.epsilon  * self.pre_proba_matrix[i]))
                #proba_vector = np.exp(self.epsilon  * self.pre_proba_matrix[i]) / softmax_divider
                proba_vector = softmax(epsilon * self.pre_proba_matrix[i])
                private_symbol = np.random.choice(
                    np.arange(0, self.vocab_size), p=proba_vector)
                private_seq.append(private_symbol)
            fake_data.append(private_seq)
            #print("\nOriginal:", seq, "\nPrivate:", np.array(private_seq))

        filename_fullpath = self.to_privatize_output_fullpath.format(
            to_privatize_name=dataset_name, epsilon=epsilon, iteration=iteration)
        data_utils.write_file(fake_data, filename_fullpath)
