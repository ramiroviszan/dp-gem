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

    def __init__(self, experiment, datasets_params, network_fullpath, network_params, to_privatize_output_fullpath):
        self.exp_name = experiment

        self.datasets_params = datasets_params
        self.network_fullpath = network_fullpath.format(exp_name=self.exp_name)
        self.network_params = network_params

        #self.pre_proba_matrix_fullpath = pre_proba_matrix_fullpath.format(
        #    exp_name=self.exp_name)
        self.to_privatize_output_fullpath = to_privatize_output_fullpath.format(
            exp_name=self.exp_name)

        self._get_model()
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

    def _train_model(self, model_type, vocab_size, window_size, emb_size, train_sessions):

        all_data = data_utils.load_multiple_files(self.datasets_params['train'], shuffle=True, dtype=int, max_len=window_size, exp_name=self.exp_name)
        
        if window_size == 0:
            max_len, _ = data_utils.dataset_longest_seq(all_data)
            window_size = max_len

        train_x = np.array(data_utils.pad_dataset(all_data, window_size, 'pre'))
        train_y_oh = data_utils.to_onehot(train_x, vocab_size)

        model = models.create_model(model_type, [window_size, vocab_size, emb_size])

        trainer = NNTrainer()
        model = trainer.train(model, self.network_fullpath, train_x, train_y_oh, train_sessions)

        return model

    def _load_data_to_privatize(self):
        t_sets = self.datasets_params['to_privatize']
        self.datasets_to_privatize = {}
        for dataset_name, dataset in t_sets.items():
            path = dataset["fullpath"].format(exp_name=self.exp_name)
            self.datasets_to_privatize[dataset_name] = data_utils.load_file(
                path, to_read=dataset["to_read"], shuffle=False, max_len=self.max_len, dtype=int, split_token='')

    def generate(self, trial, iteration):
        for dataset_name, dataset in self.datasets_to_privatize.items():
            print('\n\nGenerating dataset:', dataset_name, '- Num seqs:', len(dataset))
            if trial['eps'] == 'no_dp':
                self._generate_synthetic_no_dp(trial, iteration, dataset_name, dataset)
            else:
                self._generate_synthetic(trial, iteration, dataset_name, dataset)

    def _generate_synthetic(self, trial, iteration, dataset_name, dataset):
        seq_x = np.array(data_utils.pad_dataset(dataset, self.max_len, 'pre'))
        lens = np.array([len(seq) for seq in dataset])

        epsilon = trial.get('eps', False)
        maxdelta = trial.get('maxdelta', 1)
        variable_eps = trial.get('variable_eps', False)
        if not variable_eps:
            scale =  epsilon / (2 * maxdelta)
        else:
            epsilons = (lens*epsilon)/self.max_len
            scale = epsilons / (2 * maxdelta)
            scale = scale[:, np.newaxis, np.newaxis] #multiply each symbol proba for each position for each sequence by the scale


        probas = self.model.predict(seq_x) * scale
      
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
            to_privatize_name=dataset_name, iteration=iteration, **trial)
        data_utils.write_file(fake_data, filename_fullpath)

    def _generate_synthetic_no_dp(self, trial, iteration, dataset_name, dataset):
        seq_x = np.array(data_utils.pad_dataset(dataset, self.max_len, 'pre'))
        probas = self.model.predict(seq_x) 

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
                        private_symbol = np.argmax(probas[seq_i][index])
                    private_seq.append(private_symbol)
            fake_data.append(private_seq)

        filename_fullpath = self.to_privatize_output_fullpath.format(
            to_privatize_name=dataset_name, iteration=iteration, **trial)
        data_utils.write_file(fake_data, filename_fullpath)