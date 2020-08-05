from tensorflow.keras.models import load_model
from tensorflow.python.distribute.mirrored_strategy import MirroredStrategy


import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.special import softmax

import common.plot_utils as plot_utils
import common.data_utils as data_utils
from common.nn_trainer import NNTrainer

import study_cases.deeplog.models as models

class DPGen:

    def __init__(self, experiment, datasets_params, network_fullpath, network_params, to_privatize_output_fullpath):
        self.exp_name = experiment

        self.datasets_params = datasets_params
        self.network_fullpath = network_fullpath.format(exp_name=self.exp_name)
        self.network_params = network_params

        self.to_privatize_output_fullpath = to_privatize_output_fullpath.format(
            exp_name=self.exp_name)
        
        #strategy = MirroredStrategy()
        #with strategy.scope():
        self._get_model()
        self._load_data_to_privatize()

    def _get_model(self):
        try:
            self.model = models.load_model_adapter(self.network_fullpath)
        except:
            print("\nModel", self.network_fullpath,
                  "not found. Training started...")
            self.model = self._train_model(*self.network_params.values())

        self.max_len = self.model.layers[0].input_shape[0][1]
        self.embedding = self.model.layers[1].get_weights()[0]
        self.vocab_size = self.embedding.shape[0]
        self.vocab_range = np.arange(1, self.vocab_size-1)
        self.hidden_state_size = self.network_params['hidden_state_size']
        return

    def _train_model(self, model_type, vocab_size, window_size, emb_size, hidden_state_size, train_sessions):

        all_data = data_utils.load_multiple_files(self.datasets_params['train'], shuffle=True, dtype=int, max_len=window_size, exp_name=self.exp_name)
        
        if window_size == 0:
            max_len, _ = data_utils.dataset_longest_seq(all_data)
            window_size = max_len

        train_x = np.array(data_utils.pad_dataset(all_data, window_size, 'pre'))
        noise_x = np.zeros(shape=(len(train_x), hidden_state_size))
        train_y_oh = data_utils.to_onehot(train_x, vocab_size)

        model = models.create_model(model_type, [window_size, vocab_size, emb_size, hidden_state_size])

        trainer = NNTrainer()
        model = trainer.train(model, self.network_fullpath, [train_x, noise_x], train_y_oh, train_sessions)

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
            self._generate_synthetic(trial, iteration, dataset_name, dataset)

    def _generate_synthetic(self, trial, iteration, dataset_name, dataset):
        padding = 0
        
        seq_x = np.array(data_utils.pad_dataset(dataset, self.max_len, 'pre'))
        lens = np.array([len(seq) for seq in dataset])

        epsilon = trial.get('eps', 'no_dp')
        if epsilon == 'no_dp':
            noise = np.zeros(shape=(len(dataset), self.hidden_state_size))
        else:
            variable_eps = trial.get('variable_eps', False)
            maxdelta = trial.get('maxdelta', 0)
            if not variable_eps:
                scale =  maxdelta/epsilon
            else:
                epsilons = epsilon/lens
                scale = maxdelta/epsilons
                #scale = scale[:, np.newaxis, np.newaxis] #multiply each symbol proba for each position for each sequence by the scale
            noise = np.random.laplace(0, scale, (len(dataset), self.hidden_state_size))

        probas = self.model.predict([seq_x, noise])

        fake_data = []
        for seq_i, seq in enumerate(seq_x):
            private_seq = []
            last_index = len(seq) - 1
            for index, real_symbol in enumerate(seq):
                if real_symbol != padding:#do not include padding
                    if index == last_index:#do not privatize end token
                        private_symbol = real_symbol
                    else:
                        #proba_vector = softmax(probas[seq_i][index][1:-1])
                        #private_symbol = np.random.choice(self.vocab_range, p=proba_vector)
                        proba_vector = softmax(probas[seq_i][index][1:-1])
                        private_symbol = np.argmax(proba_vector) + 1

                    private_seq.append(private_symbol)
            fake_data.append(private_seq)

        filename_fullpath = self.to_privatize_output_fullpath.format(
            to_privatize_name=dataset_name, iteration=iteration, **trial)
        data_utils.write_file(fake_data, filename_fullpath)