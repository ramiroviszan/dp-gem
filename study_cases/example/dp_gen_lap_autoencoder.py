from tensorflow.keras.models import load_model
from tensorflow.python.distribute.mirrored_strategy import MirroredStrategy

import wandb
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.special import softmax

import common.plot_utils as plot_utils
import common.data_utils as data_utils
from common.nn_trainer import NNTrainer
from common.trials_utils import flat_trial
from common.logger_utils import get_logger

import study_cases.example.models as models

class Gen:

    def __init__(self, experiment, datasets_params, network_fullpath, network_params, to_privatize_output_fullpath):
        self.exp_name, self.exp_path, self.parent_trial = experiment
        
        self.datasets_params = datasets_params
        self.network_fullpath = network_fullpath.format(exp_path=self.exp_path)
        self.network_params = network_params

        self.logger = get_logger('gen', self.exp_name, self.parent_trial)
        wandb.config.network_params = network_params
        wandb.config.parent_trial = self.parent_trial
        
        
        self.to_privatize_output_fullpath = to_privatize_output_fullpath.format(
            exp_path=self.exp_path)
        
        #strategy = MirroredStrategy()
        #with strategy.scope():
        self._get_model()
        self._load_data_to_privatize()

    def _get_model(self):
        self.vocab_size = self.network_params['model_params']['vocab_size'] 
        self.vocab_range = np.arange(1, self.vocab_size-1)
        self.hidden_state_size = self.network_params['model_params']['hidden_state_size']
        
        try:
            self.model = models.load_model_adapter(self.network_fullpath)
        except:
            print("\nModel", self.network_fullpath,
                  "not found. Training started...")
            self.model = self._train_model(*self.network_params.values())

        self.max_len = self.model.layers[0].input_shape[0][1]

        return

    def _train_model(self, model_type, model_params, train_sessions):

        window_size = model_params.get('window_size', 0)
        all_data = data_utils.load_multiple_files(self.datasets_params['train'], shuffle=True, dtype=int, max_len=window_size, exp_path=self.exp_path)

        if window_size == 0:
            max_len, _ = data_utils.dataset_longest_seq(all_data)
            window_size = max_len

        train_x = np.array(data_utils.pad_dataset(all_data, window_size, 'pre'))
        noise_x = np.zeros(shape=(len(train_x), self.hidden_state_size))
        train_y_oh = data_utils.to_onehot(train_x, self.vocab_size)

        model = models.create_model(model_type, model_params.values())

        trainer = NNTrainer()
        model = trainer.train(model, self.network_fullpath, [train_x, noise_x], train_y_oh, train_sessions)

        return model

    def _load_data_to_privatize(self):
        t_sets = self.datasets_params['to_privatize']
        self.datasets_to_privatize = {}
        for dataset_name, dataset in t_sets.items():
            path = dataset["fullpath"].format(exp_path=self.exp_path)
            self.datasets_to_privatize[dataset_name] = data_utils.load_file(
                path, to_read=dataset["to_read"], shuffle=False, max_len=self.max_len, dtype=int, split_token='')

    def run(self, trial):
        for dataset_name, dataset in self.datasets_to_privatize.items():
            print('\n\nGenerating dataset:', dataset_name, '- Num seqs:', len(dataset))
            self._generate_synthetic(trial, dataset_name, dataset)

    def _generate_synthetic(self, trial, dataset_name, dataset):
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

        use_random_sampling = "trials_per_seq" in trial
        trials_per_seq = trial.get('trials_per_seq', 1)
        fake_data = []
        for seq_i, seq in enumerate(seq_x):
            for i in range(0, trials_per_seq):
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
                            if use_random_sampling:
                                private_symbol = np.random.choice(self.vocab_range, p=proba_vector)
                            else:
                                private_symbol = np.argmax(proba_vector) + 1 #+ 1 because padding is 0

                        private_seq.append(private_symbol)
                fake_data.append(private_seq)

        filename_fullpath = self.to_privatize_output_fullpath.format(to_privatize_name=dataset_name, trial=flat_trial(trial))
        data_utils.write_file(fake_data, filename_fullpath)