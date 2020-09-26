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

import study_cases.deeplog2.models as models

class Gen:

    def __init__(self, experiment, datasets_params, network_fullpath, network_params, pre_proba_matrix_fullpath, to_privatize_output_fullpath):
        self.exp_name, self.exp_path, self.parent_trial = experiment

        self.datasets_params = datasets_params
        self.network_fullpath = network_fullpath.format(exp_path=self.exp_path)
        self.network_params = network_params

        self.pre_proba_matrix_fullpath = pre_proba_matrix_fullpath.format(
            exp_path=self.exp_path)
        self.to_privatize_output_fullpath = to_privatize_output_fullpath.format(
            exp_path=self.exp_path)
        
        self.logger = get_logger('gen', self.exp_name, self.parent_trial)
        wandb.config.network_params = network_params
        wandb.config.parent_trial = self.parent_trial

        #strategy = MirroredStrategy()
        #with strategy.scope():
        self._get_model()

        self._get_pre_proba_matrix()
        self._load_data_to_privatize()

    def _get_model(self):
        self.vocab_size = self.network_params['model_params']['vocab_size'] 
        self.offset_padding = 1
        self.vocab_range = np.arange(self.offset_padding, self.vocab_size-1)
        self.vocab_size_out = len(self.vocab_range)
    
        try:
            self.model = load_model(self.network_fullpath)
        except:
            print("\nModel", self.network_fullpath,
                  "not found. Training started...")
            self.model = self._train_model(*self.network_params.values())

        self.embedding = self.model.layers[0].get_weights()[0]

    def _train_model(self, model_type, model_params, train_sessions):

        max_len = model_params.get('max_len', 0)
        train_x, train_y = data_utils.load_multiple_files_with_class(self.datasets_params['train'], shuffle=True, max_len=max_len, dtype=int, exp_path=self.exp_path)
        if max_len == 0:
            max_len, _ = data_utils.dataset_longest_seq(train_x)
            model_params['max_len'] = max_len

        train_x = np.array(data_utils.pad_dataset(train_x, max_len, 'pre'))
        train_y = np.array(train_y)

        model = models.create_model(model_type, model_params.values())
        trainer = NNTrainer()
        model = trainer.train(model, self.network_fullpath, train_x, train_y, train_sessions, use_wandb=True)
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
        u_matrix = np.zeros(shape=(self.vocab_size_out, self.vocab_size_out))
        for i in self.vocab_range:
            for j in self.vocab_range:
                u_matrix[i-self.offset_padding][j-self.offset_padding] = cosine_similarity(self.embedding[i].reshape(
                    1, -1), self.embedding[j].reshape(1, -1))[0][0]
                u_matrix[j-self.offset_padding][i-self.offset_padding] = u_matrix[i-self.offset_padding][j-self.offset_padding]

        values = []
        for i in range(0, self.vocab_size_out):
            for j in range(i, self.vocab_size_out):
                values.append(abs(u_matrix[i] - u_matrix[j]))
        delta_u = np.max(values)
        print('delta_U:', delta_u)

        pre_proba_matrix = np.zeros(shape=(self.vocab_size_out, self.vocab_size_out))
        for i in range(0, self.vocab_size_out):
            for j in range(i, self.vocab_size_out):
                pre_proba_matrix[i][j] = (u_matrix[i][j]*0.5)/delta_u

        np.save(self.pre_proba_matrix_fullpath,
                pre_proba_matrix, allow_pickle=True)
        return pre_proba_matrix

    def _load_data_to_privatize(self):
        t_sets = self.datasets_params['to_privatize']
        self.datasets_to_privatize = {}
        for dataset_name, dataset in t_sets.items():
            path = dataset["fullpath"].format(exp_path=self.exp_path)
            self.datasets_to_privatize[dataset_name] = data_utils.load_file(
                path, to_read=dataset["to_read"], shuffle=False, dtype=int, split_token='')

    def generate(self, trial, iteration):
        for dataset_name, dataset in self.datasets_to_privatize.items():
            print('\n\nGenerating dataset:', dataset_name, '- Num seqs:', len(dataset))
            self._generate_synthetic(trial, dataset_name, dataset)

    def _generate_synthetic(self, trial, dataset_name, dataset):
        padding = 0
        epsilon = trial.get('eps', 1)
        proba_matrix = softmax(epsilon * self.pre_proba_matrix, axis = 1)
        fake_data = []
        for seq_i, seq in enumerate(dataset):
            private_seq = []
            last_index = len(seq) - 1
            
            for index, real_symbol in enumerate(seq):
                 if real_symbol != padding:#do not include padding
                    if index == last_index:#do not privatize end token
                        private_symbol = real_symbol
                    else:
                        #print("Proba:", i, "-", pre_proba_matrix[i], "\n")
                        #seq_proba = self.model.predict(self.predict([seq[:index]]))
                        #proba_vector = softmax(epsilon * self.pre_proba_matrix[real_symbol])
                        private_symbol = np.random.choice(self.vocab_range, p=proba_matrix[real_symbol-self.offset_padding])
                    private_seq.append(private_symbol)
            fake_data.append(private_seq)
            #print("\nOriginal:", seq, "\nPrivate:", np.array(private_seq))

        filename_fullpath = self.to_privatize_output_fullpath.format(to_privatize_name=dataset_name, trial=flat_trial(trial))
        data_utils.write_file(fake_data, filename_fullpath)
