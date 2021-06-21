from tensorflow.keras.models import load_model
from tensorflow.python.distribute.mirrored_strategy import MirroredStrategy

#import wandb
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

    def __init__(self, experiment, datasets_params, network_fullpath, network_params, to_privatize_output_fullpath):
        self.exp_path, self.exp_path, self.parent_trial = experiment

        self.datasets_params = datasets_params
        self.network_fullpath = network_fullpath.format(exp_path=self.exp_path)
        self.network_params = network_params
        
        #self.logger = get_logger('gen', self.exp_name, self.parent_trial)
        #wandb.config.network_params = network_params
        #wandb.config.parent_trial = self.parent_trial

        self.to_privatize_output_fullpath = to_privatize_output_fullpath.format(
            exp_path=self.exp_path)
        
        #strategy = MirroredStrategy()
        #with strategy.scope():
        self._get_model()
        self._load_data_to_privatize()

    def _get_model(self):
        self.vocab_size = self.network_params['model_params']['vocab_size'] 
        self.vocab_range = np.arange(1, self.vocab_size-1)
        
        try:
            self.model = load_model(self.network_fullpath)
        except:
            print("\nModel", self.network_fullpath,
                  "not found. Training started...")
            self.model = self._train_model(*self.network_params.values())

        self.max_len = self.model.layers[0].input_shape[1]
        self.embedding = self.model.layers[0].get_weights()[0]
       
        return

    def _train_model(self, model_type, model_params, train_sessions):

        window_size = model_params.get('window_size', 0)
        all_data = data_utils.load_multiple_files(self.datasets_params['train'], shuffle=True, dtype=int, max_len=window_size, exp_path=self.exp_path)
        
        if window_size == 0:
            max_len, _ = data_utils.dataset_longest_seq(all_data)
            window_size = max_len
            model_params['window_size'] = window_size

        train_x = np.array(data_utils.pad_dataset(all_data, window_size, 'pre'))
        train_y_oh = data_utils.to_onehot(train_x, self.vocab_size)

        model = models.create_model(model_type, model_params.values())

        trainer = NNTrainer()
        model = trainer.train(model, self.network_fullpath, train_x, train_y_oh, train_sessions, use_wandb=False)

        return model

    def _load_data_to_privatize(self):
        t_sets = self.datasets_params['to_privatize']
        self.datasets_to_privatize = {}
        for dataset_name, dataset in t_sets.items():
            path = dataset["fullpath"].format(exp_path=self.exp_path)
            self.datasets_to_privatize[dataset_name] = data_utils.load_file(
                path, to_read=dataset["to_read"], shuffle=False, dtype=int, split_token='') #max_len=self.max_len, 

    def run(self, trial):
        for dataset_name, dataset in self.datasets_to_privatize.items():
            print('\n\nGenerating dataset:', dataset_name, '- Num seqs:', len(dataset))
            self._generate_synthetic(trial, dataset_name, dataset)

    def _generate_synthetic(self, trial, dataset_name, dataset):
        padding = 0

        #if dataset_name == "abnormal_test":
        #    print("Before pad", dataset[0])
        seq_x = np.array(data_utils.pad_dataset(dataset, self.max_len, 'pre'))
        
        #if dataset_name == "abnormal_test":
        #    print("After pad", seq_x[0])
        
        epsilon = trial.get('eps', 'no_dp')
        maxdelta = trial.get('maxdelta', 1)
        if epsilon == 'no_dp':
            scale = 1
        else:   
            scale =  epsilon / (2 * maxdelta)

        probas = self.model.predict(seq_x) * scale
      
        fake_data = []
        for seq_i, seq in enumerate(dataset):
            
            private_seq = []
            last_index = len(seq) - 1

            for index, real_symbol in enumerate(seq):
                if real_symbol != padding:#do not include padding
                    if index == last_index:#do not privatize end token
                        private_symbol = real_symbol
                    else:
                        #Sacar la probalilidad de generar padding y endtoken
                        proba_vector = softmax(probas[seq_i][index][1:-1])
                        private_symbol = np.random.choice(self.vocab_range, p=proba_vector)
                    private_seq.append(private_symbol)         
            fake_data.append(private_seq)
            #print("\nOriginal:", seq, "\nPrivate:", np.array(private_seq))
     
        filename_fullpath = self.to_privatize_output_fullpath.format(
            to_privatize_name=dataset_name, trial=flat_trial(trial))
        data_utils.write_file(fake_data, filename_fullpath)