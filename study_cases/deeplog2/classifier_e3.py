from tensorflow.keras.models import load_model
from tensorflow.python.distribute.mirrored_strategy import MirroredStrategy

import copy
#import wandb
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve


import common.plot_utils as plot_utils
import common.data_utils as data_utils
from common.nn_trainer import NNTrainer
from common.csv_result import CSVResult
from common.trials_utils import flat_trial
from common.logger_utils import get_logger

import study_cases.deeplog2.models as models


class Classifier:

    def __init__(self, experiment, datasets_params, network_fullpath, results_fullpath):
        self.exp_name, self.exp_path, self.parent_trial = experiment

        self.datasets_params = datasets_params
        self.network_fullpath = network_fullpath.format(exp_path=self.exp_path, parent_trial=flat_trial(self.parent_trial))
        

        result_header = list(self.parent_trial.keys())
        results_header = result_header + ['tn', 'fp', 'fn', 'tp', 'acc']
        test_results_fullpath = results_fullpath.format(exp_path=self.exp_path, dataset_type='test')
        self.test_results = CSVResult(test_results_fullpath, results_header)

        self._get_model()
        
    def _get_model(self):
        try:
            self.model = models.load_model_adapter(self.network_fullpath)
        except:
            print("\nModel", self.network_fullpath, "not found. Training started...")

    def run(self, trial):
        classifier_params = trial
        self._run(*classifier_params.values())

    def _run(self, eps, maxdelta, hidden_state_size):

        test_x, test_y = data_utils.load_multiple_files_with_class(self.datasets_params['test'], shuffle=False, dtype=int, exp_path=self.exp_path, parent_trial=flat_trial(self.parent_trial))
        max_len, _ = data_utils.dataset_longest_seq(test_x)
        seq_x = np.array(data_utils.pad_dataset(test_x, max_len, 'pre'))
        
        scale =  maxdelta/eps
        noise_x = np.random.laplace(0, scale, (len(test_x), hidden_state_size))
        
        y_hat = self.model.predict([seq_x, noise_x]) >= 0.50
        
        metrics = self._metrics(test_y, y_hat)
        results = list(self.parent_trial.values())
        self.test_results.save_results(results + [*metrics])
        
    def _metrics(self, y, y_hat): 
        tn, fp, fn, tp = confusion_matrix(y, y_hat).ravel()
        acc = accuracy_score(y, y_hat)
        return tn, fp, fn, tp, acc
        
        

    