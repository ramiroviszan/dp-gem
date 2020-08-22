from tensorflow.keras.models import load_model
from tensorflow.python.distribute.mirrored_strategy import MirroredStrategy

import wandb
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve


import common.plot_utils as plot_utils
import common.data_utils as data_utils
from common.nn_trainer import NNTrainer
from common.csv_result import CSVResult
from common.trials_utils import flat_trial

import study_cases.example.models as models


class Classifier:

    def __init__(self, experiment, logger, datasets_params, network_fullpath, network_params, results_fullpath):
        self.exp_name, self.parent_trial = experiment
        self.logger = logger

        self.datasets_params = datasets_params
        self.network_fullpath = network_fullpath.format(exp_name=self.exp_name, parent_trial=flat_trial(self.parent_trial))
        self.network_params = network_params
        wandb.config.network_params = network_params
        wandb.config.parent_trial = self.parent_trial


        result_header = list(self.parent_trial.keys())
        results_header = result_header + ['threshold', 'tn', 'fp', 'fn', 'tp', 'acc']
        val_results_fullpath = results_fullpath.format(exp_name=self.exp_name, dataset_type='val')
        test_results_fullpath = results_fullpath.format(exp_name=self.exp_name, dataset_type='test')
        self.val_results = CSVResult(val_results_fullpath, results_header)
        self.test_results = CSVResult(test_results_fullpath, results_header)

        #strategy = MirroredStrategy()
        #with strategy.scope():
        self._get_model()
        
    def _get_model(self):
        try:
            self.model = models.load_model_adapter(self.network_fullpath)
        except:
            print("\nModel", self.network_fullpath, "not found. Training started...")
            self.model = self._train_model(*self.network_params.values())

    def _train_model(self, model_type, model_params, train_sessions):
        
        all_data = data_utils.load_multiple_files(self.datasets_params['train'], shuffle=True, dtype=int, exp_name=self.exp_name, parent_trial=flat_trial(self.parent_trial))

        window_size = model_params.get('window_size', 0)
        vocab_size = model_params['vocab_size']

        if window_size == 0:
            max_len, _ = data_utils.dataset_longest_seq(all_data)
            window_size = max_len
            
        train_x = data_utils.generate_windows_from_dataset(all_data, window_size, True)
        train_x, train_y = data_utils.shift_windows(train_x)

        train_x = np.expand_dims(train_x, axis=2)
        train_y_oh = data_utils.to_onehot(train_y, vocab_size)

        model = models.create_model(model_type, model_params.values())
        trainer = NNTrainer()
        model = trainer.train(model, self.network_fullpath, train_x, train_y_oh, train_sessions)

        return model

    def run(self, trial):
        classifier_params = trial
        classifier_params['probas_fullpath'] = trial['probas_fullpath'].format(exp_name=self.exp_name, parent_trial=flat_trial(self.parent_trial))
        self._run(*classifier_params.values())

    def _run(self, recalulate_probas, probas_fullpath):

        val_x, val_y = data_utils.load_multiple_files_with_class(self.datasets_params['val'], shuffle=False, dtype=int, exp_name=self.exp_name, parent_trial=flat_trial(self.parent_trial))
        val_fullpath = probas_fullpath.format(dataset_type = 'val')
        val_probas = self._get_dataset_proba(val_fullpath, val_x, recalulate_probas)

        thresholds = [0.5]
        roc_ts, fpr, tpr = self._get_roc_threshold(val_y, val_probas)
        thresholds.append(roc_ts)

        plot_utils.plot_probas_vs_threshold(val_fullpath, val_probas, val_y, thresholds)
        self._try_different_thresholds(val_probas, val_y, thresholds, self.val_results)

        test_x, test_y = data_utils.load_multiple_files_with_class(self.datasets_params['test'], shuffle=False, dtype=int, exp_name=self.exp_name, parent_trial=flat_trial(self.parent_trial))
        test_fullpath = probas_fullpath.format(dataset_type = 'test')
        test_probas = self._get_dataset_proba(test_fullpath, test_x, recalulate_probas)
        plot_utils.plot_probas_vs_threshold(test_fullpath, test_probas, test_y, thresholds)
        self._try_different_thresholds(test_probas, test_y, thresholds, self.test_results)

    def _get_dataset_proba(self, probas_fullpath, x_test, recalulate_probas):
        if recalulate_probas:
             probas = self._calculate_probas(probas_fullpath, x_test)
        else:
            try:
                probas = np.load(probas_fullpath, allow_pickle=True)
            except:
                probas = self._calculate_probas(probas_fullpath, x_test)
        return probas

    def _get_roc_threshold(self, y, probas):
        fpr, tpr, thresholds = roc_curve(y, probas)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        return optimal_threshold, fpr, tpr

    def _try_different_thresholds(self, probas, y, thresholds, result_writer):
        results = list(self.parent_trial.values())
        for ts in thresholds:
            y_hat = self._classify(probas, ts)
            metrics = self._metrics(y, y_hat)
            tn, fp, fn, tp, acc = metrics
            result_writer.save_results(results + [ts, *metrics])
            wandb.log({
                f"tn_{ts}":  tn,
                f"fp_{ts}": fp, 
                f"fn_{ts}": fn,
                f"tp_{ts}": tp,
                f"acc_{ts}": acc
            })

    def _classify(self, probas, threshold):
        return probas >= threshold
        
    def _metrics(self, y, y_hat): 
        tn, fp, fn, tp = confusion_matrix(y, y_hat).ravel()
        acc = accuracy_score(y, y_hat)
        return tn, fp, fn, tp, acc

    def _calculate_probas(self, probas_fullpath, x_test):

        print('\n\nCalculating Probas for', len(x_test), 'seqs.')

        window_size = self.model.layers[0].input_shape[1]
        probas = []
        for i, x in enumerate(x_test):
            print(str(i) + "/" + str(len(x_test)), end="\r")
            proba = self._probability_of_a_seq(x, window_size, verbose=0)
            probas.append(proba)
        probas = np.array(probas)
        np.save(probas_fullpath, probas, allow_pickle=True)

        return probas

    def _probability_of_a_seq(self, seq, window_size, use_padded_start = False, verbose=2):
        if use_padded_start:
            seq = [0] + list(seq)

        if(verbose > 0):
            print("Evaluating sequence...")
            print(seq)

        padded_prefixes = data_utils.generate_windows_from_dataset([seq], window_size)
        padded_prefixes = np.expand_dims(padded_prefixes, axis=2)
        yhat = self.model.predict(padded_prefixes, verbose=0)
        
        next_symbol_proba_vector = -1
        proba = 0
        start = 1
        for i in range(start, len(seq)):
            if(verbose > 2):
                print("\n##################################")
                print("\nCurrent Subsequence:", str(seq[:i]))
                print(padded_prefixes[i-start])
            
            expected_symbol = seq[i]
            expected_symbol_index = expected_symbol
            
            yhat_last = yhat[i-start][next_symbol_proba_vector]
            precited_proba_for_expected_symbol = yhat_last[expected_symbol_index]
            
            if(verbose > 2):
                print("Max Symbol proba:", np.argmax(yhat_last), "proba:", np.max(yhat_last))
                print("Expected Symbol proba:", expected_symbol_index, "proba:", precited_proba_for_expected_symbol)
                top5 = yhat_last.argsort()[-5:][::-1]
                print("Top 5 Symbols:", top5)

            proba = proba + np.log(precited_proba_for_expected_symbol) 

            if(verbose > 1):
                print("Subsequence:", str(seq[:i+1]), "proba:", str(proba))

        if(verbose > 2):
            print("\n##################################")
        if(verbose > 0):
            print("Final proba:", str(proba), "\n\n")

        return proba
