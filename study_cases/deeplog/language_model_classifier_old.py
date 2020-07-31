import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.python.distribute.mirrored_strategy import MirroredStrategy

from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve

import common.plot_utils as plot_utils
import common.data_utils as data_utils
from common.nn_trainer import NNTrainer
from common.csv_result import CSVResult

import study_cases.deeplog.models as models
import study_cases.deeplog.deeplog_data_utils as d_utils


class LMClassifier:

    def __init__(self, experiment, datasets_params, network_fullpath, network_params, classifier_params, results_fullpath):
        self.exp_name, self.epsilon, self.iteration = experiment

        self.datasets_params = datasets_params
        self.network_fullpath = network_fullpath.format(exp_name=self.exp_name, epsilon=self.epsilon, iteration=self.iteration)
        self.network_params = network_params

        self.classifier_params = classifier_params
        self.classifier_params['probas_fullpath'] = self.classifier_params['probas_fullpath'].format(exp_name=self.exp_name, topk=self.classifier_params['use_top_k'], epsilon=self.epsilon, iteration =self.iteration)

        results_header = ['eps', 'iter', 'use_top_k', 'threshold', 'tn', 'fp', 'fn', 'tp', 'acc']
        val_results_fullpath = results_fullpath.format(exp_name=self.exp_name, dataset_type='val')
        test_results_fullpath = results_fullpath.format(exp_name=self.exp_name, dataset_type='test')
        self.val_results = CSVResult(val_results_fullpath, results_header)
        self.test_results = CSVResult(test_results_fullpath, results_header)

        #strategy = MirroredStrategy()
        #with strategy.scope():
        self._get_model()

    def _get_model(self):
        try:
            self.model = load_model(self.network_fullpath)
        except:
            print("\nModel", self.network_fullpath, "not found. Training started...")
            self.model = self._train_model(*self.network_params.values())

    def _train_model(self, model_type, window_size, vocab_size, train_sessions):
        
        t_sets = self.datasets_params['train']
        train_x = np.array([])
        train_y = np.array([])
        for dataset_name in t_sets:
            t_set = t_sets[dataset_name]
            path = t_set["fullpath"].format(exp_name=self.exp_name, epsilon=self.epsilon, iteration=self.iteration)
            temp_x, temp_y =  d_utils.load_dataset_for_lm(path, window_size=window_size, amount_to_load=t_set["to_read"])
            train_x = data_utils.stack_datasets(train_x, temp_x, axis=0)
            train_y = data_utils.stack_datasets(train_y, temp_y, axis=0)
        train_x, train_y = data_utils.unison_shuffled_copies(train_x, train_y)

        train_x = np.expand_dims(train_x, axis=2)
        train_y_oh = data_utils.to_onehot(train_y, vocab_size)

        model = models.create_model(model_type, [vocab_size])
        trainer = NNTrainer()
        model = trainer.train(model, self.network_fullpath, train_x, train_y_oh, train_sessions)

        return model

    def run_test(self):
        self._run(*self.classifier_params.values())

    def _run(self, use_top_k, roc_thresholds, custom_thresholds, recalulate_probas, probas_fullpath):

        val_x, val_y = self._load_test(dataset_type ='val')
        val_fullpath = probas_fullpath.format(dataset_type = 'val')
        val_probas = self._get_dataset_proba(val_fullpath, val_x, recalulate_probas, use_top_k)

        if use_top_k > 0:
            thresholds = [0.5]
        else:
            thresholds = custom_thresholds
            if roc_thresholds:
                roc_ts, fpr, tpr = self._get_roc_threshold(val_y, val_probas)
                thresholds.append(roc_ts)

        plot_utils.plot_probas_vs_threshold(val_fullpath, val_probas, val_y, thresholds)
        self._try_different_thresholds(val_probas, val_y, thresholds, self.val_results, use_top_k)

        test_x, test_y = self._load_test(dataset_type ='test')
        test_fullpath = probas_fullpath.format(dataset_type = 'test')
        test_probas = self._get_dataset_proba(test_fullpath, test_x, recalulate_probas, use_top_k)

        plot_utils.plot_probas_vs_threshold(test_fullpath, test_probas, test_y, thresholds)
        self._try_different_thresholds(test_probas, test_y, thresholds, self.test_results, use_top_k)

    def _load_test(self, dataset_type):
        t_sets = self.datasets_params[dataset_type]
        set_x = []
        set_y = []
        for dataset_name, t_set in t_sets.items():
            path = t_set["fullpath"].format(exp_name=self.exp_name, epsilon=self.epsilon, iteration=self.iteration)
            temp_x = data_utils.load_file(path, to_read=t_set["to_read"], shuffle=False, dtype=int, split_token='')
            temp_y = [t_set['class']]*len(temp_x) #np.ones(len(temp_x)) * t_set['class']
            set_x = data_utils.stack_datasets(set_x, temp_x, 1)
            set_y = data_utils.stack_datasets(set_y, temp_y, 1)
        return set_x, set_y

    def _get_dataset_proba(self, probas_fullpath, set_x, recalulate_probas, use_top_k):
        if recalulate_probas:
             probas = self._calculate_probas(probas_fullpath, set_x, use_top_k)
        else:
            try:
                probas = np.load(probas_fullpath, allow_pickle=True)
            except:
                probas = self._calculate_probas(probas_fullpath, set_x, use_top_k)
        return probas

    def _calculate_probas(self, probas_fullpath, set_x, use_top_k):
        probas = []
        print('\n\nCalculating Probas for', len(set_x), 'seqs.')
        for i, x in enumerate(set_x):
            print(str(i) + "/" + str(len(set_x)), end="\r")
            proba = self._evaluate_seq(x, use_top_k, verbose=0)
            probas.append(proba)
        probas = np.array(probas)
        np.save(probas_fullpath, probas, allow_pickle=True)
        return probas

    def _get_roc_threshold(self, y, probas):
        fpr, tpr, thresholds = roc_curve(y, probas)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        return optimal_threshold, fpr, tpr

    def _try_different_thresholds(self, probas, y, thresholds, result_writer, use_top_k):
        for ts in thresholds:
            y_hat = self._classify(probas, ts)
            metrics = self._metrics(y, y_hat)
            result_writer.save_results([self.epsilon, self.iteration, use_top_k, ts, *metrics])

    def _classify(self, probas, threshold):
        return probas >= threshold
        
    def _metrics(self, y, y_hat): 
        tn, fp, fn, tp = confusion_matrix(y, y_hat).ravel()
        acc = accuracy_score(y, y_hat)
        return tn, fp, fn, tp, acc

    def _evaluate_seq(self, seq, use_top_k=0, verbose=2):
        sequences = []
        probas = np.log(1)
        batch_index = 0
        next_symbol_proba_vector = -1
        if(verbose > 0):
            print("Evaluating sequence...")
            print(seq)

        for i in range(1, len(seq)):
            if(verbose > 2):
                print("\n##################################")
                print("\nCurrent Subsequence:", str(seq[:i]))

            expected_symbol_index = seq[i]

            input_model_seq = np.expand_dims([seq[:i]], axis=2)
            if(verbose > 2):
                print("In Seq:", input_model_seq.shape)
            yhat = self.model.predict(input_model_seq)

            yhat_last = yhat[batch_index][next_symbol_proba_vector]
            precited_proba_for_expected_symbol = yhat_last[expected_symbol_index]

            if(verbose > 2):
                print("Max Symbol proba:", np.argmax(
                    yhat_last), "proba:", np.max(yhat_last))
                print("Expected Symbol proba:", expected_symbol_index,
                      "proba:", precited_proba_for_expected_symbol)

            if(use_top_k > 0):
                topk = yhat_last.argsort()[-use_top_k:][::-1]
                in_top = expected_symbol_index in topk
                if(verbose > 2):
                    print("Top", use_top_k, "Symbols:", topk)
                    print("Is Expected in top:", use_top_k, in_top)
                if not in_top:
                    #probas = probas * precited_proba_for_expected_symbol
                    return 0
            else:
                probas = probas + np.log(precited_proba_for_expected_symbol) 

            if(verbose > 1):
                print("Subsequence:", str(seq[:i+1]), "proba:", str(probas))

        if(verbose > 2):
            print("\n##################################")
        if(verbose > 0):
            print("Final proba:", str(probas), "\n\n")
        return probas

