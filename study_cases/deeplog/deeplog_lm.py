from keras.models import load_model

from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
from uuid import uuid1
import csv
import os.path



from data_utils import load_dataset_for_lm, unison_shuffled_copies, dataset_vocab, dataset_longest_seq, to_onehot, load_test, stack_datasets

from nn_trainer import NNTrainer
from study_cases.deeplog.models import create_model

class DeepLogLMClassifier:

    def __init__(self, experiment, iteration, datasets_params, network_fullpath, network_params, classifier_params):
        self.experiment = experiment
        self.iteration = iteration

        self.datasets_params = datasets_params
        self.network_fullpath = network_fullpath.format(exp_name=experiment, iteration=iteration)
        self.network_params = network_params

        self.classifier_params = classifier_params
        self.classifier_params['probas_fullpath'] = self.classifier_params['probas_fullpath'].format(exp_name=experiment, topk=self.classifier_params['use_top_k'], iteration = iteration)
        self.classifier_params['results_fullpath'] = self.classifier_params['results_fullpath'].format(exp_name=experiment)
        self.classifier_params['plots_fullpath'] = self.classifier_params['plots_fullpath'].format(exp_name=experiment)

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
            path = t_set["fullpath"].format(exp_name=self.experiment, iteration=self.iteration)
            temp_x, temp_y =  load_dataset_for_lm(path, window_size=window_size, amount_to_load=t_set["to_read"])
            train_x = stack_datasets(train_x, temp_x, 1)
            train_y = stack_datasets(train_y, temp_y, 1)
        train_x, train_y = unison_shuffled_copies(train_x, train_y)

        train_x = np.expand_dims(train_x, axis=3)
        train_y_oh = to_onehot(train_y, vocab_size)

        model = create_model(model_type, [vocab_size])
        trainer = NNTrainer()
        model = trainer.train(model, self.network_fullpath, train_x, train_y_oh, train_sessions)

        return model

    def run_test(self):
        self._run(*self.classifier_params.values())

    def _run(self, use_top_k, thresholds, recalulate_probas, probas_fullpath, results_fullpath, plots_fullpath):

        t_sets = self.datasets_params['test']
        test_x = np.array([])
        test_y = np.array([])
        for dataset_name in t_sets:
            t_set = t_sets[dataset_name]
            path = t_set["fullpath"].format(exp_name=self.experiment, iteration=self.iteration)

            temp_x = load_test(path, amount_to_load=t_set["to_read"])
            temp_y = np.ones(len(temp_x)) * t_set['class']
            test_x = stack_datasets(test_x, temp_x, 1)
            test_y = stack_datasets(test_y, temp_y, 1)

        if recalulate_probas:
             probas = self._calculate_test_probas(probas_fullpath, test_x, use_top_k)
        else:
            try:
                probas = np.load(probas_fullpath, allow_pickle=True)
            except:
                probas = self._calculate_test_probas(probas_fullpath, test_x, use_top_k)

        for ts in thresholds:
            plot_name = self._plot_probas_vs_threshold(plots_fullpath, probas, test_y, ts)
            res = self._classify(probas, test_y, ts)
            self._save_results(results_fullpath, use_top_k, ts, plot_name, *res)

    def _calculate_test_probas(self, probas_fullpath, X_test, use_top_k):
        probas = []
        for i, x in enumerate(X_test):
            print(str(i) + "/" + str(len(X_test)), end="\r")
            proba = self._evaluate_seq(x, use_top_k, verbose=0)
            probas.append(proba)
        probas = np.array(probas)
        np.save(probas_fullpath, probas, allow_pickle=True)
        return probas

    def _classify(self, probas, y_test, threshold):
        normal = probas > threshold
        tn, fp, fn, tp = confusion_matrix(y_test, normal).ravel()
        acc = accuracy_score(y_test, normal)
        return tn, fp, fn, tp, acc

    def _plot_probas_vs_threshold(self, plots_fullpath, probas, y_test, threshold):
        plot_name = plots_fullpath.format(uuid=str(uuid1()))
        colors = ['red' if x == 0 else 'green' for x in y_test]

        ax = plt.subplot(3, 1, 1) 
        ax.axis([0, len(probas), 0, 0.0005])
        ax.scatter(x=np.arange(len(probas)), y=probas, c=colors)
        ax.hlines(y=threshold, xmin=0, xmax=len(probas))
        ax.set_ylabel('Proba')
        ax.set_title(plot_name)

        ax2 = plt.subplot(3,1,2) 
        ax2.axis([0, len(probas), 0, 0.05])
        ax2.scatter(x=np.arange(len(probas)), y=probas, c=colors)
        ax2.hlines(y=threshold, xmin=0, xmax=len(probas))
        ax2.set_ylabel('Proba')
        
        ax3 = plt.subplot(3,1,3) 
        ax3.axis([0, len(probas), 0, 1])
        ax3.scatter(x=np.arange(len(probas)), y=probas, c=colors)
        ax3.hlines(y=threshold, xmin=0, xmax=len(probas))
        ax3.set_ylabel('Proba')

        plt.savefig(plot_name)
        return plot_name
        #plt.show()

    def _save_results(self, results_fullpath, use_top_k, threshold, plot_name, tn, fp, fn, tp, acc):
        add_header = False
        if not os.path.isfile(results_fullpath):
            add_header = True
        with open(results_fullpath, 'a+', newline='') as csvfile:
            writer = csv.writer(csvfile)
            if add_header:
                writer.writerow(['date', 'iter', 'use_top_k', 'threshold', 'tn', 'fp', 'fn', 'tp', 'acc', 'plot_name'])
            writer.writerow([datetime.timestamp(datetime.now()), self.iteration, use_top_k, threshold, tn, fp, fn, tp, acc, plot_name])

    def _evaluate_seq(self, seq, use_top_k=0, verbose=2):
        sequences = []
        probas = 1
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

            onehot_seq = np.expand_dims([seq[:i]], axis=2)
            if(verbose > 2):
                print("In Seq:", onehot_seq.shape)
            yhat = self.model.predict(onehot_seq)

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
                probas = probas * precited_proba_for_expected_symbol

            if(verbose > 1):
                print("Subsequence:", str(seq[:i+1]), "proba:", str(probas))

        if(verbose > 2):
            print("\n##################################")
        if(verbose > 0):
            print("Final proba:", str(probas), "\n\n")
        return probas

