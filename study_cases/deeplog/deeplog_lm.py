from keras.models import Sequential, load_model
from keras.layers import Embedding, LSTM, Dense, TimeDistributed
from keras.optimizers import Adam

from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
from uuid import uuid1
import csv
import os.path


import plot_utils
from data_utils import load_dataset_for_lm, unison_shuffled_copies, dataset_vocab, dataset_longest_seq, to_onehot, load_test, stack_datasets


class DeepLogLMClassifier:

    def __init__(self, experiment, iteration, datasets_params, model_fullpath, train_params, classifier_params):
        self.iteration = iteration
        self.datasets_params = datasets_params

        self.model_fullpath = model_fullpath.format(exp_name=experiment)
        self.train_params = train_params

        self.classifier_params = classifier_params
        self.classifier_params['probas_fullpath'] = self.classifier_params['probas_fullpath'].format(
            exp_name=experiment, topk=self.classifier_params['use_top_k'])

        self.classifier_params['results_fullpath'] = self.classifier_params['results_fullpath'].format(
            exp_name=experiment)

        self.classifier_params['plots_fullpath'] = self.classifier_params['plots_fullpath'].format(
            exp_name=experiment)

        self._get_model()

    def _get_model(self):
        try:
            self.model = load_model(self.model_fullpath)
        except:
            print("\nModel", self.model_fullpath, "not found. Training started...")
            self.model = self._train_model(*self.train_params.values())

    def _train_model(self, window_size, vocab_size, train_sessions):
        dataset_info = self.datasets_params["train"]

        train_x, train_y = load_dataset_for_lm(
            dataset_info["fullpath"], window_size=window_size, amount_to_load=dataset_info["to_read"])
        train_x, train_y = unison_shuffled_copies(train_x, train_y)

        train_x = np.expand_dims(train_x, axis=3)
        train_y_oh = to_onehot(train_y, vocab_size)

        #The LSTM input layer must be 3D.
        #The meaning of the 3 input dimensions are: samples, time steps, and features.
        #The LSTM input layer is defined by the input_shape argument on the first hidden layer.
        #The input_shape argument takes a tuple of two values that define the number of time steps and features.
        #The number of samples is assumed to be 1 or more.
        #Either input_shape(length=None or window_size for fixed length, input_dim=vocab_size,)...
        ####...if symbols in each x of X is a onehot vector [[0 0 0 1] [1 0 0 0] [0 0 0 1]]
        ####...In this case X for train must be (batch, window_size, vocab_size)
        #OR input_shape(length=None or window_size, input_dim=1,)...
        ###...if X is a vector of integer symbols [4 1 4].
        ###...In this chase X for train must be (batch, window_size, 1) and np.expand_dims(train_x, axis=3) is needed

        model = Sequential()
        model.add(LSTM(128, return_sequences=True, input_shape=(None, 1,)))
        model.add(TimeDistributed(Dense(vocab_size, activation='softmax')))

        for key in train_sessions.keys():
            print("\nTrain Session:", key)
            session_info = train_sessions[key]
            epochs, batch_size, lr = session_info.values()

            model.compile(loss='categorical_crossentropy', optimizer= Adam(lr=lr))
           
            history = model.fit(train_x, train_y_oh, epochs = epochs, batch_size = batch_size, validation_split=0.2, verbose=1)
            #plot_utils.plot(history)

        
        model.save(self.model_fullpath)
        return model

    def run_test(self):
        self._run(*self.classifier_params.values())

    def _run(self, use_top_k, thresholds, recalulate_probas, probas_fullpath, results_fullpath, plots_fullpath):

        X_test_normal, y_test_normal = self._load_test_data('test_normal', 1)
        X_test_abnormal, y_test_abnormal = self._load_test_data('test_abnormal', 0)

        X_test = stack_datasets(X_test_normal, X_test_abnormal, axis=1)
        y_test = stack_datasets(y_test_normal, y_test_abnormal, axis=1)

        if recalulate_probas:
             probas = self._calculate_test_probas(probas_fullpath, X_test, use_top_k)
        else:
            try:
                probas = np.load(probas_fullpath, allow_pickle=True)
            except:
                probas = self._calculate_test_probas(probas_fullpath, X_test, use_top_k)

        for ts in thresholds:
            plot_name = self._plot_probas_vs_threshold(plots_fullpath, probas, y_test, ts)
            res = self._classify(probas, y_test, ts)
            self._save_results(results_fullpath, use_top_k, ts, plot_name, *res)

    def _load_test_data(self, test_type, y_class):
        dataset_info = self.datasets_params[test_type]
        X_test_name = dataset_info["fullpath"]
        X_test = load_test(X_test_name, amount_to_load=dataset_info["to_read"])
        y_test = np.ones(len(X_test)) * y_class
        return X_test, y_test

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

