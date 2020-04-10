import numpy as np
import matplotlib.pyplot as plt
from uuid import uuid1

def plot(history, metric=None, save_path=None):
    
    plt.clf()
    if metric == 'acc':
        # Plot training & validation accuracy values
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        #plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(save_path + '_plot.png')

def plot_probas_vs_threshold(plots_fullpath, probas, y, thresholds):
        plot_name = plots_fullpath + '_plot.png' #plots_fullpath.format(uuid=str(uuid1()))
        colors = ['red' if y_i == 0 else 'green' for y_i in y]

        plt.clf()
        ax = plt.subplot(3, 1, 1) 
        ax.axis([0, len(probas), 0, 0.0005])
        ax.scatter(x=np.arange(len(probas)), y=probas, c=colors)
        for ts in thresholds:
            ax.hlines(y=ts, xmin=0, xmax=len(probas))
        ax.set_ylabel('Proba')
        ax.set_title(plot_name)

        ax2 = plt.subplot(3,1,2) 
        ax2.axis([0, len(probas), 0, 0.05])
        ax2.scatter(x=np.arange(len(probas)), y=probas, c=colors)
        for ts in thresholds:
            ax2.hlines(y=ts, xmin=0, xmax=len(probas))
        ax2.set_ylabel('Proba')
        
        ax3 = plt.subplot(3,1,3) 
        ax3.axis([0, len(probas), 0, 1])
        ax3.scatter(x=np.arange(len(probas)), y=probas, c=colors)
        for ts in thresholds:
            ax3.hlines(y=ts, xmin=0, xmax=len(probas))
        ax3.set_ylabel('Proba')

        plt.savefig(plot_name)