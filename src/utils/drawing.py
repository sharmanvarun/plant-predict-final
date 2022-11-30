import os
import keras.callbacks
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as sm
from typing import List

__all__ = ['training_plot', 'confusion_matrix']


# Code modified form: https://www.kaggle.com/code/deepmalviya7/plant-disease-detection-using-cnn-with-96-84

def training_plot(history: dict, epochs: int, show=False, y_loss_max: float = None):
    ml = max(max(history['loss']), max(history['val_loss']))
    if y_loss_max is None:
        y_max = ml + 0.6 if (ml > 1) else 1.0
    else:
        y_max = y_loss_max

    plt.figure(figsize=(20, 5))
    plt.subplot(1, 2, 1)
    plt.title("Train and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(history['loss'], label="Train Loss")
    plt.plot(history['val_loss'], label="Validation Loss")
    plt.xlim(0, epochs - 1)
    plt.ylim(0.0, y_max)
    plt.legend()

    mc = max(max(history['accuracy']), max(history['val_accuracy']))
    y_min = np.asarray((0.0, 0.5, 0.75), np.float)
    i = (np.abs(y_min - mc)).argmin()
    y_min = y_min[i]

    plt.subplot(1, 2, 2)
    plt.title("Train and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.plot(history['accuracy'], label="Train Accuracy")
    plt.plot(history['val_accuracy'], label="Validation Accuracy")
    plt.xlim(0, epochs - 1)
    plt.ylim(y_min, 1.0)
    plt.legend()
    plt.tight_layout()

    if show:
        plt.show()


def confusion_matrix(cm: np.ndarray, show=False):
    disp = sm.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(range(1, 39)))
    fig, ax = plt.subplots(figsize=(15, 15))
    disp.plot(ax=ax, colorbar=False, cmap='YlGnBu')
    plt.title("Confusion Matrix")
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')

    if show:
        plt.show()


def training_plots(hists: List[dict], labels: list, epoch: int, y_min: float = None, show: bool = True):
    plt.figure(figsize=(20, 5))
    plt.subplot(1, 2, 1)
    plt.title("Train Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    mc = 0.76
    y_mins = np.asarray((0.0, 0.5, 0.75), np.float)

    for i, history in enumerate(hists):
        plt.plot(history['accuracy'], label=labels[i])
        mc_t = max(history['accuracy'])
        if mc_t < mc:
            mc = mc_t

    if y_min is None:
        i = (np.abs(y_mins - mc)).argmin()
        y_min = y_mins[i]

    plt.xlim(0, epoch - 1)
    plt.ylim(y_min, 1.0)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.title("Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    mc = 0.76
    for i, history in enumerate(hists):
        plt.plot(history['val_accuracy'], label=labels[i])
        mc_t = max(history['val_accuracy'])
        if mc_t < mc:
            mc = mc_t

    if y_min is None:
        i = (np.abs(y_mins - mc)).argmin()
        y_min = y_mins[i]

    plt.xlim(0, epoch - 1)
    plt.ylim(y_min, 1.0)
    plt.legend()
    plt.tight_layout()

    if show:
        plt.show()


def training_loss_plots(hists: List[dict], labels: list, epoch: int, y_max: float = None, show: bool = True):
    plt.figure(figsize=(20, 5))
    plt.subplot(1, 2, 1)
    plt.title("Train loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    mc = 0.0
    set_max = y_max is None

    # y_maxes = np.asarray((1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0), np.float)

    for i, history in enumerate(hists):
        plt.plot(history['loss'], label=labels[i])

        mc_t = max(history['loss'])
        if mc_t > mc:
            mc = mc_t

    if set_max:
        #i = (np.abs(y_maxes - mc)).argmin()
        #y_max = y_maxes[i]
        y_max = mc

    plt.xlim(0, epoch - 1)
    plt.ylim(0.0, y_max)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.title("Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("loss")

    mc = 0.0
    for i, history in enumerate(hists):
        plt.plot(history['val_loss'], label=labels[i])

        mc_t = max(history['val_loss'])
        if mc_t > mc:
            mc = mc_t

    if set_max:
        #i = (np.abs(y_maxes - mc)).argmin()
        #y_max = y_maxes[i]
        y_max = mc

    plt.xlim(0, epoch - 1)
    plt.ylim(0.0, y_max)
    plt.legend()
    plt.tight_layout()

    if show:
        plt.show()
