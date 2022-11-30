from tensorflow import keras
import tensorflow as tf
from keras.utils.image_dataset import image_dataset_from_directory
from keras.layers import Rescaling
import time
from src.utils.io import save_model_weights, save_model_config, TrainParams, save_model, load_datasets, checkDataset
from src import models
from src.testing import model_test
from typing import Tuple
import os
from src.utils.paths import CONFIGS_DIR, WEIGHTS_DIR, TRAIN_DIR, VALID_DIR
import matplotlib.pyplot as plt


def train_model(name: str,
                train_gen: tf.data.Dataset,
                valid_gen: tf.data.Dataset,
                opt: keras.optimizers,
                ep: int,
                batch_size: int,
                learn_rate: float,
                loss: str,
                save_name: str = None,
                test=True,
                save=True,
                reinitialize=False):

    model = getattr(models, name, None)

    if reinitialize:
        model = keras.Sequential.from_config(model.get_config())

    model.compile(optimizer=opt, loss=loss, metrics=['accuracy'])
    model.summary()

    start = time.time()
    history = model.fit(train_gen, validation_data=valid_gen, epochs=ep, batch_size=batch_size)
    stop = time.time() - start

    params = TrainParams(ep, batch_size, learn_rate, loss, stop, history.history)

    if save and save_name is not None:
        save_model(CONFIGS_DIR, WEIGHTS_DIR, save_name, model, opt, params)
    if test:
        model_test(model, params, valid_gen)


if __name__ == "__main__":
    # Name of model in models.py
    name = "base15m"
    # Name you want to save the model as ex: base2m_adam
    name_sv = "deep15m_sgd_tune"

    # Defaults: ep: 30, batch_size: 32, learn_rate: 0.0001
    ep = 30
    batch_size = 32
    learn_rate = 0.01
    loss = "sparse_categorical_crossentropy"

    # LR=[1e-2, 5e-3]
    # opt = keras.optimizers.Adagrad(learning_rate=learn_rate, epsilon=1e-3)

    # OLD: epsilon = [0.00001, ] = [1e-5, ], beta_2 = [0.888, 0.999, 1.11], beta_1 = [0.85, 0.9, 0.95]

    # momentum=[0.1, 0.115, 0.125, 0.2]
    # learn_rate=[0.125, ]
    # opt = keras.optimizers.SGD(learning_rate=learn_rate, momentum=0.5)

    opt = keras.optimizers.SGD(learning_rate=learn_rate, momentum=0.3)

    train, valid = load_datasets(TRAIN_DIR, VALID_DIR, batch_size=batch_size)

    train_model(name, train, valid, opt, ep, batch_size, learn_rate, loss, save_name=name_sv, test=True)












