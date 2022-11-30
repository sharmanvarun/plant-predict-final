from tensorflow import keras
import tensorflow as tf
from src.utils.io import save_to_json, load_datasets, load_json_files, load_json_file, file_search
from src import models
import matplotlib.pyplot as plt
import numpy as np
import os
from src.utils.paths import TRAIN_DIR, VALID_DIR, TUNING_DIR
import logging
from typing import Union, List, Iterable, Tuple, Dict, Optional


class LRTuner:
    def __init__(self, mod_name: str, opt: keras.optimizers.Optimizer, loss: str, beta: float = 0.98, thresh=169):

        self.beta = beta
        self.thresh = thresh
        self.mod_name = mod_name
        self.opt = opt
        self.loss_func = loss

        self.l_rates = []
        self.losses = []
        self.raw_losses = []
        self.min_lr = 0.001
        self.max_lr = 1.0
        self.curr_batch = 0
        self.lr_mul = 1
        self.avg_loss = 0
        self.min_loss = 1e9

        self.batch_size = -1
        self.epochs = -1

        config = getattr(models, mod_name, None).get_config()
        self.model = keras.Sequential.from_config(config)
        self.model.compile(optimizer=opt, loss=loss, metrics=['accuracy'])
        self.model.summary()

    def reset(self):
        self.l_rates = []
        self.losses = []
        self.raw_losses = []
        self.min_lr = 0.001
        self.max_lr = 1.0
        self.curr_batch = 0
        self.lr_mul = 1
        self.avg_loss = 0
        self.min_loss = 1e9

        self.batch_size = -1
        self.epochs = -1

        config = getattr(models, self.mod_name, None).get_config()
        self.model = keras.Sequential.from_config(config)
        self.model.compile(optimizer=self.opt, loss=self.loss_func, metrics=['accuracy'])

    def on_batch_end(self, batch, logs):
        lr = keras.backend.get_value(self.model.optimizer.lr).item()
        self.l_rates.append(lr)

        loss = logs["loss"]
        self.raw_losses.append(loss)
        self.curr_batch += 1

        self.avg_loss = (self.beta * self.avg_loss) + ((1 - self.beta) * loss)
        smooth = self.avg_loss / (1 - self.beta**self.curr_batch)
        self.losses.append(smooth)

        thresh = self.thresh * self.min_loss

        if self.curr_batch > 1 and smooth > thresh:
            logging.info(f"Loss threshold exceeded, aborting tuning")
            self.model.stop_training = True
            return

        if self.curr_batch > 1 and smooth < self.min_loss:
            self.min_loss = smooth

        lr *= self.lr_mul
        keras.backend.set_value(self.model.optimizer.lr, np.float32(lr))

    def tune(self, train_gen: tf.data.Dataset, valid_gen: tf.data.Dataset, min_lr : float, max_lr: float, epochs: int,
             batch_size, verbose=1):

        self.min_lr = min_lr
        self.max_lr = max_lr
        self.epochs = epochs
        self.batch_size = batch_size

        steps_per_epoch = len(train_gen)
        num_batch_updates = epochs * steps_per_epoch

        self.lr_mul = (max_lr / min_lr) ** (1.0 / num_batch_updates)

        keras.backend.set_value(self.model.optimizer.lr, min_lr)

        callback = keras.callbacks.LambdaCallback(
            on_batch_end=lambda batch, logs: self.on_batch_end(batch, logs)
        )

        self.model.fit(train_gen, validation_data=valid_gen, epochs=epochs, verbose=verbose,
                       steps_per_epoch=steps_per_epoch, callbacks=[callback])

    def save(self, folder: str, name: str):
        data = {
            "opt": self.model.optimizer.get_config(),
            "ep": self.epochs,
            "batch_size": self.batch_size,
            "min_lr": self.min_lr,
            "max_lr": self.max_lr,
            "loss_func": self.loss_func,
            "losses": self.losses[:],
            "raw_losses": self.raw_losses[:],
            "lr_rates": self.l_rates[:]
        }

        parsed = []
        conf: dict = data["opt"]
        conf["str_float"] = parsed

        for key in conf.keys():
            v = conf[key]
            if type(v) == np.float32:
                parsed.append(key)
                conf[key] = str(v)

        save_to_json(os.path.join(folder, f"{name}.json"), data)

    @classmethod
    def create_opt_attr_label(cls, opt_conf: dict, attrs: list, decimals: int = 3, tune_conf: dict = None) -> str:
        strings = []
        for attr in attrs:
            v = opt_conf.get(attr, None)

            if v is None and tune_conf is not None:
                v = tune_conf.get(attr, None)

            if type(v) in (np.float32, float):
                v = float(np.round(v, decimals))

            strings.append(f"{attr}={v}")
        return ", ".join(strings)

    def plot_loss(self, skipBegin=10, skipEnd=1, opt_attrs: list = None):
        lrs = self.l_rates[skipBegin:-skipEnd]
        losses = self.losses[skipBegin:-skipEnd]

        if opt_attrs is not None:
            conf = self.model.optimizer.get_config()
            plt.plot(lrs, losses, label=self.create_opt_attr_label(conf, opt_attrs))
            plt.legend()
        else:
            plt.plot(lrs, losses)

        plt.xscale("log")
        plt.xlim(self.min_lr, self.max_lr)
        plt.xlabel("Learning Rate (Log Scale)")
        plt.ylabel("Loss")
        plt.show()

    @classmethod
    def plot_loss_from_file(cls, fp: str, skipBegin=10, skipEnd=1, opt_attrs:list = None, useLog=True,
                            fn_in_label=False, loss_lim: tuple = None):
        data = load_json_file(fp)

        lrs = data["lr_rates"][skipBegin:-skipEnd]
        losses = data["losses"][skipBegin:-skipEnd]

        if opt_attrs is not None:
            label = cls.create_opt_attr_label(data["opt"], opt_attrs, tune_conf=data)
            if fn_in_label:
                fn, ext = os.path.splitext(os.path.basename(fp))
                label = f"{fn}: {label}"

            plt.plot(lrs, losses, label=label)
            plt.legend()
        else:
            plt.plot(lrs, losses)

        plt.xlim(data["min_lr"], max(lrs))
        plt.xlabel("Learning Rate (Log Scale)")
        plt.ylabel("Loss")
        if useLog:
            plt.xscale("log")
        if loss_lim is not None:
            plt.ylim(loss_lim[0], loss_lim[1])
        # plt.tight_layout()
        plt.show()

    @classmethod
    def plot_loss_from_files(cls, folder: str, prefix: Union[str, Iterable[str]], skipBegin=10, skipEnd=1,
                             opt_attrs: list = None, loss_lim: tuple = None, separate_prefix_plots=False,
                             use_log=True, fn_in_label=False, opt_name_in_label=False):

        paths = []
        data = []

        if type(prefix) != str:
            for pf in prefix:
                p = file_search(folder, pf)
                d = load_json_files(p)
                paths.append(p)
                data.append(d)
        else:
            p = file_search(folder, prefix, recursive=False)
            d = load_json_files(p)
            paths.append(p)
            data.append(d)

        for j, dat in enumerate(data):
            mins = []
            maxes = []
            pat = paths[j]

            if separate_prefix_plots:
                plt.figure()

            if opt_attrs is None:
                for d in dat:
                    lrs = d["lr_rates"][skipBegin:-skipEnd]
                    losses = d["losses"][skipBegin:-skipEnd]
                    mins.append(d["min_lr"])
                    maxes.append(max(lrs))

                    plt.plot(lrs, losses)
            else:
                for i, d in enumerate(dat):
                    lrs = d["lr_rates"][skipBegin:-skipEnd]
                    losses = d["losses"][skipBegin:-skipEnd]
                    mins.append(d["min_lr"])
                    maxes.append(max(lrs))

                    label = cls.create_opt_attr_label(d["opt"], opt_attrs, tune_conf=d)
                    if fn_in_label:
                        fn = os.path.basename(pat[i])
                        label = f"{fn}: {label}"
                    elif opt_name_in_label:
                        label = f"{d['opt']['name']} {label}"
                    plt.plot(lrs, losses, label=label)
                plt.legend()

            if loss_lim is not None:
                plt.ylim(loss_lim[0], loss_lim[1])

            plt.xlim(np.min(mins), np.max(maxes))
            plt.xlabel("Learning Rate (Log Scale)")
            plt.ylabel("Loss")

            if use_log:
                plt.xscale("log")

        plt.show()
        # plt.tight_layout()

    @classmethod
    def findFileWithLowestLoss(cls, folder: str, prefix: Union[str, Iterable[str]], recursive=False) -> Dict[str, Optional[dict]]:
        paths = file_search(folder, prefix, recursive=recursive)
        data = load_json_files(paths)

        min_smooth = 1e9
        min_raw = 1e9
        smooth_idx = -1
        raw_idx = -1

        for i, file in enumerate(data):
            smooth = min(file["losses"])
            raw = min(file["raw_losses"])
            file["fn"] = os.path.basename(paths[i])

            if smooth < min_smooth:
                min_smooth = smooth
                smooth_idx = i
            if raw < min_raw:
                min_raw = raw
                raw_idx = i

        if (smooth_idx + raw_idx) == -2:
            return dict(smooth=None, raw=None)
        elif smooth_idx == -1:
            return dict(smooth=None, raw=data[raw_idx])
        elif raw_idx == -1:
            return dict(smooth=data[smooth_idx], raw=None)
        else:
            return dict(smooth=data[smooth_idx], raw=data[raw_idx])


if __name__ == "__main__":
    # Name of model in models.py
    name = "base15m"
    # Name you want to save the model as ex: base2m_adam
    name_sv = "base15m_adagrad_tune"

    # Defaults: ep: 30, batch_size: 32, learn_rate: 0.0001
    ep = 10
    batch_size = 32
    min_lr = 1e-3      # 1e-6
    max_lr = 2.0        # 1.0
    loss = "sparse_categorical_crossentropy"

    # SDG temp: lr_rate = [1.0, 0.5, 0.1, 0.01]
    # momentum = [0, 0.1, 0.3, 0.6, 0.9, 1.0]
    # opt = keras.optimizers.SGD()
    # attrs = ["momentum", "nesterov"]

    # beta_1 = [0.85, 0.9, 0.95]
    # beta_2 = [0.888, 0.999, 1.11]
    # epsilon = [1e-6, 1e-5]
    #opt = keras.optimizers.Adam(epsilon=1e-4)
    #attrs = ["epsilon", "beta_1", "beta_2", "batch_size"]

    opt = keras.optimizers.Adagrad()
    attrs = []

    train, valid = load_datasets(TRAIN_DIR, VALID_DIR, batch_size=batch_size)
    tuner = LRTuner(name, opt, loss)

    #tuner.tune(train, valid, min_lr, max_lr, ep, batch_size)
    #tuner.save(TUNING_DIR, name_sv)
    #tuner.plot_loss(opt_attrs=attrs)

    # base15m_adam_lr
    # base15m_sgd_lr
    # base15m_adagrad_tune.json
    #tuner.plot_loss_from_file(f"{TUNING_DIR}/base15m_adagrad_tune.json", loss_lim=(0, 5), useLog=True)

    prefix = ["base15m_adam_lr.json", "base15m_sgd_lr.json", "base15m_adagrad_tune.json"]

    #for val in momentum:
    #    keras.backend.set_value(opt.momentum, val)
    #    tuner.tune(train, valid, min_lr, max_lr, ep, batch_size)
    #    tuner.save(TUNING_PATH, name_sv)
    #    tuner.plot_loss(opt_attrs=["momentum", "nesterov"])
    #    tuner.reset()

    #prefix = ("base15m_sgd_lr*", "base15m_sgd_nst*")
    #prefix = ("base15m_adam_eps*", )


    #lowest = tuner.findFileWithLowestLoss(TUNING_PATH, prefix)
    #for k, v in lowest.items():
    #    print(f"{k}: {v['fn']}")

    tuner.plot_loss_from_files(TUNING_DIR, prefix, loss_lim=(0, 4.6), opt_name_in_label=True,
                               separate_prefix_plots=False, use_log=True, opt_attrs=attrs)


