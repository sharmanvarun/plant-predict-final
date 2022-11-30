from src.utils.io import model_from_config, TrainParams, load_datasett, file_search
from src.utils.metrics import calc_predictions, CMResult, calc_error_interval
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score
from src.utils import drawing as draw
import tensorflow as tf
from tensorflow import keras
from src.utils.metrics import time_format
from keras.utils.image_dataset import image_dataset_from_directory
from keras.layers import Rescaling
from src.utils.paths import CONFIGS_DIR, WEIGHTS_DIR, TRAIN_DIR, VALID_DIR, TEST_DIR
import src.utils.metrics as metrics
from typing import List, Tuple
import os
import re
import numpy as np
import matplotlib.pyplot as plt


def print_model_data(model: keras.Sequential, params: TrainParams, start: str = ''):
    opt = model.optimizer.get_config()
    if opt['name'] == 'Adam':
        ofo = f"Adam(b1={np.round(opt['beta_1'], 3)} b2={np.round(opt['beta_2'], 3)} eps={opt['epsilon']})"
    elif opt['name'] == "Adagrad":
        ofo = f"Adagrad(eps={opt['epsilon']}, iav={opt['initial_accumulator_value']})"
    elif opt['name'] == 'SGD':
        ofo = f"SGD(momentum={np.round(opt['momentum'], 3)})"
    else:
        ofo = None
    print(f"{start} lr={params.learning_rate}, batch_size={params.batch_size}, {ofo}")


def model_test(model: keras.Sequential, params: TrainParams, test_data: tf.data.Dataset, y_lim: float = None,
               interval_confidence: float = 0.95, fn: str = None, plot: str = 'train', show_cmr=False):

    predictions, labels = calc_predictions(model, test_data)
    cmr = metrics.confusion_matrix(predictions, labels)

    print(f"Name: {fn}")
    print(f"Time to Train  : {time_format(params.train_time)}")
    print(f"Train Accuracy  : {round(params.history['accuracy'][-1] * 100, 2)}")

    # sklearn functions
    # test_acc = accuracy_score(labels, predictions) * 100
    # print(f"Test Accuracy   : {round(test_acc, 2)}")
    # ! sklearn functions
    print(f"Test Accuracy (own)   : {round(metrics.accuracy_score(cmr) * 100, 2)}\n")

    print_model_data(model, params, start=f"\n{fn}:")

    mean, interval = calc_error_interval(cmr, interval_confidence)
    mean, interval = np.round(mean, 3), np.round(interval, 3)
    print(f"\nInterval: {mean} +/- {interval} -> [{np.round(mean - interval, 3)}, {np.round(mean +  interval, 3)}]")

    if plot == 'train':
        if show_cmr:
            draw.training_plot(params.history, params.ep, show=False, y_loss_max=y_lim)
            draw.confusion_matrix(cmr.cm, show=True)
        else:
            draw.training_plot(params.history, params.ep, show=True, y_loss_max=y_lim)
    elif plot == 'loss':
        if show_cmr:
            draw.training_loss_plots([params.history], [fn], params.ep, y_max=y_lim, show=False)
            draw.confusion_matrix(cmr.cm, show=True)
        else:
            draw.training_loss_plots([params.history], [fn], params.ep, y_max=y_lim, show=True)


def print_max_acc_of_old(test_data: tf.data.Dataset):
    cf = file_search(os.path.join(CONFIGS_DIR, "old"), "base15m_adam*")
    vf = file_search(os.path.join(WEIGHTS_DIR, "old"), "base15m_adam*")
    names = []
    cmrs: List[CMResult] = []
    accs = []

    for i in range(len(cf)):
        model, ps = model_from_config(cf[i], vf[i])
        predictions, class_ids = calc_predictions(model, test_data)

        cmrs.append(metrics.confusion_matrix(predictions, class_ids))
        names.append(os.path.basename(cf[i]))

    best = 0
    best_fn = None

    for i, name in enumerate(names):
        acc = metrics.accuracy_score(cmrs[i])
        accs.append(acc)

        if acc > best:
            best = acc
            best_fn = name

        print(f"{name} Accuracy: {round(acc * 100, 2)}%")

    print(f"\nMost accurate: {best_fn} = {round(best * 100, 2)}")


def print_models_acc(names: List[str], config_dir: str, model_dir: str, test_data: tf.data.Dataset):
    models = []
    params = []
    cmrs: List[CMResult] = []

    for name in names:
        model, ps = model_from_config(f"{config_dir}/{name}.json", f"{model_dir}/{name}.h5")
        models.append(model)
        params.append(ps)

        predictions, class_ids = calc_predictions(model, test_data)
        cmrs.append(metrics.confusion_matrix(predictions, class_ids))

    for i, name in enumerate(names):
        print(f"{name} Accuracy: {round(metrics.accuracy_score(cmrs[i]) * 100, 2)}%")


def models_train_plot(names: List[str], config_dir: str, model_dir: str, test_data: tf.data.Dataset,
                      labels: List[str] = None, y_min: float = None):
    if labels is None:
        labels = names

    models = []
    params = []

    for name in names:
        model, ps = model_from_config(f"{config_dir}/{name}.json", f"{model_dir}/{name}.h5")
        models.append(model)
        params.append(ps)

    hists = tuple(map(lambda ps: ps.history, params))
    epoch = max(map(lambda ps: ps.ep, params))

    draw.training_loss_plots(hists, labels, epoch, y_min)


def models_train_plot_for_opts(names: List[str], config_dir: str, model_dir: str, test_data: tf.data.Dataset,
                               y_min=None,
                               custom_labels: List[str] = None, add_param_data_to_labels=True):
    models = []
    params = []
    labels = []
    cmrs: List[CMResult] = []

    for i, name in enumerate(names):
        model, ps = model_from_config(f"{config_dir}/{name}.json", f"{model_dir}/{name}.h5")
        opt = model.optimizer.get_config()
        models.append(model)
        params.append(ps)

        predictions, class_ids = calc_predictions(model, test_data)
        cmrs.append(metrics.confusion_matrix(predictions, class_ids))

        # Remove version from names
        # label = re.sub("([\(]).*?([\)])", "", name)
        label = custom_labels[i] if custom_labels is not None else name

        if add_param_data_to_labels:
            if opt['name'] == 'Adam':
                label = f"{label}: lr={round(float(opt['learning_rate']), 3)}, b1={round(float(opt['beta_1']), 3)}, b2={round(float(opt['beta_2']), 3)}"  # , eps={opt['epsilon']}
            elif opt['name'] == "Adagrad":
                label = f"{label}: lr={round(float(opt['learning_rate']), 3)}, eps={opt['epsilon']}, iav={opt['initial_accumulator_value']}"
            elif opt['name'] == 'SGD':
                label = f"{label}: lr={round(float(opt['learning_rate']), 3)}, momentum={round(float(opt['momentum']), 3)}"
            else:
                label = f"{label}: size={ps.batch_size} rate={ps.learning_rate}"

        # default = f"{label}: size={ps.batch_size} rate={ps.learning_rate}"
        # adam_tuning = f"{label}: b1={np.round(opt['beta_1'],3)} b2={np.round(opt['beta_2'],3)} eps={opt['epsilon']}"
        # adagrad_tuning = f"{label}: lr={round(opt['learning_rate'], 5)}, eps={opt['epsilon']}, iav={opt['initial_accumulator_value']}"
        # sdg_tuning = f"{label}: lr={round(opt['learning_rate'], 5)}, momentum={opt['momentum']}"
        # label = adagrad_tuning

        labels.append(label)

    for i, name in enumerate(names):
        print(f"{name} Train Accuracy  : {round(params[i].history['accuracy'][-1] * 100, 2)}")
        print(f"{name} Accuracy        : {round(metrics.accuracy_score(cmrs[i]) * 100, 2)}%")

    hists = tuple(map(lambda ps: ps.history, params))
    epoch = max(map(lambda ps: ps.ep, params))
    draw.training_plots(hists, labels, epoch, y_min)


def get_dir_names(folder: str) -> List[str]:
    root, dirs, files = next(os.walk(folder))
    return list(sorted(dirs))


def test_on_1_img(name: str, test_data: tf.data.Dataset):
    model, params = model_from_config(f"{CONFIGS_DIR}/{name}.json", f"{WEIGHTS_DIR}/{name}.h5")
    label_names = get_dir_names(VALID_DIR)

    for batch, ids in valid_gen:
        img, cls_id = batch[0], ids[0].numpy()
        res = model.predict(batch)

        print("result = \n", res[0])
        print("\nlen(result)  =", len(res[0]))
        print("argmax(result) =", np.argmax(res[0]))

        plt.imshow(img)
        plt.title(f"class: {label_names[cls_id]} (id = {cls_id})")
        plt.show()

        exit(0)


if __name__ == "__main__":
    valid_gen = load_datasett(VALID_DIR, batch_size=32)
    # test_gen = load_datasett(TEST_DIR)

    #name = "base15m_adagrad_tune (19)"
    #model, params = model_from_config(os.path.join(CONFIGS_DIR, f"{name}.json"), os.path.join(WEIGHTS_DIR, f"{name}.h5"))
    #model.summary()
    #model_test(model, params, valid_gen)
    #test_on_1_img(name, valid_gen)

    # Best models to plot:
    names = [
        "base15m_sgd (1)",
        "base15m_sgd (4)",
        "base15m_sgd_tune",
        "base15m_sgd_tune (1)",
        "base15m_sgd_tune (10)",
        "base15m_sgd_tune (11)",
        "base15m_sgd_tune (12)",
        "base15m_sgd_tune (13)",
        "base15m_sgd_tune (14)",
        "base15m_sgd_tune (15)",

        "base15m_adagrad_tune (10)",
        "base15m_adagrad_tune (11)",
        "base15m_adagrad_tune (13)",
        "base15m_adagrad_tune (15)",
        "base15m_adagrad_tune (16)",
        "base15m_adagrad_tune (19)",
        "base15m_adagrad_tune (2)",
        "base15m_adagrad_tune (3)",
        "base15m_adagrad_tune (9)",
        "base15m_adam_tune (1)",
        "base15m_adam_tune",
        "base15m_adam_tune_new (1)",
        "base15m_adam_tune_new (3)",
        "base15m_adam_tune_new (5)",
        "base15m_adam_tune_new (6)",
        "base15m_adam_tune_new (7)",
        "base15m_adam_tune_new",

        # base15m_adam (6)",
        # base15m_adam",
        # base15m_adam_tune (5)",
        # base15m_adam_tune_b (2)",
        # base15m_adam_tune_b (4)"
    ]

    #names = [
    #    "base15m", "base2m", "base15m_adam_tune (3)", "base15m_sgd (4)", "base15m_adagrad_tune (19)"
    #]

    # names = ["base15m_15ep", "base15m", "base15m_50ep"]  # "base2m_15ep", "base2m", "base2m_50ep" ,

    #for name in names:
    #    model, params = model_from_config(os.path.join(CONFIGS_DIR, f"{name}.json"), os.path.join(WEIGHTS_DIR, f"{name}.h5"))
    #    #models_train_plot(names, CONFIGS_DIR, WEIGHTS_DIR, valid_gen, y_min=0.25)
    #    model_test(model, params, valid_gen, fn=name)


    # name = "base15m_adam_tune (3)"
    # model, params = model_from_config(os.path.join(os.path.join(CONFIGS_DIR, "old"), f"{name}.json"),
    #                                  os.path.join(os.path.join(WEIGHTS_DIR, "old"), f"{name}.h5"))
    # model_test(model, params, valid_gen)

    # names = ["base2m", "deep2m", "shallow2m", "spatial2m", "base15m", "deep15m", "shallow15m", "spatial15m"]
    # names = ["base2m", "spatial2m", "base15m", "spatial15m"]
    # names = ["base15m", "base15m_adam", "base15m_adam (1)", "base15m_adam (2)", "base15m_adam (3)"]
    # names = ["base15m", "base15m_adam", "base15m_adam (1)", "base15m_adam (2)", "base15m_adam (3)", "base15m_adam (4)",
    #         "base15m_adam (5)", "base15m_adam (6)", "base15m_adam (7)"]
    # names = ["base15m", "base15m_adam (1)", "base15m_adam (2)", "base15m_adam (4)"]
    # names = ["base15m_adam_tune", "base15m_adam_tune (1)", "base15m_adam_tune (2)", "base15m_adam_tune (2)",
    #         "base15m_adam_tune (3)", "base15m_adam_tune (4)", "base15m_adam_tune (5)", "base15m_adam_tune (6)",
    #         "base15m_adam_tune (7)", "base15m_adam_tune (8)"]
    # names = ["base15m_adam_tune (8)", "base15m_adam_tune_b", "base15m_adam_tune_b (1)", "base15m_adam_tune_b (2)", "base15m_adam_tune_b (3)",
    #         "base15m_adam_tune_b (4)", "base15m_adam_tune_b (5)", "base15m_adam_tune_b (6)", "base15m_adam_tune_b (7)"]

    # names = ["base15m_adagrad_tune (2)", "base15m_adagrad_tune (1)"]

    # names = ["base2m_avg", "deep2m", "fred2m", "shallow2m", "deep15m", "fred15m", "shallow15m", "noDrop15m"]

    # names = ["base15m_adam_tune", "base15m_adam_tune (1)", "base15m_adam_tune_new",
    #         "base15m_adam_tune_new (1)", "base15m_adam_tune_new (2)", "base15m_adam_tune_new (3)",
    #         "base15m_adam_tune_new (4)", "base15m_adam_tune_new (5)", "base15m_adam_tune_new (6)",
    #         "base15m_adam_tune_new (7)"]

    #names = ["base15m_adam_tune (3)", "base15m_sgd (4)", "base15m_adagrad_tune (19)"]

    names = [
        "base15m_adam_tune_b", "base15m_adam_tune_b (2)", "base15m_adam_tune_b (3)",
        "base15m_adam_tune_b (5)", "base15m_adam_tune_b (6)" ]

    #names = [
    #    "base15m_adagrad_tune (1)", "base15m_adagrad_tune (4)", "base15m_adagrad_tune (5)",
    #    "base15m_adagrad_tune (6)", "base15m_adagrad_tune (7)", "base15m_adagrad_tune (8)",
    #    "base15m_adagrad_tune (9)"
    #]

    # base15m_sgd* , base15m_adagrad_tune*, base15m_adam*
    # names = file_search(f"{CONFIGS_DIR}", ("base15m_adagrad_tune*"), sort=True)

    #for i in range(len(names)):
    #    f = names[i]
    #    fn, ext = os.path.splitext(os.path.basename(f))
    #    names[i] = fn

    #data: List[Tuple[keras.Sequential, TrainParams]] = []

    #for name in names:
    #    model, params = model_from_config(f"{CONFIGS_DIR}/{name}.json", f"{WEIGHTS_DIR}/{name}.h5")
    #    data.append((model, params))

    #print()

    #for i, (model, params) in enumerate(data):
    #    if (params.ep == 30) and (params.batch_size == 32):
    #        model_test(model, params, valid_gen, y_lim=None, fn=names[i])

    models_train_plot_for_opts(names, f"{CONFIGS_DIR}/old", f"{WEIGHTS_DIR}/old", valid_gen, y_min=0.875)
    # models_train_plot(names, CONFIGS_DIR, WEIGHTS_DIR, valid_gen, y_min=0.90)
    # print_models_acc(names, CONFIGS_PATH, WEIGHTS_PATH, valid_gen)
    # print_max_acc_of_old(valid_gen)



    # Best tuned Models:
        # base15m_adam_tune (3):     97.79%
        # base15m_sgd (4):           97.63%
        # base15m_adagrad_tune (19): 98.49%

    # Final Best Models to Plot:
        # base15m_sgd_tune(10)
        # base15m_adam_tune

        # base15m_sgd_tune (11)
        # base15m_sgd_tune (12)
        # base15m_adagrad_tune (19)
        # base15m_adam_tune (1)
        # base15m_adam_tune
        # base15m_sgd_tune (8)
