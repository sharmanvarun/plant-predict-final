import json
import logging
import os
from tensorflow import keras
import numpy as np
from typing import Tuple, Optional, List, Union, Iterable
from dataclasses import dataclass, field
import dataclasses
import glob
from keras.utils.image_dataset import image_dataset_from_directory
from keras.layers import Rescaling
import tensorflow as tf
import warnings

# warnings.filterwarnings("ignore")
tf.compat.v1.set_random_seed(0)
np.random.seed(0)


@dataclass
class TrainParams:
    ep: int
    batch_size: int
    learning_rate: float
    loss: str
    train_time: float
    history: dict


def load_datasets(tp: str, vp: str, batch_size: int = 32) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    # Loading Datasets
    train_gen = image_dataset_from_directory(directory=tp, image_size=(256, 256), batch_size=batch_size)
    valid_gen = image_dataset_from_directory(directory=vp, image_size=(256, 256), batch_size=batch_size)

    # Normalizing Image parameters: [0-255] -> [0-1]
    scale = Rescaling(scale=1.0 / 255)
    train_gen = train_gen.map(lambda img, label: (scale(img), label))
    valid_gen = valid_gen.map(lambda img, label: (scale(img), label))

    return train_gen, valid_gen


def load_datasett(folder: str, img_size: tuple = (256, 256), batch_size=32) -> tf.data.Dataset:
    datasett = image_dataset_from_directory(directory=folder, image_size=img_size, batch_size=batch_size)

    # Normalizing Image parameters: [0-255] -> [0-1]
    scale = Rescaling(scale=1.0 / 255)

    return datasett.map(lambda img, label: (scale(img), label))


def checkDataset(datasett: tf.data.Dataset):
    for i, d in enumerate(datasett):
        print(f"{i}: {np.any(np.isnan(d[0]))}")


def find_file_ver(folder: str, name: str) -> str:
    fn, ext = os.path.splitext(name)

    ver = 1
    fn_new = f"{fn} ({ver}){ext}"

    while True:
        if not os.path.exists(os.path.join(folder, fn_new)):
            break
        ver += 1
        fn_new = f"{fn} ({ver}){ext}"

    return os.path.join(folder, fn_new)


def file_search(search_dir: str, fn: Union[str, Iterable[str]], sort=True, reverse=False, recursive=True,
                include_top_level=True) -> List[str]:
    prefixes = []
    files = []
    sep = os.path.sep

    if type(fn) == str:
        if recursive:
            #if include_top_level:
            #    prefixes.append(f"{search_dir}{sep}{fn}")
            prefixes.append(f"{search_dir}{sep}**{sep}{fn}")
    else:
        if recursive:
            #if include_top_level:
            #    for name in fn:
            #        prefixes.append(f"{search_dir}{sep}{name}")
            #        prefixes.append(f"{search_dir}{sep}**{sep}{name}")
            # else:
            prefixes = [f"{search_dir}{sep}**{sep}{name}" for name in fn]

    for prefix in prefixes:
        files.extend(glob.glob(prefix, recursive=recursive))

    if sort:
        files = list(sorted(files, reverse=reverse))
    return files


def save_model_config(folder: str, name: str, model: keras.Sequential, opt: keras.optimizers.Optimizer,
                      params: TrainParams, overwrite=True):

    data = model.get_config()
    data["opt"] = opt.get_config()
    data["params"] = dataclasses.asdict(params)

    parsed = []
    conf: dict = data["opt"]
    conf["str_float"] = parsed

    for key in conf.keys():
        v = conf[key]
        if type(v) == np.float32:
            parsed.append(key)
            conf[key] = str(v)

    os.makedirs(folder, exist_ok=True)
    fp = os.path.join(folder, name)

    if os.path.isfile(fp) and not overwrite:
        fp = find_file_ver(folder, name)

    with open(fp, mode='w') as f:
        json.dump(data, f, indent=3)


def save_model_params(config_path: str, params: TrainParams):
    if not os.path.isfile(config_path):
        logging.info(f"File doesn't exist: {config_path}")
        return

    with open(config_path, mode='r') as f:
        data: dict = json.load(f)

    data["params"] = dataclasses.asdict(params)

    with open(config_path, mode='w') as f:
        json.dump(data, f, indent=3)


def save_model_weights(folder: str, name: str, model: keras.Sequential, overwrite=True):
    os.makedirs(folder, exist_ok=True)
    fp = os.path.join(folder, name)

    if os.path.exists(fp) and not overwrite:
        fp = find_file_ver(folder, name)

    model.save_weights(fp, overwrite=overwrite)


def save_model(config_sp: str,
               weights_sp: str,
               name: str,
               model: keras.Sequential,
               opt: keras.optimizers.Optimizer,
               params: TrainParams,
               overwrite=False):

    save_model_weights(weights_sp, f"{name}.h5", model, overwrite)
    save_model_config(config_sp, f"{name}.json", model, opt, params, overwrite)


def save_to_json(fp: str, data: Union[dict, list], indent: int = 3, overwrite=False):
    folder = os.path.dirname(fp)
    os.makedirs(folder, exist_ok=True)

    if ".json" not in fp:
        fp = f"{fp}.json"

    if os.path.isfile(fp) and not overwrite:
        fp = find_file_ver(folder, os.path.basename(fp))

    with open(fp, mode='w') as f:
        json.dump(data, f, indent=indent)


def load_json_files(paths: Iterable[str]) -> list:
    data = []
    for path in paths:
        with open(path, mode='r') as f:
            data.append(json.load(f))
    return data


def load_json_file(fp: str) -> Union[list, dict]:
    if not os.path.exists(fp):
        folder, fn = os.path.split(fp)
        file = file_search(folder, fn)
        if len(file) > 1:
            logging.warning(f"Multiple files found, selecting {file[0]}")
            fp = file[0]
        elif len(file) == 1:
            fp = file[0]
        else:
            return []

    with open(fp, mode='r') as f:
        return json.load(f)


def model_from_config(config_path: str, weights_path: str = None) -> Tuple[keras.Sequential, TrainParams]:

    if not os.path.exists(config_path):
        root, fn = os.path.split(config_path)
        paths = file_search(root, fn)

        if len(paths) > 1:
            logging.warning(f"Multiple files found, selecting {paths[0]}")
            config_path = paths[0]
        elif not len(paths):
            raise FileNotFoundError(f"Specified file doesn't exist: {config_path}")

    with open(config_path, mode='r') as f:
        data: dict = json.load(f)

    params = data.pop("params")
    params = TrainParams(**params)
    opt_conf = data.pop("opt")
    model = keras.Sequential.from_config(data)

    for key in opt_conf["str_float"]:
        opt_conf[key] = np.float32(opt_conf[key])
    opt_conf.pop("str_float")

    opt = getattr(keras.optimizers, opt_conf["name"], None)

    if opt is None:
        logging.warning(f"Failed to load optimizer {opt_conf['name']}")
    else:
        opt = opt(**opt_conf)
        model.compile(optimizer=opt, loss=params.loss, metrics=['accuracy'])

    if weights_path is not None:
        if not os.path.exists(weights_path):
            root, fn = os.path.split(weights_path)
            paths = file_search(root, fn)

            if not len(paths):
                logging.warning(f"Failed to load weights from  path {weights_path}")
                return model, params
            elif len(paths) > 1:
                logging.warning(f"Multiple files found, selecting {paths[0]}")
                weights_path = paths[0]
            else:
                weights_path = paths[0]

        model.load_weights(weights_path)
    else:
        logging.warning(f"Failed to load weights from  path {weights_path}")

    return model, params