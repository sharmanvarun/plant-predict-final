from tensorflow import keras
import tensorflow as tf
import itertools
from typing import Tuple, Optional
from dataclasses import dataclass
import numpy as np
import scipy.stats as st
import logging


def calc_predictions(model: keras.Sequential, test_data: tf.data.Dataset) -> Tuple[list, list]:
    """:returns (predictions, class_ids) = (predictions, expectations)"""

    class_ids = []
    predictions = []

    # batch.shape = (batch_size, 256, 256, 3) = (batch_size, img)
    # ids.shape = (batch_size) = class_ids
    for batch, ids in test_data:
        class_ids.append(list(ids.numpy()))

        # cps.shape = (batch_size, 38) = (batch_size, class_probabilities)
        cps = model.predict(batch)
        # p.shape = (batch_size) = (predicted class)
        p = tf.argmax(cps, axis=1).numpy()
        predictions.append(p)

    # Flattening
    predictions = list(itertools.chain.from_iterable(predictions))
    class_ids = list(itertools.chain.from_iterable(class_ids))

    return predictions, class_ids


@dataclass
class CMResult:
    cm: np.ndarray
    n_cls: int
    n_samples: int
    tp: np.ndarray
    tn: np.ndarray
    fp: np.ndarray
    fn: np.ndarray


def confusion_matrix(predictions: list, class_ids: list) -> Optional[CMResult]:
    num_samples = len(predictions)

    if num_samples != len(class_ids):
        print(f"Arrays have different lengths: predictions{num_samples}, class_ids({len(class_ids)})")
        return None

    # num classes
    n = len(np.unique(class_ids))
    # result matrix
    cm = np.zeros((n, n), dtype=np.int)

    for i in range(num_samples):
        cm[class_ids[i]][predictions[i]] += 1

    # TP for each class
    tp = np.diag(cm)
    # FP for each class (sum of rows - TP)
    fp = np.sum(cm, axis=0) - tp
    # Fn for each class (sum of columns - TP)
    fn = np.sum(cm, axis=1) - tp

    # For class i, TN is all samples that isn't of class i,
    # that haven't been classified as class i.
    tn = np.zeros(n)
    for i in range(n):
        tmp = np.delete(cm, i, 0)   # delete class row
        tmp = np.delete(tmp, i, 1)  # delete class column
        tn[i] = tmp.sum()           # sum of the leftover matrix

    return CMResult(cm, n, num_samples, tp, tn, fp, fn)


def accuracy_score(cmr: CMResult) -> float:
    n = cmr.n_samples
    TP = cmr.tp.sum()
    return float(TP) / n


def precision_score(cmr: CMResult, average: str = 'micro') -> float:
    if average == 'micro':
        TP = cmr.tp.sum()
        FP = cmr.fp.sum()
        return float(TP) / (TP + FP)
    elif average == 'macro':
        TP = cmr.tp.astype(np.float)
        FP = cmr.fp
        scores = TP / (TP + FP)
        return scores.sum() / cmr.n_cls
    else:
        raise ValueError(f"'average' is not of type ('micro', 'macro')")


def recall_score(cmr: CMResult, average: str = 'micro') -> float:
    if average == 'micro':
        TP = cmr.tp.sum()
        FN = cmr.fn.sum()
        return float(TP) / (TP + FN)
    elif average == 'macro':
        TP = cmr.tp.astype(np.float)
        FN = cmr.fn
        scores = TP / (TP + FN)
        return scores.sum() / cmr.n_cls
    else:
        raise ValueError(f"'average' are not of supported values ('micro', 'macro')")


def f1_score(cmr: CMResult, average: str = 'micro') -> float:
    if average == 'micro':
        TP = cmr.tp.sum()
        FP = cmr.fp.sum()
        FN = cmr.fn.sum()
        f1 = 2.0 * TP / (2.0 * TP + FP + FN)
    elif average == 'macro':
        TP = cmr.tp.astype(np.float)
        FP = cmr.fp
        FN = cmr.fn
        f1 = 2 * TP / (2 * TP + FP + FN)
        f1 = f1.sum() / cmr.n_cls
    else:
        raise ValueError(f"'average' are not of supported values ('micro', 'macro')")
    return f1


def time_format(sec: float) -> str:
    m, s = divmod(sec, 60)
    h, m = divmod(m, 60)
    return "{:d}:{:02d}:{:02d}".format(round(h), round(m), round(s))


def calc_error_interval(cmr: CMResult, confidence: float = 0.95) -> Tuple[float, float]:
    z = st.norm.ppf(confidence)
    error = 1 - accuracy_score(cmr)
    result = z * np.sqrt((error * (1 - error)) / cmr.n_samples)
    return error, result

