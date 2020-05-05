# -*- coding: utf-8 -*-

from sklearn.metrics import roc_auc_score, average_precision_score
from read_data import read_test_data, read_val_data
import numpy as np


def get_evals(label, pred):
    evals = {
        'auroc': np.zeros(919),
        'aupr': np.zeros(919),
    }
    for i in range(919):
        try:
            auroc = roc_auc_score(y_true=label[:, i], y_score=pred[:, i])
            aupr = average_precision_score(y_true=label[:, i], y_score=pred[:, i])
            evals['auroc'][i] = auroc
            evals['aupr'][i] =  aupr
        # only one class present
        except ValueError:
            evals['auroc'][i] = np.nan
            evals['aupr'][i] = np.nan
    return evals


def test_performance(model):
    test_x, test_y = read_test_data()

    test_pred = model.predict(test_x, batch_size=512, verbose=1)
    evals = get_evals(test_y, test_pred)
    return evals


def val_performance(model):
    val_x, val_y = read_val_data()

    val_pred = model.predict(val_x, batch_size=500, verbose=1)
    evals = get_evals(val_y, val_pred)
    return evals

