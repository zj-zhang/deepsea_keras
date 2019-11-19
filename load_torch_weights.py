# -*- coding: utf-8 -*-

"""
Load legacy pytorch weights into Keras
ZZJ, 11.18.2019
"""

import torch
import keras.backend as K
import numpy as np
from model import build_model

weight_setter =  lambda l, v : K.set_value(l.weights[0], v)
bias_setter = lambda l, v: K.set_value(l.weights[1], v)

weight_conversion_map = {
    '0.weight': ('conv1d_1', weight_setter),
    '0.bias': ('conv1d_1', bias_setter),
    '4.weight': ('conv1d_2', weight_setter),
    '4.bias': ('conv1d_2', bias_setter),
    '8.weight': ('conv1d_3', weight_setter),
    '8.bias': ('conv1d_3', bias_setter),
    '12.1.weight': ('dense_1', weight_setter),
    '12.1.bias': ('dense_1', bias_setter),
    '14.1.weight': ('dense_2', weight_setter),
    '14.1.bias': ('dense_2', bias_setter),
}


def read_torch():
    # is a OrderedDict
    weights = torch.load('data/deepsea_cpu.pth')
    for k, v in weights.items():
        weights[k] = np.array(v)
    return weights


def build_and_load_weights():
    model = build_model()
    torch_weights = read_torch()

    layer_dict = {l.name:l for l in model.layers}

    def conv_converter(x):
        x = np.squeeze(np.array(x), 2)
        return np.transpose(x, [2,1,0])

    def dense_converter(x):
        return np.transpose(np.array(x), [1,0])

    for th_lname, (k_lname, func) in weight_conversion_map.items():
        th_w = torch_weights[th_lname]
        if th_lname.endswith('weight'):
            if k_lname.startswith('conv1d'):
                th_w = conv_converter(th_w)
            elif k_lname.startswith('dense'):
                th_w = dense_converter(th_w)
        k_layer = layer_dict[k_lname]
        func(k_layer, th_w)
    return model


def test_performance(model):
    from sklearn.metrics import roc_auc_score, average_precision_score
    from read_data import read_test_data
    test_x, test_y = read_test_data()

    test_pred = model.predict(test_x, batch_size=512, verbose=1)
    evals = np.zeros((919, 2))
    for i in range(919):
        try:
            auroc = roc_auc_score(y_true=test_y[:, i], y_score=test_pred[:, i])
            aupr = average_precision_score(y_true=test_y[:, i], y_score=test_pred[:, i])
            evals[i, :] = [auroc, aupr]
        # only one class present
        except ValueError:
            evals[i, :] = np.nan


