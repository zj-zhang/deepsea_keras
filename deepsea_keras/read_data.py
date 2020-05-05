# -*- coding: utf8 -*-

"""Read in deepsea training data in python
ZZJ
2019.3.6
"""

import h5py
from scipy.io import loadmat
import numpy as np
import pandas as pd


def read_train_data():
    f = h5py.File("./data/train.mat", "r")
    print(list(f.keys()))
    y = f['traindata'].value
    x = f['trainxdata'].value
    x = np.moveaxis(x, -1, 0)
    y = np.moveaxis(y, -1, 0)
    return x, y


def read_val_data():
   f = loadmat("./data/valid.mat")
   print(list(f.keys()))
   x = f['validxdata']
   y = f['validdata']
   x = np.moveaxis(x, 1, -1)
   return x, y


def read_test_data():
   f = loadmat("./data/test.mat")
   print(list(f.keys()))
   x = f['testxdata']
   y = f['testdata']
   x = np.moveaxis(x, 1, -1)
   return x, y


def read_label_annot():
    label_annot = pd.DataFrame(columns=[
        'index',
        'cell',
        'target',
        'condition',
        'category'
        ])

    def category_classifier(t):
        if t.startswith(('H2', 'H3', 'H4')):
            return 'Histone'
        elif t=='DNase':
            return 'DNase'
        elif t.startswith('Pol'):
            return 'Pol'
        else:
            return 'TF'

    with open("./resources/label_names.txt", "r") as f:
        i = 0
        for line in f:
            c, t, d = line.strip().split('|')
            cat = category_classifier(t)
            label_annot = label_annot.append(pd.Series({
                'index':i,
                'cell':c,
                'target':t,
                'condition':d,
                'category': cat}), ignore_index=True)
            i += 1
    return label_annot