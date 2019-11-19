# -*- coding: utf8 -*-

"""Read in deepsea training data in python
ZZJ
2019.3.6
"""

import h5py
from scipy.io import loadmat
import numpy as np

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
