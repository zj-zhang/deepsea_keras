# -*- coding: utf-8 -*-

from keras.models import Model
from keras.models import load_model as keras_load_model
from keras.layers import Dense, Dropout, Activation, Flatten, Conv1D, MaxPooling1D, Input, Lambda
from keras import metrics

NUM_OUTPUT = 919
DENSE_UNITS = 925
NUM_INPUT = 1000


def build_model():
    inp = Input(shape=(NUM_INPUT, 4))
    x = Conv1D(filters=320, kernel_size=8, activation="relu", name="conv1d_1")(inp)
    x = MaxPooling1D(pool_size=4, strides=4)(x)
    x = Dropout(0.2)(x)
    x = Conv1D(filters=480, kernel_size=8, activation="relu", name="conv1d_2")(x)
    x = MaxPooling1D(pool_size=4, strides=4)(x)
    x = Dropout(0.2)(x)
    x = Conv1D(filters=960, kernel_size=8, activation="relu", name="conv1d_3")(x)
    x = Dropout(0.5)(x)
    x = Flatten(data_format="channels_first")(x)
    x = Dense(DENSE_UNITS, activation="relu", name="dense_1")(x)
    out = Dense(NUM_OUTPUT, activation="sigmoid", name="dense_2")(x)
    model = Model(inputs=[inp], outputs=[out])
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["acc"])
    return model


def load_model():
    model = keras_load_model('./data/deepsea_keras.h5')
    model.summary()
    return model


def convert_to_multigpu(
        vanilla_model,
        gpus=4,
        model_compile_dict=None,
        **kwargs):
    try:
        from keras.utils import multi_gpu_model
    except Exception as e:
        raise Exception("Exception %s\nmulti gpu not supported in keras. check your version."%e)
    model = multi_gpu_model(vanilla_model, gpus=gpus, **kwargs)
    if model_compile_dict is None:
        model_compile_dict = {
            "optimizer": "adam",
            "loss": "binary_crossentropy",
            "metrics": ["acc"]
        }
    model.compile(**model_compile_dict)
    return model
