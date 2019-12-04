from model import build_model, load_model
from read_data import read_train_data, read_val_data
from keras.callbacks import ModelCheckpoint, EarlyStopping
import tensorflow as tf
import numpy as np
import keras.backend as K


def reset_model(model, reinit_layers):
    layer_dict = {l.name:l for l in model.layers}
    s = K.get_session()
    init = tf.contrib.layers.variance_scaling_initializer()
    with tf.variable_scope('reset_model', initializer=init, reuse=tf.AUTO_REUSE):
        for layer_name in reinit_layers:
            layer = layer_dict[layer_name]
            new_weights = tf.get_variable(
                shape=layer.get_weights()[0].shape,
                dtype=tf.float32,
                name="nw_%s"%layer_name
            )
            bias = np.zeros(layer.get_weights()[1].shape)
            s.run(tf.initialize_variables([new_weights]))
            layer.set_weights([s.run(new_weights), bias])

def freeze_model(model, freeze_layers):
    layer_dict = {l.name:l for l in model.layers}
    for layer_name in freeze_layers:
        layer_dict[layer_name].trainable = False

model = load_model()

reinit_layers = ['dense_1', 'dense_2']
reset_model(model, reinit_layers)

freeze_layers = ['conv1d_1', 'conv1d_2', 'conv1d_3']
freeze_model(model, freeze_layers)

x_train, y_train = read_train_data()
x_val, y_val = read_val_data()

model.compile(optimizer="adam",
        loss="binary_crossentropy",
        metrics=["acc"])

checkpointer = ModelCheckpoint(filepath="best_deepsea_model.h5", verbose=1, save_best_only=True)
earlystopper = EarlyStopping(monitor="val_loss", patience=5, verbose=1)

model.fit(x_train, y_train,
    epochs=200,
    batch_size=1000,
    verbose=1,
    validation_data=(x_val, y_val),
    callbacks=[checkpointer, earlystopper])
