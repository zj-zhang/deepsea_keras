from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten, Conv1D, MaxPooling1D, Input
from keras import metrics

NUM_OUTPUT = 919
NUM_INPUT = 1000

def build_model():
    inp = Input(shape=(NUM_INPUT, 4))
    x = Conv1D(filters=320, kernel_size=8, activation="relu")(inp)
    x = MaxPooling1D(pool_size=10, strides=10)(x)
    x = Dropout(0.2)(x)
    x = Conv1D(filters=480, kernel_size=8, activation="relu")(x)
    x = MaxPooling1D(pool_size=4, strides=4)(x)
    x = Dropout(0.2)(x)
    x = Conv1D(filters=960, kernel_size=8, activation="relu")(x)
    x = MaxPooling1D(pool_size=4, strides=4)(x)
    x = Dropout(0.2)(x)
    x = Flatten()(x)
    out = Dense(NUM_OUTPUT, activation="sigmoid")(x)
    model = Model(inputs=[inp], outputs=[out])
    model.compile(
        optimizer="adam",
	loss="binary_crossentropy", 
	metrics=["acc"])
    return model
