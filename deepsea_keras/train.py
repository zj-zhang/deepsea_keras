from .model import build_model
from .read_data import read_train_data, read_val_data
from keras.callbacks import ModelCheckpoint, EarlyStopping


def train_model():
    model = build_model()
    x_train, y_train = read_train_data()
    x_val, y_val = read_val_data()

    checkpointer = ModelCheckpoint(filepath="best_deepsea_model.h5", verbose=1, save_best_only=True)
    earlystopper = EarlyStopping(monitor="val_loss", patience=5, verbose=1)

    model.fit(x_train, y_train,
              epochs=200,
              batch_size=1000,
              verbose=1,
              validation_data=(x_val, y_val),
              callbacks=[checkpointer, earlystopper])
    return model
