import numpy as np
from load_data import load

def normalize(X_train, Y_train):

    X_train = np.array(X_train)
    X_train = X_train.astype('float32') / 255.0
    Y_train = np.array(Y_train)
    Y_train = Y_train.astype('float32') / 255.0

    return X_train, Y_train