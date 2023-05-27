import os
from PIL import Image


def load():
    X_train_path = "/../DIV2K/DIV2K_train_LR_bicubic/X4/"
    Y_train_path = "/../DIV2K/DIV2K_train_HR/X4"

    for X_filename in os.listdir(X_train_path):
        X_train = []
        
        if X_filename.endswith(".png") or X_filename.endswith(".jpg"):
                image = Image.open(os.path.join(X_train_path, X_filename))
                X_train.append(image)
        
    for Y_filename in os.listdir(Y_train_path):
        Y_train = []
        
        if Y_filename.endswith(".png") or Y_filename.endswith(".jpg"):
                image = Image.open(os.path.join(Y_train_path, Y_filename))
                Y_train.append(image)

    return X_train, Y_train