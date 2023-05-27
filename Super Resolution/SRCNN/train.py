from model import model
from load_data import load
from preprocess import normalize

epochs = 50
batch_size = 8

srcnn = model()
X_train, Y_train = load()
X_train, Y_train = normalize(X_train, Y_train)

srcnn.fit(X_train, Y_train, epochs = epochs, batch_size = batch_size, verbose = '2')