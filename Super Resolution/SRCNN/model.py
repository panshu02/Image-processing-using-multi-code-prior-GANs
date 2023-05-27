from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Activation, BatchNormalization, Add, UpSampling2D
from keras.optimizers import Adam


def model():
    # Define model type
    SRCNN = Sequential()

    # Add model layers
    SRCNN.add(Conv2D(filters=128, kernel_size=(9, 9), kernel_initializer='glorot_uniform',
                     use_bias=True, input_shape=(None, None, 3)))
    SRCNN.add(BatchNormalization())
    SRCNN.add(Activation('relu'))

    SRCNN.add(Conv2D(filters=64, kernel_size=(1, 1), kernel_initializer='glorot_uniform',
                     use_bias=True))
    SRCNN.add(BatchNormalization())
    SRCNN.add(Activation('relu'))

    SRCNN.add(Conv2D(filters=32, kernel_size=(5, 5), kernel_initializer='glorot_uniform',
                     use_bias=True))
    SRCNN.add(BatchNormalization())
    SRCNN.add(Activation('relu'))

    SRCNN.add(Conv2D(filters=3, kernel_size=(5, 5), kernel_initializer='glorot_uniform',
                     use_bias=True))
    SRCNN.add(BatchNormalization())
    SRCNN.add(Activation('sigmoid'))

    SRCNN.add(UpSampling2D(size=(2, 2)))
    SRCNN.add(Conv2D(filters = 3, kernel_size = (3, 3), padding = 'same'))
    SRCNN.add(BatchNormalization())
    SRCNN.add(UpSampling2D(size=(2, 2)))
    SRCNN.add(Conv2D(filters = 3, kernel_size = (3, 3), padding = 'same'), activation = 'sigmoid')

    # Define optimizer
    adam = Adam(lr=0.0003)

    # Compile model
    SRCNN.compile(optimizer=adam, loss='mean_squared_error', metrics=['mean_squared_error'])

    return SRCNN
