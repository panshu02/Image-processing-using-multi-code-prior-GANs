import os
import numpy as np
from PIL import Image
from preprocess import normalize
from train import train
from matplotlib import pyplot as plt

X_test_path = "/../DIV2K/DIV2K_valid_LR_bicubic/X4/"
Y_test_path = "/../DIV2K/DIV2K_valid_HR/X4"


# Loading a random image from test data
N = len(os.listdir(X_test_path))
img_ind = np.random.randint(0, N)

X_img_filename = os.listdir(X_test_path)[img_ind]
Y_img_filename = os.listdir(Y_test_path)[img_ind]

X_image = Image.open(os.path.join(X_test_path, X_img_filename))
Y_image = Image.open(os.path.join(Y_test_path, Y_img_filename))

# Normalize to 0 to 1
X_image, Y_image = normalize(X_image, Y_image)

# Epochs and batch size
epochs = 50
batch_size = 8

# Train the model and predict
SRCNN = train(epochs, batch_size)
Y_pred = SRCNN.predict(X_image, batch_size = 1)

# Plot the predicted and orignal image
fig, axs = plt.subplots(1, 2)
axs[0, 1].imshow(Y_pred)
axs[0, 1].xlabel = 'Upscaled image from LR image (ratio = 4X)'
axs[0, 0].imshow(Y_image)
axs[0, 0].xlabel = 'Original HR image'
plt.show()