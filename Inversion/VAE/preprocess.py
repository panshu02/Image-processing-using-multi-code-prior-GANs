import numpy as np
import os, sys
import cv2

from load_data import get_input, get_output


def preprocess_input(img):
    # Resize image
    img = cv2.resize(img, (128, 128))

    # convert between -1 and 1
    return img.astype('float32') / 127.5 -1

def image_generator(files, label_file, batch_size = 8):
    while True:

        batch_paths = np.random.choice(a = files, size = batch_size)
        batch_input = []
        batch_output = []

        for input_path in batch_paths:

            input = get_input(input_path)
            input = preprocess_input(input)
            output = get_output(input_path, label_file = label_file)
            batch_input += [input]
            batch_output += [output]
        batch_x = np.array(batch_input)
        batch_y = np.array(batch_output)

        yield batch_x, batch_y

def auto_encoder_generator(files, batch_size=8):
    batch_x, batch_y = [], []
    for file in files:
        img = get_input(file)
        img = preprocess_input(img)
        batch_x.append(img)
        batch_y.append(img)

    # Split the data into batches
    num_samples = len(batch_x)
    num_batches = num_samples // batch_size
    batched_data = []
    for i in range(num_batches):
        start = i * batch_size
        end = (i + 1) * batch_size
        batch_x_batch = np.array(batch_x[start:end])
        batch_y_batch = np.array(batch_y[start:end])
        batched_data.append((batch_x_batch, batch_y_batch))

    return batched_data

IMG_NAME_LENGTH = 6
file_path = os.path.join(os.path.dirname(__file__), "../celebA/img_align_celeba/img_align_celeba/")
img_id = np.arange(1,1001)
img_path = []

for i in range(len(img_id)):
    img_path.append(file_path + (IMG_NAME_LENGTH - len(str(img_id[i])))*'0' + str(img_id[i]) + '.jpg')

# pick 80% as training set and 20% as validation set
train_path = img_path[:int((0.8)*len(img_path))]
val_path = img_path[int((0.8)*len(img_path)):]

train_generator = auto_encoder_generator(train_path, 8)
val_generator = auto_encoder_generator(val_path,8)