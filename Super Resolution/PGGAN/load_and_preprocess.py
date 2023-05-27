import tensorflow as tf
import os

# Define the paths to the LR and HR image directories
lr_image_dir = '/../DIV2K/DIV2K_train_LR_bicubic/X4'
hr_image_dir = '/../DIV2K/DIV2K_train_HR'

# Function to load and preprocess the LR and HR images
def load_and_preprocess(lr_image_path, hr_image_path):
    # Load LR image
    lr_image = tf.io.read_file(lr_image_path)
    lr_image = tf.image.decode_image(lr_image, channels=3)
    lr_image = tf.image.resize(lr_image, [96, 96])
    
    # Load HR image
    hr_image = tf.io.read_file(hr_image_path)
    hr_image = tf.image.decode_image(hr_image, channels=3)
    
    return lr_image, hr_image

# Function to preprocess the LR and HR images dataset
def preprocess_dataset(lr_image_paths, hr_image_paths):
    # Create a dataset from the LR and HR image paths
    dataset = tf.data.Dataset.from_tensor_slices((lr_image_paths, hr_image_paths))
    
    # Load and preprocess the LR and HR images
    dataset = dataset.map(load_and_preprocess)
    
    return dataset

# Function to resize HR images to 4X of LR images
def resize_hr(lr_image, hr_image):
    hr_image = tf.image.resize(hr_image, [4 * lr_image.shape[0], 4 * lr_image.shape[1]])
    return lr_image, hr_image

# Get the list of LR and HR image paths
lr_image_paths = [os.path.join(lr_image_dir, filename) for filename in os.listdir(lr_image_dir)]
hr_image_paths = [os.path.join(hr_image_dir, filename) for filename in os.listdir(hr_image_dir)]

# Preprocess the LR and HR images dataset
dataset = preprocess_dataset(lr_image_paths, hr_image_paths)

def make_batches(data, batch_size):
    # Apply resizing to the dataset
    data = data.map(resize_hr)

    # Shuffle and batch the dataset
    data = data.shuffle(buffer_size=1000)
    data = data.batch(batch_size)

    # Prefetch the dataset for better performance
    data = data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return data