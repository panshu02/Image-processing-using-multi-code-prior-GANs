import tensorflow as tf
import numpy as np
from PIL import Image
import train

# Load the LR image
lr_image_path = '/../DIV2K/DIV2K_valid_LR_bicubic/X4/0801x4.png'
lr_image = Image.open(lr_image_path)
lr_image = lr_image.resize((96, 96))
lr_image = np.array(lr_image) / 255.0  # Normalize the LR image

# Generate a random latent code
latent_dim = 64
n_codes = 10

tf.random.set_seed(42)
latent_code = tf.random.normal([1, n_codes, latent_dim])

# Reshape and expand dimensions of the LR image and latent code
lr_image = np.expand_dims(lr_image, axis=0)
latent_code = np.expand_dims(latent_code, axis=0)

# Load the generator and discriminator
generator = train.generator
discriminator = train.discriminator

# Generate the HR image using the generator
generated_hr_image = generator.predict([lr_image, latent_code])
generated_hr_image = np.squeeze(generated_hr_image, axis=0)

# Scale the pixel values back to the range [0, 255]
generated_hr_image = (generated_hr_image) * 255.0
generated_hr_image = generated_hr_image.astype(np.uint8)

# Display the LR and HR images
lr_image = Image.fromarray(lr_image[0])
hr_image = Image.fromarray(generated_hr_image)
lr_image.show(title="LR Image")
hr_image.show(title="Generated HR Image")
