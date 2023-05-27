import tensorflow as tf
from matplotlib import pyplot as plt
import cv2
from tensorflow.keras.layers import Dense, Conv2DTranspose, LeakyReLU, Flatten, BatchNormalization
import numpy as np
from networks import build_discriminator, build_generator
from losses import generator_loss, discriminator_loss
from AdaIN_layer import AdaIN
import os, sys

# Output channels and resolution
output_channels = 3
generator_output_shape = (128, 128, 3)

# Latent code parameters
batch_size = 8; latent_dim = 128

# Build models
generator = build_generator(latent_dim, output_channels)
discriminator = build_discriminator(generator_output_shape, latent_dim)

# Learning rate
learning_rate_gen = 0.0002
learning_rate_disc = 0.00003

# No. of epochs
num_epochs = 50

# Generator and Discriminator Optimizers
generator_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_gen, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_disc, beta_1=0.5)

# Generating latent codes from normal distribution
images_path = os.path.join(os.path.dirname(__file__), "../celebA/img_align_celeba/img_align_celeba/")
train_size = int(0.8*1000)

# Training loop
print("\n\n")
tf.random.set_seed(42)
for epoch in range(num_epochs):
    n = 1   # Image index
    while n <= train_size:
      real_images = []
      for _ in range(min(train_size-n+1, batch_size)):
        image = cv2.imread(images_path+(6-len(str(n)))*'0' + str(n)+".jpg")
        image = cv2.resize(image, (generator_output_shape[0], generator_output_shape[1]))
        real_images.append(image)
        n += 1
      
      real_images = np.array(real_images)
      real_images = real_images.astype('float32') / 127.5 -1
      latent_codes = tf.random.normal([len(real_images), latent_dim], 0, 1, tf.float32)
            
      # Train the discriminator
      with tf.GradientTape() as disc_tape:
          fake_images = generator(latent_codes, training=True)

          real_predictions = discriminator(real_images, training=True)
          fake_predictions = discriminator(fake_images, training=True)
                
          disc_loss = discriminator_loss(real_predictions, fake_predictions)
            
      disc_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
      discriminator_optimizer.apply_gradients(zip(disc_gradients, discriminator.trainable_variables))
            
      # Train the generator
      with tf.GradientTape() as gen_tape:
          fake_images = generator(latent_codes, training=True)

          fake_predictions = discriminator(fake_images, training=True)
                
          gen_loss = generator_loss(fake_predictions)
            
      gen_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
      generator_optimizer.apply_gradients(zip(gen_gradients, generator.trainable_variables))

      # Logging and evaluation
      print('Epoch {} --- Batch {}, Generator Loss: {}, Discriminator Loss: {}'.format(epoch+1, (n//batch_size), gen_loss, disc_loss))

random_codes = tf.random.normal([batch_size, latent_dim], 0, 1, tf.float32)