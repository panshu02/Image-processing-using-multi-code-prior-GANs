from tensorflow.keras import backend as K
from tensorflow.keras.losses import mse, binary_crossentropy
import tensorflow as tf
import os, sys
import numpy as np

import vae
import preprocess


epochs = 50
steps_per_epoch = 100
input_shape = (128, 128, 3)
latent_dim = 128

train_generator = preprocess.train_generator

# Create an instance of the VAE model
vae_2 = vae.VAE(input_shape, latent_dim)

# Compile the model
vae_2.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss = vae_2.compute_loss)
optimizer = tf.keras.optimizers.Adam()

def train_vae(train_generator, epochs, steps_per_epoch, optimizer):
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        print("Training...\n")

        # Number of steps per epoch
        n = 0

        # Initialize total loss for the epoch
        total_loss = 0

        # Iterate over the batches of the training data
        for batch_x, _ in train_generator:

            n += 1

            with tf.GradientTape() as tape:
                # Forward pass through the VAE
                reconstructed = vae_2(batch_x)
                # Compute the reconstruction loss
                loss_value = vae_2.train_step(batch_x)
                # Update the total loss
                total_loss += loss_value

            # Print the batch loss for the epoch
            print(f"Epoch {epoch+1}/{epochs} ---- Batch {n}, Loss: {loss_value:.4f}")

train_vae(train_generator, epochs, steps_per_epoch, optimizer)