import tensorflow as tf
from tensorflow.keras import layers, models
# from tensorflow.keras import backend as K
import tensorflow as tf
import numpy as np
from skimage.io import imread

from load_data import get_input
import preprocess

img_path = preprocess.img_path
img_sample = get_input(img_path[np.random.randint(0, len(img_path))])

def sampling(args):
    z_mean, z_log_sigma = args
    batch_size = tf.shape(z_mean)[0]
    bottleneck_size = tf.shape(z_mean)[1]
    epsilon = tf.random.normal(shape=(batch_size, bottleneck_size))
    return z_mean + tf.exp(0.5 * z_log_sigma) * epsilon

class VAE(tf.keras.Model):
    def __init__(self, input_shape, bottleneck_size):
        super(VAE, self).__init__()
        self.encoder, self.decoder = self.build_conv_vae(input_shape, bottleneck_size)
        self.bottleneck_size = bottleneck_size
    
    def build_conv_vae(self, input_shape, bottleneck_size):
        # ENCODER
        inputs = tf.keras.Input(shape=input_shape)
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        
        # Latent Variable Calculation
        shape = tf.keras.backend.int_shape(x)
        flatten = layers.Flatten()(x)
        z_mean = layers.Dense(bottleneck_size, name='z_mean')(flatten)
        z_mean = layers.BatchNormalization()(z_mean)
        z_log_sigma = layers.Dense(bottleneck_size, name='z_log_sigma')(flatten)
        z_log_sigma = layers.BatchNormalization()(z_log_sigma)
        z = layers.Lambda(sampling)([z_mean, z_log_sigma])
        encoder = tf.keras.Model(inputs, [z_mean, z_log_sigma, z], name='encoder')

        # DECODER
        latent_input = tf.keras.Input(shape=(bottleneck_size,), name='decoder_input')
        x = layers.Dense(shape[1] * shape[2] * shape[3])(latent_input)
        x = layers.Reshape((shape[1], shape[2], shape[3]))(x)
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.Conv2DTranspose(256, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.Conv2DTranspose(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.Conv2DTranspose(32, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        output = layers.Conv2DTranspose(3, (3, 3), activation='tanh', padding='same')(x)
        decoder = tf.keras.Model(latent_input, output, name='decoder')

        return encoder, decoder
    
    def call(self, inputs):
        z_mean, z_log_sigma, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        return reconstructed
    
    def compute_loss(self, inputs, reconstructed):
        _, z_log_sigma, z_mean = self.encoder(inputs)
        reconstruction_loss = tf.reduce_mean(tf.square(inputs - reconstructed))
        kl_loss = -0.5 * tf.reduce_mean(1 + z_log_sigma - tf.square(z_mean) - tf.exp(z_log_sigma))
        total_loss = reconstruction_loss + kl_loss
        return total_loss


    def train_step(self, inputs):
        with tf.GradientTape() as tape:
            reconstructed = self(inputs, training=True)
            loss = self.compute_loss(inputs, reconstructed)
        
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        return loss
