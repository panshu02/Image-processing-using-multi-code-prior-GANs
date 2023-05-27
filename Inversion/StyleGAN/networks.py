from AdaIN_layer import AdaIN
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Conv2DTranspose, LeakyReLU, Flatten, BatchNormalization, Conv2D

def build_generator(latent_dim, output_channels):
    input_latent_code = tf.keras.layers.Input(shape=(latent_dim,))
    latent_code = input_latent_code
    x = tf.keras.layers.Dense(4 * 4 * latent_dim, use_bias=False)(input_latent_code)
    x = LeakyReLU(alpha = 0.2)(x)
    x = tf.keras.layers.Dense(4 * 8 * latent_dim, use_bias=True)(x)
    x = LeakyReLU(alpha = 0.2)(x)
    x = tf.keras.layers.Dense(8 * 4 * latent_dim, use_bias=True)(x)
    x = LeakyReLU(alpha = 0.2)(x)
    x = tf.keras.layers.Dense(8 * 4 * latent_dim, use_bias=True)(x)
    x = LeakyReLU(alpha = 0.2)(x)
    x = tf.keras.layers.Dense(4 * 4 * latent_dim, use_bias=False)(x)
    x = LeakyReLU(alpha = 0.2)(x)
    x = tf.keras.layers.Reshape((4, 4, latent_dim))(x)
    
    n = 1
    filters = latent_dim // 2
    while n <= 5:
        x = Conv2DTranspose(filters=filters, kernel_size=3, strides=1, padding='same')(x)
        target_shape = (x.shape[1],x.shape[2], filters)
        latent_code = tf.keras.layers.Dense(np.prod(target_shape), use_bias=False)(latent_code)
        latent_code = tf.keras.layers.Reshape(target_shape)(latent_code)
        x = AdaIN()([x, latent_code])
        latent_code = Flatten()(latent_code)

        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        x = tf.keras.layers.UpSampling2D(size=(2, 2))(x)  # Upsample the feature map
        filters //= 2
        n += 1
      
    x = Conv2DTranspose(filters = output_channels, kernel_size = 3, strides = 1, padding  = 'same')(x)
    
    output = tf.keras.layers.Activation('tanh')(x)

    return tf.keras.Model(inputs=input_latent_code, outputs=output)


def build_discriminator(input_shape, latent_dim):
    input_images = tf.keras.layers.Input(shape=(input_shape))
    x = input_images
    
    n = 1
    while n <= 2:
        filters = latent_dim // (2 ** n)
        x = Conv2D(filters=filters, kernel_size=4, strides=2, padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
        n += 1
    
    output = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    return tf.keras.Model(inputs=input_images, outputs=output)

