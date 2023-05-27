from AdaIN_layer import AdaIN
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Conv2DTranspose, Flatten, Dense, LeakyReLU, Conv2D, BatchNormalization

def initialize_weights(shape):
    initializer = tf.initializers.HeNormal()
    weights = tf.Variable(initializer(shape))
    return weights

def build_generator(latent_dim, n_codes, output_channels):
    inputs = tf.keras.layers.Input(shape=(n_codes, latent_dim))
    latent_codes = tf.keras.layers.Reshape((n_codes * latent_dim,))(inputs)
    x = tf.keras.layers.Dense(4 * 4 * n_codes * latent_dim, use_bias=False)(latent_codes)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.Dense(4 * 8 * n_codes * latent_dim, use_bias=True)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.Dense(4 * 8 * n_codes * latent_dim, use_bias=True)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.Dense(4 * 8 * n_codes * latent_dim, use_bias=True)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.Dense(4 * 4 * n_codes * latent_dim, use_bias=False)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.Reshape((4, 4, n_codes * latent_dim))(x)

    # First sub-generator
    sub_outputs = []
    for i in range(n_codes):
        sub_input = x

        n = 1
        while n <= 5:
            filters = (latent_dim * n_codes) // (2 ** n)
            sub_output = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=3, strides=1, padding='same')(sub_input)
            sub_output = BatchNormalization()(sub_output)

            target_shape = sub_output.shape
            latent_sub_codes = tf.keras.layers.Dense(tf.reduce_prod(target_shape[1:]), use_bias=False)(latent_codes)
            latent_sub_codes = tf.keras.layers.Reshape(target_shape[1:])(latent_sub_codes)
            sub_output = AdaIN()([sub_output, latent_sub_codes])

            sub_output = tf.keras.layers.LeakyReLU(alpha=0.2)(sub_output)
            sub_output = tf.keras.layers.UpSampling2D(size=(2, 2))(sub_output)
            sub_input = sub_output
            n += 1

        sub_output = tf.keras.layers.Conv2DTranspose(filters=output_channels, kernel_size=2, strides=1, padding='same')(sub_output)
        sub_output = BatchNormalization()(sub_output)
        sub_outputs.append(sub_output)
    
    # Second sub-generator
    weights = initialize_weights((n_codes, 1))

    # Calculate sum of weights and normalize all weights
    sum_weights = tf.reduce_sum(weights)
    normalized_weights = weights / sum_weights
    
    # Perform element-wise multiplication and summation
    weighted_outputs = [normalized_weights[i] * sub_output[i] for i in range(len(normalized_weights))]
    output_image = tf.reduce_sum(weighted_outputs, axis=0)

    # Reshape output_image to include a batch dimension
    output_image = tf.expand_dims(output_image, axis=0)

    # Final Conv2D and normalization layer
    output_image = Conv2D(filters = output_channels, kernel_size = 3, strides = 1, padding = 'same')(output_image)
    output_image = tf.keras.layers.BatchNormalization()(output_image)
    
    # Tanh activation function
    output = tf.keras.layers.Activation('tanh')(output_image)

    return tf.keras.Model(inputs=inputs, outputs=output)



def build_discriminator(input_shape, latent_dim):
    input_images = tf.keras.layers.Input(shape=input_shape)
    x = input_images
    
    n = 0
    while n <= 6:
        filters = latent_dim // (2 ** n)
        x = Conv2D(filters=filters, kernel_size=4, strides=2, padding='same', use_bias = True)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
        n += 1
    
    x = Flatten()(x)
    outputs = Dense(1, activation='sigmoid')(x)
    
    return tf.keras.Model(inputs=input_images, outputs=outputs)