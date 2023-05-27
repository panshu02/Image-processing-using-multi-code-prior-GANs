import tensorflow as tf
from tensorflow.keras import layers

def ResidualBlock(x, scale, kernel_size, strides, alpha, use_bias = False):
    z = layers.Conv2D(64*scale, kernel_size, strides, padding = 'same', use_bias = use_bias)
    z = layers.BatchNormalization()(z)
    z = layers.LeakyReLU(alpah = alpha)(z)

    return z

# Generator model
def build_generator(n_codes, latent_dim):
    input_lr = layers.Input(shape=(96, 96, 3))
    input_latent = layers.Input(shape=(n_codes, latent_dim))
    
    # Embedding layer to transform latent codes
    latent_embedding = layers.Dense(256)(input_latent)
    latent_embedding = layers.LeakyReLU()(latent_embedding)
    latent_embedding = layers.BatchNormalization()(latent_embedding)

    # Add residual blocks to the network
    for _ in range(4):
        scale = (_ + 1) if _ < 2 else abs(_ - 4)
        use_bias = True if _ in [1, 2] else False
        latent_embedding = ResidualBlock(latent_embedding, scale, 3, 1, 0.2, use_bias)
        input_lr = ResidualBlock(input_lr, scale, 3, 1, 0.2, use_bias)
    
    # Concatenate embedded latent codes with the input LR image
    x = layers.Concatenate()([input_lr, latent_embedding])

    for _ in range(3):
        scale = 3 if _ % 2 else 2
        use_bias = True if _ % 2 else False
        x = ResidualBlock(x, scale, 3, 1, 0.2, use_bias)
    
    output_hr = layers.Conv2D(3, kernel_size=3, strides=1, padding='same', activation='sigmoid')(x)
    
    generator = tf.keras.Model(inputs=[input_lr, input_latent], outputs=output_hr)
    
    return generator

# Discriminator model
def build_discriminator():
    input_hr = layers.Input(shape=(96*4, 96*4, 3))
    
    # Discriminator network architecture
    for _ in range(4):
        scale = 4 - _
        use_bias = True if _ in [1, 2] else False
        x = ResidualBlock(input_hr, scale, 3, 2, 0.2, use_bias)
    
    x = layers.Dense(96 * 96 * 64, activation = 'relu')(x)
    x = layers.Dense(48 * 32 * 24, activation = 'relu')(x)
    x = layers.Dense(1000, activation = 'relu')(x)

    # Output layer
    output = layers.Dense(1, activation='sigmoid')(x)
    
    discriminator = tf.keras.Model(inputs=input_hr, outputs=output)

    return discriminator
