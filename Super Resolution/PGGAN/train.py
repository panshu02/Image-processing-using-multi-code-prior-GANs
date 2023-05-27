import tensorflow as tf
from tensorflow.keras import layers
import model
import losses
import load_and_preprocess

# Define the batch size
batch_size = 8

# Define the latent code dimension
latent_dim = 128

# Define the number of latent_codes
n_codes = 10

# Load dataset and make batches
dataset = load_and_preprocess.dataset
dataset = load_and_preprocess.make_batches(dataset, batch_size)

# Define the PGGAN models
generator = model.build_generator(n_codes, latent_dim)
discriminator = model.build_discriminator()

# Define the optimizer for the generator and discriminator
generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)

# Define the number of epochs for training
num_epochs = 0

# Function to generate random latent codes
def generate_latent_codes(batch_size, latent_dim, n_codes):
    tf.random.set_seed(42)
    return tf.random.normal([batch_size, n_codes, latent_dim])

# Function to compute gradients and update the generator and discriminator parameters
@tf.function
def train_step(lr_image_batch, hr_image_batch, latent_dim, n_codes):
    # Create latent codes of the required shape
    batch_size = lr_image_batch.shape[0]
    latent_codes = generate_latent_codes(batch_size, latent_dim, n_codes)

    # Compute gradients and apply to update weights
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_hr = generator([lr_image_batch, latent_codes], training=True)
        
        real_output = discriminator(hr_image_batch, training=True)
        fake_output = discriminator(generated_hr, training=True)

        disc_loss = losses.discriminator_loss(real_output, fake_output)
        gen_loss = losses.generator_loss(fake_output)
    
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    
    return disc_loss, gen_loss


# Iterate over the dataset for the specified number of epochs
for epoch in range(num_epochs):
    for lr_images, hr_images in dataset:
        disc_loss, gen_loss = train_step(lr_images, hr_images)
    
    # Print training progress
    print(f"Epoch {epoch+1}/{num_epochs} - Discriminator Loss: {disc_loss:.4f} - Generator Loss: {gen_loss:.4f}")