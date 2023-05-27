# Imports and function definitions
from absl import logging
import matplotlib.pyplot as plt
import numpy as np
import os, sys

import tensorflow as tf
tf.random.set_seed(42)

import tensorflow_hub as hub

logging.set_verbosity(logging.ERROR)


# Preprocessing Functions
def convert_image(image):   # Simple way to display an image.
  image = tf.constant(image)
  image = tf.image.convert_image_dtype(image, tf.uint8)
  return image

# Loading and importing GANs
model_dir = ['StyleGAN', 'StyleGAN_multi_code_prior', 'VAE']

sys.path.insert(1, os.path.join(os.path.dirname(__file__), model_dir[0]))
print("Training {}".format(model_dir[0]))
from train import generator as stylegan

sys.path.insert(1, os.path.join(os.path.dirname(__file__), model_dir[1]))
print("\n\nTraining {}".format(model_dir[1]))
from train_multi import generator as stylegan_multi

sys.path.insert(1, os.path.join(os.path.dirname(__file__), model_dir[2]))
print("\n\nTraining {}".format(model_dir[2]))
from train_vae import vae_2

for _ in range(3):
    sys.path.pop(1)

latent_dim = 128
n_codes = 10

progan = hub.load("https://tfhub.dev/google/progan-128/1").signatures['default']
stylegan_multi = stylegan_multi
stylegan_single = stylegan
vae = vae_2

image_ind = 801
target_img_path = "/celebA/img_align_celeba/img_align_celeba/{}.jpg".format((6-len(str(image_ind)))*'0'+str(image_ind))
target_image = plt.imread(target_img_path)

target_image_disp = convert_image(target_image)
plt.imshow(target_image_disp)
plt.show()


# Image inversion and reconstruction
def find_closest_latent_vector(model, initial_vector, num_optimization_steps):
  losses = []

  vector = tf.Variable(initial_vector)  
  optimizer = tf.optimizers.Adam(learning_rate=0.01)
  loss_fn = tf.losses.MeanAbsoluteError(reduction="sum")

  for step in range(num_optimization_steps):
    if (step % 10)==0:
      print()
    print('.', end='')

    with tf.GradientTape() as tape:
      image = model(vector.read_value())['default'][0]

      target_image_difference = loss_fn(image, target_image[:,:,:3])
      regularizer = tf.abs(tf.norm(vector) - np.sqrt(latent_dim))
      
      loss = target_image_difference + regularizer
      losses.append(loss.numpy())
    grads = tape.gradient(loss, [vector])
    optimizer.apply_gradients(zip(grads, [vector]))
    
  return image, losses


# Plotting and visualizing reconstruction process
num_optimization_steps=50
models = [progan, stylegan_single, stylegan_multi, vae]
model_names = ['ProGAN', 'StyleGAN', 'StyleGAN (multi-code)', 'VAE']
results = []


for _, model in enumerate(models):
  tf.random.set_seed(42)

  if _ in [1, 2]:
    initial_vector = tf.random.normal([1, latent_dim])
  elif _ == 3:
    initial_vector = tf.random.normal([1, n_codes, latent_dim])
  else:
     break
  
  image, loss = find_closest_latent_vector(model, initial_vector, num_optimization_steps)
  results.append(image)

processed_target_image = np.expand_dims(target_image, axis = 0).astype('float32')
vae_reconstruct = vae.predict(processed_target_image / 127.5 - 1, batch_size = 1)
vae_reconstruct = ((vae_reconstruct + 1) * 127.5)

results.append(vae_reconstruct)

for i, image in enumerate(results):
    results[i] = convert_image(image)


# View reconstructed images
fig, axs = plt.subplots(1, 3)
for i in range(3):
    axs[0, i].imshow(results[i])
    axs[0, i].xlabel(model_names[i])

plt.title("Reconstructed images")
plt.show()