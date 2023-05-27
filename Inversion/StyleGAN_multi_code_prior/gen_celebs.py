import numpy as np
import tensorflow as tf
from train import train_GAN_multi

latent_dim = 64
batch_size = 8
n_codes = 10

StyleGAN_multi_code, latent_codes = train_GAN_multi(batch_size, latent_dim, n_codes)

gen_celebs = StyleGAN_multi_code(latent_codes, training = False)

from matplotlib import pyplot as plt

for celeb in gen_celebs:
    celeb = (celeb + 1) / 2
    plt.imshow(celeb)
    plt.show()
