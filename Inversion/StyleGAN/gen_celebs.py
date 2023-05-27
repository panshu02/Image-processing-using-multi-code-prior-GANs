import numpy as np
import tensorflow as tf
from train import train_GAN_single

latent_dim = 64
batch_size = 8

StyleGAN, latent_codes = train_GAN_single(batch_size, latent_dim)

gen_celebs = StyleGAN(latent_codes, training = False)

from matplotlib import pyplot as plt

for celeb in gen_celebs:
    celeb = (celeb + 1) / 2
    plt.imshow(celeb)
    plt.show()
