import random
from matplotlib import pyplot as plt
import numpy as np

import train
from load_data import get_input
import preprocess


vae_2 = train.vae_2
img_id = np.arange(1, 1001)
img_path = preprocess.img_path
b_size = train.b_size


def evaluate():
    x_test = []
    for i in range(64):
        x_test.append(get_input(img_path[random.randint(0,len(img_id))]))
    x_test = np.array(x_test)
    figure_Decoded = vae_2.predict(x_test.astype('float32')/127.5 -1, batch_size = b_size)

    for i in range(4):
        plt.axis('off')
        plt.subplot(2,4,1+i*2)
        plt.imshow(x_test[i])
        plt.axis('off')
        plt.subplot(2,4,2 + i*2)
        plt.imshow((figure_Decoded[i]+1)/2)
        plt.axis('off')
        plt.show()
    
evaluate()