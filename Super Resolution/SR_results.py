import numpy as np
from matplotlib import pyplot as plt
import os, sys
from PIL import Image
import tensorflow as tf
import tensorlayerx as tlx
import pandas as pd

folders = ['Bicubic', "PGGAN", "SRCNN", "SRGAN"]

sys.path.insert(1, os.path.join(os.path.dirname(__file__), folders[0]))
import bicubic as DIP

sys.path.insert(1, os.path.join(os.path.dirname(__file__), folders[1]))
print("Training {}".format(folders[1]))
import train as PGGAN

sys.path.insert(1, os.path.join(os.path.dirname(__file__), folders[2]))
print("Training {}".format(folders[2]))
import train as SRCNN

sys.path.insert(1, os.path.join(os.path.dirname(__file__), folders[3]))
print("Training {}".format(folders[3]))
import train as SRGAN

for _ in range(4):
    sys.path.pop(1)


bicubic = DIP.bicubic
pggan = PGGAN.generator
srcnn = SRCNN.srcnn
srgan = SRGAN.G

lr_image_path = './DIV2K/DIV2K_valid_LR_bicubic/X4/0801x4.png'
lr_image = Image.open(lr_image_path)
lr_image = lr_image.resize((96, 96))

def normalize(img):
    normalized_img = np.array(img).astype('float32') / 255.0
    return normalized_img

normalized_img = normalize(lr_image)

n_codes = 10
latent_dim = 128

tf.random.set_seed(42)
latent_codes = tf.random.normal([1, n_codes, latent_dim])

def SRGAN_predict(valid_lr_img, srgan):
    valid_lr_img_tensor = (valid_lr_img / 127.5) - 1  # rescale to ［－1, 1]

    valid_lr_img_tensor = np.asarray(valid_lr_img_tensor, dtype=np.float32)
    valid_lr_img_tensor = np.transpose(valid_lr_img_tensor,axes=[2, 0, 1])
    valid_lr_img_tensor = valid_lr_img_tensor[np.newaxis, :, :, :]
    valid_lr_img_tensor= tlx.ops.convert_to_tensor(valid_lr_img_tensor)
    valid_lr_img_tensor = np.expand_dims(valid_lr_img_tensor, axis = 0)

    out = tlx.ops.convert_to_numpy(srgan(valid_lr_img_tensor))
    out = np.squeeze(out, axis = 0)
    out = np.asarray((out + 1) * 127.5, dtype=np.uint8)
    out = np.transpose(out[0], axes=[1, 2, 0])

    return out


bicubic_out = bicubic(lr_image)

srcnn_out = srcnn.predict(normalized_img, batch_size = 1)
srcnn_out = (srcnn_out * 255.0).astype('uint8')

srgan_out = SRGAN_predict(lr_image, srgan)

pggan_out = pggan.predict([normalized_img, latent_codes])
pggan_out = (pggan_out * 255.0).astype('uint8')

results = []
results.append(bicubic)
results.append(srcnn_out)
results.append(srgan_out)
results.append(pggan_out)

results_dict = {1 : "DIP (Bicubic)", 2 : "SRCNN", 3: "SRGAN", 4:"PGGAN (multi-code prior)"}

fig, axs = plt.subplots(2, 2)
for i in range(4):
    axs[i // 2, i % 2].imshow(results[i])
    axs[i // 2, i % 2].xlabel("HR image using {}".format(results_dict[i]))

plt.show()


from metrics import compare_images

# Function to display metrics of generated images using a dataframe
def display_results(image_results, original_image, index):
    # Create an empty dataframe
    df = pd.DataFrame(index=index, columns=['PSNR', 'MSE', 'SSIM'])

    # Iterate over the reconstructed images and compute scores
    for generated_image in image_results:
        # Compute scores using the compare_scores function
        scores = compare_images(original_image, generated_image)
        
        # Assign the scores to the corresponding row in the dataframe
        df.loc[index[i]] = scores

    # Display the dataframe
    print(df)

display_results(results, lr_image, list(results_dict.values()))