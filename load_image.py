from tensorflow import keras
import numpy as np

def loadImage(path):
    img = keras.preprocessing.image.load_img(path, target_size=(160, 160))
    img = keras.preprocessing.image.img_to_array(img)
    img = img.reshape((1,) + img.shape)

    return img
