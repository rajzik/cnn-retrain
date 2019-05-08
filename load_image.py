
import numpy as np
import matplotlib.image as mpimg

def loadImage(path):
    img = mpimg.imread(path)
    img = np.reshape(img, (160, 160))

    return img
