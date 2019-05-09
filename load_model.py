from __future__ import absolute_import, division, print_function

from tensorflow import keras
import h5py

def loadModel(path):
    model = keras.models.load_model(path)

    return model
