from __future__ import absolute_import, division, print_function

import os


import tensorflow as tf

import numpy as np

from getdata import getData
from prepare_model import prepareModel
from fit_generator import fit
from show_graphs import showGraphs
from tune_model import tuneModel
from save_model import saveModel

def retrain(shouldTune=False, showGraph=False):

    print("TensorFlow version is ", tf.__version__)
    # Image props
    image_size = 160  # All images will be resized to 160x160
    batch_size = 32
    channels = 3

    # model vars
    epochs = 10

    train_generator, validation_generator = getData(image_size, batch_size)

    model, base_model = prepareModel(image_size, channels)

    len(model.trainable_variables)

    history = fit(model, epochs, train_generator, validation_generator)

    saveModel(model, 'models/model.h5')

    if showGraph == True:
        showGraphs(history)

    if shouldTune == True:
        tunedModel = tuneModel(base_model, model)
        tunedHistory = fit(tunedModel,
                           epochs,
                           train_generator,
                           validation_generator)
        saveModel(tunedModel, 'models/tuned_model.h5')

        if showGraph == True:
            showGraphs(tunedHistory)


if __name__ == "__main__":
    retrain()
