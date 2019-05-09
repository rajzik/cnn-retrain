from __future__ import absolute_import, division, print_function

import os
import sys
import getopt

import tensorflow as tf

import numpy as np

from getdata import getData
from prepare_model import prepareModel
from fit_generator import fit
from show_graphs import showGraphs
from tune_model import tuneModel
from save_model import saveModel
from load_image import loadImage



def parseArgs(argv):

    graph = False
    tuned = False
    try:
        opts, args = getopt.getopt(argv, "htg", ["graph", "tuned"])
    except getopt.GetoptError:
        print('retrain.py -t -g')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('test.py -i <inputfile> -t')
            sys.exit()
        elif opt in ("-g", "--graph"):
            graph = arg
        elif opt in ("-t", "--tuned"):
            tuned = True

    return graph, tuned


def retrain(graph=False, tuned=False):

    print("TensorFlow version is ", tf.__version__)
    # Image props
    image_size = 160  # All images will be resized to 160x160
    batch_size = 32
    channels = 3

    # model vars
    epochs = 10
    print('Fetching Data')
    train_generator, validation_generator = getData(image_size, batch_size)
    print('Preparing model')
    model, base_model = prepareModel(image_size, channels)

    len(model.trainable_variables)
    print('Retraining model')
    history, model = fit(model, epochs, train_generator,
                  validation_generator, batch_size)

    print('Saving model')
    saveModel(model, 'models/model.h5')

    if graph == True:
        showGraphs(history)

    if tuned == True:
        tunedModel = tuneModel(base_model, model)
        tunedHistory, tunedModel = fit(tunedModel,
                           epochs,
                           train_generator,
                           validation_generator,
                           batch_size)
        saveModel(tunedModel, 'models/tuned_model.h5')

        if graph == True:
            showGraphs(tunedHistory)


if __name__ == "__main__":
    graph, tuned = parseArgs(sys.argv[1:])
    retrain(graph, tuned)
