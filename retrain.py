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
    model = 'mobilenet'
    try:
        opts, args = getopt.getopt(argv, "htgm:", ["graph", "tuned", "model="])
    except getopt.GetoptError:
        print('retrain.py -t -g -m mobilenet \n')
        print('Models: \n')
        print('\tmobilenet\n')
        print('\tresnet\n')
        print('\tdensenet\n')
        print('\tnasnetmobile\n')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('retrain.py -t -g -m mobilenet')
            print('Models:')
            print('\tmobilenet')
            print('\tresnet')
            print('\tdensenet')
            print('\tnasnetmobile')
            sys.exit()
        elif opt in ("-g", "--graph"):
            graph = True
        elif opt in ("-t", "--tuned"):
            tuned = True
        elif opt in ('-m', '--model'):
            model = arg

    return graph, tuned, model


def retrain(graph=False, tuned=False, modelName='mobilenet'):

    print("TensorFlow version is ", tf.__version__)
    # Image props
    image_size = 224  # All images will be resized to 224x224
    batch_size = 32
    channels = 3

    # model vars
    epochs = 10
    print('Fetching Data')
    train_generator, validation_generator = getData(image_size, batch_size)
    print('Preparing model')
    model, base_model = prepareModel(image_size, channels, modelName)

    len(model.trainable_variables)
    print('Retraining model')
    history, model = fit(model, epochs, train_generator,
                  validation_generator, batch_size)

    print('Saving model')
    saveModel(model, f'models/{modelName}-model.h5')

    if graph == True:
        showGraphs(history)

    if tuned == True:
        tunedModel = tuneModel(base_model, model)
        tunedHistory, tunedModel = fit(tunedModel,
                           epochs,
                           train_generator,
                           validation_generator,
                           batch_size)
        saveModel(tunedModel, f'models/tuned_{modelName}-model.h5')

        if graph == True:
            showGraphs(tunedHistory)


if __name__ == "__main__":
    graph, tuned, model = parseArgs(sys.argv[1:])
    retrain(graph, tuned, model)
