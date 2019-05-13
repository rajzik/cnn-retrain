import sys
import getopt
import numpy as np

from load_model import loadModel
from load_image import loadImage


def predictImage(filepath, tuned=False, modelName='mobilenet'):
    modelPath = f'models/{modelName}-model.h5'
    if tuned == True:
        modelPath = f'models/tuned_{modelName}-model.h5'

    model = loadModel(modelPath)

    img = loadImage(filepath)

    predictions_classes = model.predict_classes(img)

    prediction = predictions_classes[0][0]
    print(predictions_classes[0])
    if (prediction == 0):
        print('Cat')
    else:
        print('Dogo')



def parseArgs(argv):

    inputfile = ''
    tuned = False
    model = 'mobilenet'
    try:
        opts, args = getopt.getopt(argv, "htf:m:", ["file=", "tuned", "model="])
    except getopt.GetoptError:
        print('predict.py -f <filename> -t -model mobilenet')
        print('models:')
        print('\tmobilenet')
        print('\tresnet')
        print('\tdensenet')
        print('\tnasnetlarge')
        print('\tnasnetmobile')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('predict.py -f <filename> -t -model mobilenet')
            print('models:')
            print('\tmobilenet')
            print('\tresnet')
            print('\tdensenet')
            print('\tnasnetlarge')
            print('\tnasnetmobile')
            sys.exit()
        elif opt in ("-f", "--file"):
            inputfile = arg
        elif opt in ("-t", "--tuned"):
            tuned = True
        elif opt in ('-m', '--model'):
            model = arg

    return inputfile, tuned, model

if __name__ == "__main__":
    filepath, tuned, model = parseArgs(sys.argv[1:])
    predictImage(filepath, tuned, model)
