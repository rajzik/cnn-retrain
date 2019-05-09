import sys
import getopt
import numpy as np

from load_model import loadModel
from load_image import loadImage


def predictImage(filepath, tuned=False):
    modelPath = 'models/model.h5'
    if tuned == True:
        modelPath = 'models/tuned_model.h5'

    model = loadModel(modelPath)

    img = loadImage(filepath)

    predictions = model.predict_classes(img)
    prediction = predictions[0][0]
    print(predictions[0])
    if (prediction == 0):
        print('Pussy')
    else:
        print('Dogo')



def parseArgs(argv):

    inputfile = ''
    tuned = False
    try:
        opts, args = getopt.getopt(argv, "htf:", ["file=", "tuned"])
    except getopt.GetoptError:
        print('test.py -f <inputfile> -o')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('test.py -f <inputfile> -t')
            sys.exit()
        elif opt in ("-f", "--file"):
            inputfile = arg
        elif opt in ("-t", "--tuned"):
            tuned = True

    return inputfile, tuned

if __name__ == "__main__":
    filepath, tuned = parseArgs(sys.argv[1:])
    predictImage(filepath, tuned)
