import tensorflow as tf
from tensorflow import keras

models = {
    'mobilenet': keras.applications.MobileNetV2,
    'resnet': keras.applications.InceptionResNetV2,
    'densenet': keras.applications.DenseNet201,
    'nasnetmobile': keras.applications.NASNetMobile,
}

def prepareModel(image_size, channels, modelName = 'mobilenet'):
    IMG_SHAPE = (image_size, image_size, channels)

    mdl = models.get(modelName.lower(), keras.applications.MobileNetV2)
    if mdl == None:
        raise Exception('Unknown model!')

    print(mdl)
    # Create the base model from the pre-trained model ResNet
    base_model = mdl(
        input_shape=IMG_SHAPE,
        include_top=False,
        weights='imagenet')


    base_model.trainable = False

    # Let's take a look at the base model architecture
    # base_model.summary()




    model = keras.Sequential([
        base_model,
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        loss='binary_crossentropy',
        optimizer=keras.optimizers.RMSprop(lr=0.0001),
        metrics=['accuracy'])
    # model.summary()
    len(model.trainable_variables)

    return model, base_model
