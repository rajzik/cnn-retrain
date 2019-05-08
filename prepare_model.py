import tensorflow as tf
from tensorflow import keras

def prepareModel(image_size, channels):
    IMG_SHAPE = (image_size, image_size, channels)

    # Create the base model from the pre-trained model MobileNet V2
    base_model = tf.keras.applications.ResNet50(
        input_shape=IMG_SHAPE,
        include_top=False,
        weights='imagenet')


    base_model.trainable = False

    # Let's take a look at the base model architecture
    base_model.summary()


    model = tf.keras.Sequential([
        base_model,
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(lr=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy'])
    model.summary()

    return model, base_model
