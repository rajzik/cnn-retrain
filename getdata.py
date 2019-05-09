import os
import tensorflow as tf
from tensorflow import keras

def getData(image_size, batch_size):
    zip_file = tf.keras.utils.get_file(origin="https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip",
                                       fname="cats_and_dogs_filtered.zip", extract=True)
    base_dir, _ = os.path.splitext(zip_file)

    train_dir = os.path.join(base_dir, 'train')
    validation_dir = os.path.join(base_dir, 'validation')

    # Directory with our training cat pictures
    train_cats_dir = os.path.join(train_dir, 'cats')
    print('Total training cat images:', len(os.listdir(train_cats_dir)))

    # Directory with our training dog pictures
    train_dogs_dir = os.path.join(train_dir, 'dogs')
    print('Total training dog images:', len(os.listdir(train_dogs_dir)))

    # Directory with our validation cat pictures
    validation_cats_dir = os.path.join(validation_dir, 'cats')
    print('Total validation cat images:', len(os.listdir(validation_cats_dir)))

    # Directory with our validation dog pictures
    validation_dogs_dir = os.path.join(validation_dir, 'dogs')
    print('Total validation dog images:', len(os.listdir(validation_dogs_dir)))

    # Rescale all images by 1./255 and apply image augmentation
    train_datagen = keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255)

    validation_datagen = keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255)

    # Flow training images in batches of 20 using train_datagen generator
    train_generator = train_datagen.flow_from_directory(
        train_dir,  # Source directory for the training images
        target_size=(image_size, image_size),
        batch_size=batch_size,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary'
        )

    # Flow validation images in batches of 20 using test_datagen generator
    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,  # Source directory for the validation images
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='binary'
        )

    return train_generator, validation_generator
