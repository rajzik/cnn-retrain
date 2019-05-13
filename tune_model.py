from tensorflow import keras


def tuneModel(base_model, model):
    base_model.trainable = True

    print("Number of layers in the base model: ", len(base_model.layers))


    # Fine tune from this layer onwards 60%
    fine_tune_at = round(len(base_model.layers) * .65)
    print(f'Tuning layers count: {len(base_model.layers) - fine_tune_at}')

    # Freeze all the layers before the `fine_tune_at` layer
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    model.compile(
        loss='binary_crossentropy',
        optimizer=keras.optimizers.RMSprop(lr=2e-5),
        metrics=['accuracy'])

    model.summary()

    len(model.trainable_variables)

    return model
