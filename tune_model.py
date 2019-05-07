def tuneModel(base_model, model):
    base_model.trainable = True
    print("Number of layers in the base model: ", len(base_model.layers))

    # Fine tune from this layer onwards
    fine_tune_at = 100

    # Freeze all the layers before the `fine_tune_at` layer
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.RMSprop(lr=2e-5),
        metrics=['accuracy'])

    model.summary()

    return model
