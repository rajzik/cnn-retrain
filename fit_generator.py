def fit(model, epochs, train_generator, validation_generator):
    steps_per_epoch = train_generator.n // batch_size
    validation_steps = validation_generator.n // batch_size

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        workers=4,
        validation_data=validation_generator,
        validation_steps=validation_steps)

    return history
