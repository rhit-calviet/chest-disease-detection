from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import optimizers


class Trainer:
    def __init__(self, model, train_generator, val_generator):
        self.model = model
        self.train_generator = train_generator
        self.val_generator = val_generator

    def compile_model(self):
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=optimizers.RMSprop(learning_rate=1e-4, decay=1e-6), metrics=["accuracy"])

    def train_model(self, epochs=100, patience=20):
        earlystopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=patience)
        checkpointer = ModelCheckpoint(filepath="../weights.hdf5", verbose=1, save_best_only=True)
        history = self.model.fit(
            self.train_generator,
            steps_per_epoch=self.train_generator.n // self.train_generator.batch_size,
            epochs=epochs,
            validation_data=self.val_generator,
            validation_steps=self.val_generator.n // self.val_generator.batch_size,
            callbacks=[checkpointer, earlystopping]
        )
        return history
