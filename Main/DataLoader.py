import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class DataLoader:
    def __init__(self, train_dir, test_dir, img_size=(256, 256), batch_size=20):
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.label_names = {0: 'Covid-19', 1: 'Normal', 2: 'Viral Pneumonia', 3: 'Bacterial Pneumonia'}

    def get_data_generators(self):
        image_generator = ImageDataGenerator(rescale=1. / 255, validation_split=0.2)
        train_generator = image_generator.flow_from_directory(
            batch_size=self.batch_size,
            directory=self.train_dir,
            shuffle=True,
            target_size=self.img_size,
            class_mode='categorical',
            subset="training"
        )
        validation_generator = image_generator.flow_from_directory(
            batch_size=self.batch_size,
            directory=self.train_dir,
            shuffle=True,
            target_size=self.img_size,
            class_mode='categorical',
            subset="validation"
        )
        return train_generator, validation_generator

    def get_test_generator(self):
        test_gen = ImageDataGenerator(rescale=1. / 255)
        test_generator = test_gen.flow_from_directory(
            batch_size=self.batch_size,
            directory=self.test_dir,
            shuffle=True,
            target_size=self.img_size,
            class_mode='categorical'
        )
        return test_generator
