from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Input, AveragePooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Model


class ModelBuilder:
    def __init__(self, img_size=(256, 256, 3), num_classes=4):
        self.img_size = img_size
        self.num_classes = num_classes

    def build_model(self):
        basemodel = ResNet50(weights='imagenet', include_top=False, input_tensor=Input(shape=self.img_size))
        for layer in basemodel.layers[:-10]:
            layer.trainable = False

        head_model = basemodel.output
        head_model = AveragePooling2D(pool_size=(4, 4))(head_model)
        head_model = Flatten(name='flatten')(head_model)
        head_model = Dense(256, activation='relu')(head_model)
        head_model = Dropout(0.3)(head_model)
        head_model = Dense(128, activation='relu')(head_model)
        head_model = Dropout(0.2)(head_model)
        head_model = Dense(self.num_classes, activation='softmax')(head_model)

        model = Model(inputs=basemodel.input, outputs=head_model)
        return model
