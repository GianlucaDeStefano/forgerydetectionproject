from pathlib import Path

from tensorflow.python.keras import Input
from tensorflow.python.keras.applications.resnet import ResNet50
from tensorflow.python.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, Dropout, Activation, BatchNormalization
from tensorflow.python.keras.models import Sequential, Model

from Models.BaseModels.CNNModel import CNNModel
from Models.BaseModels.DenseModel import DenseModel


class BaseClassifier(DenseModel, CNNModel):

    def __init__(self, input_shape, n_classes, model_name: str, log_dir: Path, verbose: bool = True):
        super(BaseClassifier, self).__init__(model_name, log_dir, verbose)
        self.input_structure = input_shape
        self.n_classes = n_classes

    @property
    def output_shape(self) -> tuple:
        """
        Return the input shape the model will have to support
        """
        return self.n_classes

    @property
    def input_shape(self) -> tuple:
        """
        For a resnet the outpus shape is simple the number of classes the network has to recognize
        """
        return self.input_structure

    def build_model(self, input_shape, n_classes) -> Model:
        """
        Function in charge of defining the model structure
        :param input_shape: tuple containing the shape of the data this model will recive as input
        :param n_classes: tuple containing the shape of the output produced by this model
        :return: Keras Sequential Model
        """

        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',
                         input_shape=input_shape))
        model.add(MaxPooling2D((2, 2)))

        model.add(Conv2D(16, (3, 3),kernel_initializer='he_uniform', padding='same'))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(MaxPooling2D((2, 2)))

        model.add(Conv2D(32, (3, 3),kernel_initializer='he_uniform', padding='same'))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(MaxPooling2D((2, 2)))

        model.add(Conv2D(64, (3, 3),kernel_initializer='he_uniform', padding='same'))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(MaxPooling2D((2, 2)))

        model.add(Conv2D(128, (3, 3),kernel_initializer='he_uniform', padding='same'))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(MaxPooling2D((2, 2)))

        model.add(Flatten())
        model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(64, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(1, activation='sigmoid'))


        return model