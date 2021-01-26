from pathlib import Path

from tensorflow.python.keras import Input
from tensorflow.python.keras.applications.resnet import ResNet50
from tensorflow.python.keras.layers import Flatten, Dense
from tensorflow.python.keras.models import Sequential, Model

from Models.BaseModels.CNNModel import CNNModel
from Models.BaseModels.DenseModel import DenseModel


class ClassifierBase(DenseModel,CNNModel):

    def __init__(self, input_shape, n_classes, model_name: str, log_dir: Path, verbose: bool = True):
        super(ClassifierBase, self).__init__(model_name, log_dir, verbose)
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

        input = Input(shape=input_shape)

        conv_1 = self.convolutional_block(input, filters=16, kernel_size=(3, 3), strides=1)
        conv_2 = self.convolutional_block(conv_1, filters=16, kernel_size=(3, 3), strides=1)

        max_1 = self.downsampling_block(conv_2,(2,2))

        conv_3 = self.convolutional_block(max_1, filters=32, kernel_size=(3, 3), strides=1)
        conv_4 = self.convolutional_block(conv_3, filters=32, kernel_size=(3, 3), strides=1)

        max_2 = self.downsampling_block(conv_4, (2, 2))

        conv_5 = self.convolutional_block(max_2, filters=64, kernel_size=(3, 3), strides=1)
        conv_6 = self.convolutional_block(conv_5, filters=64, kernel_size=(3, 3), strides=1)

        max_3 = self.downsampling_block(conv_6, (2, 2))

        conv_7 = self.convolutional_block(max_3, filters=64, kernel_size=(3, 3), strides=1)
        conv_8 = self.convolutional_block(conv_7, filters=64, kernel_size=(3, 3), strides=1)

        max_4 = self.downsampling_block(conv_8, (2, 2))

        conv_9 = self.convolutional_block(max_4, filters=64, kernel_size=(3, 3), strides=1)
        conv_10 = self.convolutional_block(conv_9, filters=64, kernel_size=(3, 3), strides=1)

        max_5 = self.downsampling_block(conv_10, (2, 2))

        conv_11 = self.convolutional_block(max_5, filters=128, kernel_size=(3, 3), strides=1)

        flatten = Flatten()(conv_11)

        dense_1 = Dense(128, activation='relu', kernel_initializer='he_uniform') (flatten)
        dense_3 = Dense(1, activation='sigmoid')(dense_1)


        return Model(input,dense_3)