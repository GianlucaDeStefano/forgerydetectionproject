from pathlib import Path

from tensorflow.python.keras import Input
from tensorflow.python.keras.applications.resnet import ResNet50
from tensorflow.python.keras.layers import Flatten, Dense
from tensorflow.python.keras.models import Sequential, Model

from Models.BaseModels.CNNModel import CNNModel
from Models.BaseModels.DenseModel import DenseModel
from Models.Customs.ClassifierBase import ClassifierBase


class ClassifierType1(ClassifierBase):

    def build_model(self, input_shape, n_classes) -> Model:
        """
        Function in charge of defining the model structure
        :param input_shape: tuple containing the shape of the data this model will recive as input
        :param n_classes: tuple containing the shape of the output produced by this model
        :return: Keras Sequential Model
        """

        input = Input(shape=input_shape)

        conv_1 = self.convolutional_block(input, filters=16, kernel_size=(3, 3), strides=1)

        max_1 = self.downsampling_block(conv_1,(4,4))

        conv_2 = self.convolutional_block(max_1, filters=16, kernel_size=(3, 3), strides=1)

        max_2 = self.downsampling_block(conv_2, (4, 4))

        conv_3 = self.convolutional_block(max_2, filters=16, kernel_size=(3, 3), strides=1)

        max_3 = self.downsampling_block(conv_3, (2, 2))

        conv_4 = self.convolutional_block(max_3, filters=16, kernel_size=(3, 3), strides=1)

        max_4 = self.downsampling_block(conv_4, (2, 2))

        conv_5 = self.convolutional_block(max_4, filters=128, kernel_size=(3, 3), strides=1)

        flatten = Flatten()(conv_5)

        dense_1 = Dense(256, activation='relu', kernel_initializer='he_uniform') (flatten)
        dense_2 = Dense(16, activation='relu', kernel_initializer='he_uniform') (dense_1)
        dense_3 = Dense(1, activation='sigmoid')(dense_2)


        return Model(input,dense_3)