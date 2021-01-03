from abc import ABC

from tensorflow.python.keras.models import Sequential

from Models.BaseModel import BaseModel
import tensorflow as tf

class CNNModel(BaseModel, ABC):

    def convolutional_layer(self,model:Sequential,filters,kernel_size,strides,padding="same",dropout_rate=0.4,activation="relu") -> Sequential:
        """
        This function defines the standard layer of a CNN conposed by:
            - Convolution
            - Dropout
            - BatchNormalization
            - Activation fadd unction
        We can use it as a building block to build larger networks in an organized manner

        :param model: the model to which we should append the block
        :param filters: the number of filters the CNN should output
        :param kernel_size: the kernel size of the CNN
        :param strides: the strides parameter of the CNN
        :param padding: the padding parameter to pass to the CNN layer (default: same)
        :param dropout_rate: probability of a neuron to be switched off
        :param activation: the type of activation function to use (eg: Relu)
        :return:
        """

        model = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides,padding=padding)(model)
        model = tf.keras.layers.Dropout(dropout_rate)(model)
        model = tf.keras.layers.BatchNormalization()(model)
        model = tf.keras.layers.Activation(activation)(model)
        return model
