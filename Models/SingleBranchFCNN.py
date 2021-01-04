from tensorflow.python.keras.models import Sequential, Model
import tensorflow as tf

from Models.CNNModel import CNNModel


class SingleBranchFCNN(CNNModel):
    """
        Class containing the implementation of a standard (single branch)
        convolutional neural network.
    """

    def build_model(self, input_shape, output_shape) -> Sequential:
        input = tf.keras.layers.Input(shape=input_shape)
        model = self.convolutional_block(input, 16, 3, 1)
        model = self.convolutional_block(model, 16, 3, 1)

        return Model([input], [model])

    @property
    def input_shape(self) -> tuple:
        """
        This property returns the input shape of the model
        :return: tuple, in this case the tuple is (None,None,3) because the input images can have variable width and height
        but they are always RGB images with 3 channels
        """
        return (None,None,3)

    @property
    def output_shape(self) -> tuple:
        """
        This property returns the output shape of the model
        :return: tuple, in this case the tuple is (None,None,1) because the output image can have variable width and height
        and only one class of pixel to recognize (Tampered)
        """
        return (None,None,1)


