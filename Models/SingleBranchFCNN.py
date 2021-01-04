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
