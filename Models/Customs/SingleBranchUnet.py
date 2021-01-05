from tensorflow.python.keras.layers import concatenate
from tensorflow.python.keras.models import Sequential, Model
import tensorflow as tf
from Models.CNN.CNNModel import CNNModel


class SingleBranchFCNN(CNNModel):
    """
        Class containing the implementation of a Unet networks for segmentation.
        More details here: https://en.wikipedia.org/wiki/U-Net
    """

    def build_model(self, input_shape, output_shape) -> Sequential:
        """
        Function in charge of defining the model structure
        :param input_shape: tuple containing the shape of the data this model will recive as input
        :param output_shape: tuple containing the shape of the output produced by this model
        :return: Keras Sequential Model
        """
        input = tf.keras.layers.Input(shape=input_shape)

        down_1 = self.unet_downscale_block(input, filters=32, kernel_size=(3, 3), strides=1,
                                           downsamppling_factor=(2, 2))

        down_2 = self.unet_downscale_block(down_1, filters=64, kernel_size=(3, 3), strides=1,
                                           downsamppling_factor=(2, 2))

        down_3 = self.unet_downscale_block(down_2, filters=128, kernel_size=(3, 3), strides=1,
                                           downsamppling_factor=(2, 2))

        conv = self.convolutional_block(down_3, filters=128, kernel_size=(3, 3), strides=1)

        conv = self.convolutional_block(conv, filters=128, kernel_size=(3, 3), strides=1)

        up_3 = self.unet_upscale_block(conv, filters=128, kernel_size=(3, 3), strides=1, upsampling_factor=(2, 2),
                                       input2=down_3)

        up_2 = self.unet_upscale_block(up_3, filters=64, kernel_size=(3, 3), strides=1, upsampling_factor=(2, 2),
                                       input2=down_2)

        up_1 = self.unet_upscale_block(up_2, filters=32, kernel_size=(3, 3), strides=1, upsampling_factor=(2, 2),
                                       input2=down_1)

        conv = self.convolutional_block(up_1, filters=16, kernel_size=(3, 3), strides=1)

        conv = self.convolutional_block(conv, filters=8, kernel_size=(3, 3), strides=1)

        # add a last block with activation = sigmoidal to squash the output of each pixel in the range [0,1]
        # the number of filters should be the same as the number of depth dimensions in the output image
        segmentation = self.convolutional_block(conv, filters=input_shape[2], strides=(1, 1), activation="sigmoid")

        return Model([input], [segmentation])

    @staticmeth
    def unet_downscale_block(model: Sequential, filters, kernel_size, strides, downsamppling_factor, padding="same",
                             dropout_rate=0.4,
                             activation="relu"):
        """
        This is the main building block to build the Uned encoder structure. It is composed by :

            - 2 convolutional blocks
            - 1 downsampling block

        :param model: the model to which we should append the block
        :param filters: the number of filters the CNN should output
        :param kernel_size: the kernel size of the CNN
        :param strides: the strides parameter of the CNN
        :param downsamppling_factor: the factor by which we want to downsample
        :param padding: the padding parameter to pass to the CNN layer (default: same)
        :param dropout_rate: probability of a neuron to be switched off
        :param activation: the type of activation function to use (eg: Relu)
        :return: the initial sequential model with the block appended
        """
        model = CNNModel.convolutional_block(model, filters, kernel_size, strides, padding, dropout_rate, activation)
        model = CNNModel.convolutional_block(model, filters, kernel_size, strides, padding, dropout_rate, activation)
        model = CNNModel.downsampling_block(model, downsamppling_factor)

        return model

    @staticmethod
    def unet_upscale_block(input1: Sequential, filters, kernel_size, strides, upsampling_factor,
                           input2: Sequential = None, padding="same", dropout_rate=0.4,
                           activation="relu"):
        """
        This is the main building block to build the Uned decoder structure. It is composed by :

            - 2 convolutional blocks
            - 1 upsampling block

        :param input1: the model to which we should append the block
        :param filters: the number of filters the CNN should output
        :param kernel_size: the kernel size of the CNN
        :param strides: the strides parameter of the CNN
        :param upsampling: the factor by which we want to up-sample
        :param input2: as second inut to append to the first after the upscale, essentially this is for creating a skip connection
        :param padding: the padding parameter to pass to the CNN layer (default: same)
        :param dropout_rate: probability of a neuron to be switched off
        :param activation: the type of activation function to use (eg: Relu)
        :return: the initial sequential model with the block appended
        """
        model = CNNModel.upsampling_block(input1, upsampling_factor)

        # if a second input is defined create a skip connection
        if input2:
            model = concatenate([model, input2], axis=-1)

        model = CNNModel.convolutional_block(model, filters, kernel_size, strides, padding, dropout_rate, activation)
        model = CNNModel.convolutional_block(model, filters, kernel_size, strides, padding, dropout_rate, activation)
        return model

    @property
    def input_shape(self) -> tuple:
        """
        This property returns the input shape of the model
        :return: tuple, in this case the tuple is (None,None,3) because the input images can have variable width and height
        but they are always RGB images with 3 channels
        """
        return (None, None, 3)

    @property
    def output_shape(self) -> tuple:
        """
        This property returns the output shape of the model
        :return: tuple, in this case the tuple is (None,None,1) because the output image can have variable width and height
        and only one class of pixel to recognize (Tampered)
        """
        return (None, None, 1)
