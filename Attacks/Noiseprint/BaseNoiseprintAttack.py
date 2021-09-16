from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf

from Attacks.BaseWhiteBoxAttack import BaseWhiteBoxAttack
from Detectors.Noiseprint.noiseprintEngine import NoiseprintEngine
from Detectors.Noiseprint.utility.utility import jpeg_quality_of_file, prepare_image_noiseprint
from Ulitities.Image.Picture import Picture


def normalize_gradient(gradient, margin=17):
    """
    Normalize the gradient cutting away the values on the borders
    :param margin: margin to use along the bordes
    :param gradient: gradient to normalize
    :return: normalized gradient
    """

    # set to 0 part of the gradient too near to the border
    if margin > 0:
        gradient[0:margin, :] = 0
        gradient[-margin:, :] = 0
        gradient[:, 0:margin] = 0
        gradient[:, -margin:] = 0

    # scale the final gradient using the computed infinity norm
    gradient = gradient / np.max(np.abs(gradient))
    return gradient


class BaseNoiseprintAttack(BaseWhiteBoxAttack, ABC):
    """
        This class in used as a base to implement white box attacks on the noiseprint"
    """

    name = "Base Noiseprint Attack"

    def __init__(self, target_image: Picture, target_image_mask: Picture, source_image: Picture,
                 source_image_mask: Picture, steps: int, alpha: float, quality_factor=None,
                 regularization_weight=0.05,plot_interval=5,debug_root: str = "./Data/Debug/", verbose: bool = True):
        """
        :param target_image: original image on which we should perform the attack
        :param target_image_mask: original mask of the image on which we should perform the attack
        :param source_image: image from which we will compute the target representation
        :param source_image_mask: mask of the imae from which we will compute the target representation
        :param steps: number of attack iterations to perform
        :param alpha: strength of the attack
        :param quality_factor: [101,51] specify if we need to load a noiseprint model for the specific given jpeg quality
               level, if it left to None, the right model will be inferred from the file
        :param regularization_weight: [0,1] importance of the regularization factor in the loss function
        :param plot_interval: how often (# steps) should the step-visualizations be generated?
        :param debug_root: root folder insede which to create a folder to store the data produced by the pipeline
        :param verbose: verbosity of the logs printed in the console
        """

        super().__init__(target_image, target_image_mask, source_image, source_image_mask, "Noiseprint", steps, alpha,regularization_weight,
                         plot_interval,False, debug_root, verbose)

        # compute and save the quality factor to use if it has not been specifier or if it is invalid
        self.inferred = False
        if not quality_factor or quality_factor < 51 or quality_factor > 101:
            try:
                quality_factor = jpeg_quality_of_file(target_image.path)
                self.inferred = True
            except:
                quality_factor = 101

        self.quality_factor = quality_factor

        # instantiate the noiseprint engine class
        self._engine = NoiseprintEngine()

        # load the desired model
        self._engine.load_quality(self.quality_factor)

        # create variable to store the generated adversarial noise
        self.noise = np.zeros((target_image.shape[0], target_image.shape[1]))

        # create variable to store the momentum of the gradient
        self.moving_avg_gradient = np.zeros((target_image.shape[0], target_image.shape[1]))

    def _on_before_attack(self):
        """
        Instructions executed before performing the attack, writing logs
        :return:
        """

        super()._on_before_attack()

        self.write_to_logs("Quality factor to be used: {} {}".format(self.quality_factor,"(inferred)" if self.inferred else ""))

    def loss(self, y_pred, y_true):
        """
        Specify a loss function to drive the image we are attacking towards the target representation
        The default loss is the squared-error
        :param y_pred: last output of the model
        :param y_true: target representation
        :return: loss value
        """
        return tf.norm(y_pred-y_true,2)

    def attack(self, image_to_attack: Picture, *args, **kwargs):
        """
        Perform step of the attack executing the following steps:
            (1) -> prepare the image to be used by noiseprint
            (2) -> compute the gradient
            (3) -> normalize the gradient
            (4) -> apply the gradient to the image with the desired strength
            (5) -> return the image
        :return: attacked image
        """
        # compute the attacked image using the original image and the compulative noise to reduce
        # rounding artifacts caused by translating the noise from one to 3 channels and vice versa multiple times
        image_one_channel = prepare_image_noiseprint(image_to_attack) - self.noise

        # compute the gradient
        image_gradient, loss = self._get_gradient_of_image(image_one_channel, self.target_representation)

        # save loss value to plot it
        self.loss_steps.append(loss)

        # normalize the gradient
        image_gradient = normalize_gradient(image_gradient, 0)

        # scale the gradient
        image_gradient = self.alpha * image_gradient

        # update the moving average
        self.moving_avg_gradient = self.moving_avg_gradient * 0.9 + 0.1 * image_gradient

        # add this iteration contribution to the cumulative noise
        self.noise += self.moving_avg_gradient / (1 - 0.75 ** (1 + self.step_counter))

        return self.attacked_image

    @abstractmethod
    def _get_gradient_of_image(self, image: Picture, target: Picture):
        """
        Compute the gradient for the entire image
        :param image: image for which we have to compute the gradient
        :param target: target to use
        :return: numpy array containing the gradient
        """

        raise NotImplemented

    def _get_gradient_of_patch(self, image: Picture, target: Picture, regularization_value=0):
        """
        Given an  input image and a target, compute the gradient of the target w.r.t. the inputs on the loaded noiseprint model
        :param image_patch: input image
        :param target: target representation we want our model to produce
        :param regularization_value: value to be added to the loss function as the result of a regularizaion operation
        previously computed
        :return: gradient,loss
        """
        
        # check that input image and target rerpresentation have the same shape
        assert (image.shape == target.shape)

        # prepare the tape object to compute the gradient
        with tf.GradientTape() as tape:
            # convert the input image into a tensor
            tensor_patch = tf.convert_to_tensor(image[np.newaxis, :, :, np.newaxis])
            tape.watch(tensor_patch)

            # perform feed forward pass
            noiseprint = tf.squeeze(self._engine.model(tensor_patch))

            # compute the loss with respect to the target representation
            loss = self.loss(noiseprint, target) + regularization_value

            # retrieve the gradient of the image
            gradient = np.squeeze(tape.gradient(loss, tensor_patch).numpy())

            # check that the retrieved gradient has the correct shape
            assert (gradient.shape == image.shape)

            return gradient, loss

    @property
    def attacked_image(self):
        """
        Compute the attacked image using the original image and the cumulative noise to reduce
        rounding artifacts caused by translating the noise from one to 3 channels and vie versa multiple times,
        still this operation here is done once so some rounding error is still present.
        Use attacked_image_monochannel to get the one channel version of the image withoud rounding errors
        :return:
        """
        return Picture((self.target_image - Picture(self.noise).three_channels(1/3,1/3,1/3)).clip(0,255))

    @property
    def attacked_image_monochannel(self):
        """
        Compute the attacked image using the original image and the compulative noise to reduce
        rounding artifacts caused by translating the nosie from one to 3 channels and vie versa multiple times
        :return:
        """
        return Picture((self.target_image.one_channel() - self.noise).clip(0,255))
