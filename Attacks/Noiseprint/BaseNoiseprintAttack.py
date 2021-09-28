from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf

from Attacks.BaseWhiteBoxAttack import BaseWhiteBoxAttack
from Detectors.Noiseprint.noiseprintEngine import NoiseprintEngine
from Detectors.Noiseprint.utility.utility import jpeg_quality_of_file, prepare_image_noiseprint
from Ulitities.Image.Picture import Picture


class BaseNoiseprintAttack(BaseWhiteBoxAttack, ABC):
    """
        This class in used as a base to implement white box attacks on the noiseprint"
    """

    name = "Base Noiseprint Attack"

    def __init__(self, target_image: Picture, target_image_mask: Picture, source_image: Picture,
                 source_image_mask: Picture, steps: int, alpha: float, quality_factor=None,
                 regularization_weight=0.05, plot_interval=5, debug_root: str = "./Data/Debug/", test: bool = True):
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
        :param test: is this a test mode? In test mode visualizations and superfluous steps will be skipped in favour of a
            faster execution to test the code
        """

        super().__init__(target_image, target_image_mask, source_image, source_image_mask, "Noiseprint", steps, alpha,
                         0.9, regularization_weight, plot_interval, False, debug_root, test)

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

        self.write_to_logs(
            "Quality factor to be used: {} {}".format(self.quality_factor, "(inferred)" if self.inferred else ""))

    def loss(self, y_pred, y_true):
        """
        Specify a loss function to drive the image we are attacking towards the target representation
        The default loss is the squared-error
        :param y_pred: last output of the model
        :param y_true: target representation
        :return: loss value
        """
        return tf.reduce_mean(tf.square(tf.subtract(y_pred, y_true)))

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
        image_one_channel = Picture((image_to_attack.one_channel() - self.noise).clip(0, 255))

        return super(BaseNoiseprintAttack, self).attack(image_one_channel, args, kwargs)

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

    def _get_gradient_of_image(self, image: Picture, target: Picture, old_perturbation: Picture = None):
        """
        Perform step of the attack executing the following steps:

            1) Divide the entire image into patches
            2) Compute the gradient of each patch with respect to the patch-tirget representation
            3) Recombine all the patch-gradients to obtain a image wide gradient
            4) Apply the image-gradient to the image
        :return: image_gradient, cumulative_loss
        """

        assert (len(image.shape) == 2)

        # variable to store the cumulative loss across all patches
        cumulative_loss = 0

        # image wide gradient
        image_gradient = np.zeros(image.shape)

        if image.shape[0] * image.shape[1] < NoiseprintEngine.large_limit:
            # the image can be processed as a single patch

            regularization_value = 0
            if old_perturbation is not None:
                regularization_value = np.linalg.norm(old_perturbation) * self.regularization_weight

            image_gradient, cumulative_loss = self._get_gradient_of_patch(image, target, regularization_value)

        else:
            # the image is too big, we have to divide it in patches to process separately
            # iterate over x and y, strides = self.slide, window size = self.slide+2*self.overlap
            for x in range(0, image.shape[0], self._engine.slide):
                x_start = x - self._engine.overlap
                x_end = x + self._engine.slide + self._engine.overlap
                for y in range(0, image.shape[1], self._engine.slide):
                    y_start = y - self._engine.overlap
                    y_end = y + self._engine.slide + self._engine.overlap

                    # get the patch we are currently working on
                    patch = image[
                            max(x_start, 0): min(x_end, image.shape[0]),
                            max(y_start, 0): min(y_end, image.shape[1])
                            ]

                    # get the desired target representation for this patch
                    target_patch = target[
                                   max(x_start, 0): min(x_end, image.shape[0]),
                                   max(y_start, 0): min(y_end, image.shape[1])
                                   ]

                    perturbation_patch = None
                    regularization_value = 0
                    if old_perturbation is not None:
                        perturbation_patch = old_perturbation[
                                             max(x_start, 0): min(x_end, image.shape[0]),
                                             max(y_start, 0): min(y_end, image.shape[1])
                                             ]
                        regularization_value = np.linalg.norm(perturbation_patch) * self.regularization_weight

                    patch_gradient, patch_loss = self._get_gradient_of_patch(patch, target_patch,
                                                                             regularization_value)

                    # discard initial overlap if not the row or first column
                    if x > 0:
                        patch_gradient = patch_gradient[self._engine.overlap:, :]
                    if y > 0:
                        patch_gradient = patch_gradient[:, self._engine.overlap:]

                    # add this patch loss to the total loss
                    cumulative_loss += patch_loss

                    # add this patch's gradient to the image gradient
                    # discard data beyond image size
                    patch_gradient = patch_gradient[:min(self._engine.slide, patch.shape[0]),
                                     :min(self._engine.slide, patch.shape[1])]

                    # copy data to output buffer
                    image_gradient[x: min(x + self._engine.slide, image_gradient.shape[0]),
                    y: min(y + self._engine.slide, image_gradient.shape[1])] = patch_gradient

        return image_gradient, cumulative_loss

    @property
    def attacked_image(self):
        """
        Compute the attacked image using the original image and the cumulative noise to reduce
        rounding artifacts caused by translating the noise from one to 3 channels and vie versa multiple times,
        still this operation here is done once so some rounding error is still present.
        Use attacked_image_monochannel to get the one channel version of the image withoud rounding errors
        :return:
        """
        return Picture((self.target_image - Picture(self.noise).three_channels(1 / 3, 1 / 3, 1 / 3)).clip(0, 255))

    @property
    def attacked_image_monochannel(self):
        """
        Compute the attacked image using the original image and the compulative noise to reduce
        rounding artifacts caused by translating the nosie from one to 3 channels and vie versa multiple times
        :return:
        """
        return Picture((self.target_image.one_channel() - self.noise).clip(0, 255))
