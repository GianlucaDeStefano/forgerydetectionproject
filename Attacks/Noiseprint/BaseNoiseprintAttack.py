from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf

from Attacks.BaseWhiteBoxAttack import BaseWhiteBoxAttack, normalize_gradient
from Detectors.Noiseprint.noiseprintEngine import NoiseprintEngine
from Detectors.Noiseprint.utility.utility import jpeg_quality_of_file
from Detectors.Noiseprint.utility.utilityRead import one_2_three_channels, three_2_one_channel, imconver_int_2_float
from Utilities.Image.Picture import Picture
from Utilities.Visualizers.NoiseprintVisualizer import NoiseprintVisualizer
from tensorflow.python.ops.gen_math_ops import squared_difference


class BaseNoiseprintAttack(BaseWhiteBoxAttack, ABC):
    """
        This class in used as a base to implement white box attacks on the noiseprint"
    """

    name = "Base Noiseprint Attack"

    def __init__(self, steps: int, alpha: float, momentum_coeficient: float = 0.5,
                 quality_factor=None,
                 regularization_weight=0, plot_interval=5, debug_root: str = "./Data/Debug/", verbosity: int = 2):
        """
        :param steps: number of attack iterations to perform
        :param alpha: strength of the attack
        :param momentum_coeficient: [0,1] how relevant is the velocity ferived from past gradients for computing the
            current gradient? 0 -> not relevant, 1-> is the only thing that matters
        :param quality_factor: [101,51] specify if we need to load a noiseprint model for the specific given jpeg quality
               level, if it left to None, the right model will be inferred from the file
        :param regularization_weight: [0,1] importance of the regularization factor in the loss function
        :param plot_interval: how often (# steps) should the step-visualizations be generated?
        :param debug_root: root folder insede which to create a folder to store the data produced by the pipeline
        :param verbosity: is this a test mode? In test mode visualizations and superfluous steps will be skipped in favour of a
            faster execution to test the code
        """

        visualizer = NoiseprintVisualizer()

        super().__init__(visualizer, steps, alpha, momentum_coeficient, regularization_weight, plot_interval, False,
                         debug_root, verbosity)

        self.quality_factor = quality_factor

        # reference directly the noiseprint engine class
        self._engine = visualizer._engine

        # create variable to store the generated adversarial noise
        self.noise = None

        # create variable to store the momentum of the gradient
        self.moving_avg_gradient = None

        # the amount of margin to apply to the gradient before normalization
        self.gradient_normalization_margin = 0

        # the batch size to use
        self.batch_size = 512

    def setup(self, target_image_path: Picture, target_image_mask: Picture, source_image_path: Picture = None,
              source_image_mask: Picture = None, target_forgery_mask: Picture = None):

        super().setup(target_image_path, target_image_mask, source_image_path, source_image_mask, target_forgery_mask)

        # create variable to store the generated adversarial noise
        self.noise = np.zeros((self.target_image.shape[0], self.target_image.shape[1]))

        # create variable to store the momentum of the gradient
        self.moving_avg_gradient = np.zeros((self.target_image.shape[0], self.target_image.shape[1]))

        self.quality_factor = self._engine.metadata["quality_level"]

    def _on_before_attack(self):
        """
        Instructions executed before performing the attack, writing logs
        :return:
        """

        super()._on_before_attack()

        self.logger_module.info("Quality factor to be used: {} {}".format(self.quality_factor, "(inferred)"))

    def _get_gradients_of_patches(self, patches_list: list, target_list: list, perturbations=None):
        """
        Given an  input image and a target, compute the gradient of the target w.r.t. the inputs on the loaded noiseprint model
        :param patches_list: list of patches for which we want to compute the gradient
        :param target_list: list target representations we want our model to produce
        :param perturbation:perturbation for which to compute the regularization value
        :return: gradient,loss
        """

        # check that input image and target rerpresentation have the same shape
        for i, patch in enumerate(patches_list):
            assert (patch.shape == target_list[i].shape)

        patches = np.array(patches_list)
        targets = np.array(target_list)

        # prepare the tape object to compute the gradient
        with tf.GradientTape(persistent=True) as tape:
            # convert the input batch into a tensor
            input_tensor = tf.convert_to_tensor(patches[:, :, :, np.newaxis])

            # convert the targets into another tensor
            output_tensor = tf.convert_to_tensor(targets, tf.float32)

            # watch the tensor to compute a gradient on it
            tape.watch(input_tensor)

            # perform feed forward pass
            noiseprint_maps = tf.squeeze(self._engine.model(input_tensor))

            # compute the loss with respect to the target representation
            loss = self.loss(noiseprint_maps, output_tensor, perturbations)

            # retrieve the gradient of the image
            gradients = np.squeeze(tape.gradient(loss, input_tensor).numpy())

            # check that the retrieved gradient has the correct shape
            return gradients, tf.reduce_mean(loss).numpy()

    def _get_gradient_of_image(self, image: Picture, target: Picture, old_perturbation: Picture = None):
        """
        Perform step of the attack executing the following steps:
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

        n_patches = 0

        if image.shape[0] * image.shape[1] < NoiseprintEngine.large_limit:

            # the image can be processed as a single patch
            regularization_value = 0
            if old_perturbation is not None:
                regularization_value = np.linalg.norm(old_perturbation) * self.regularization_weight

            image_gradient, cumulative_loss = self._get_gradients_of_patches([image], [target], regularization_value)

            n_patches = 1

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

                    regularization_value = 0
                    if old_perturbation is not None:
                        perturbation_patch = old_perturbation[
                                             max(x_start, 0): min(x_end, image.shape[0]),
                                             max(y_start, 0): min(y_end, image.shape[1])
                                             ]
                        regularization_value = np.linalg.norm(perturbation_patch) * self.regularization_weight

                    patch_gradient, patch_loss = self._get_gradients_of_patches([patch], [target_patch],
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

                    n_patches += 1

        return image_gradient, cumulative_loss / n_patches

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

        # format the image to attack in the required shape and format
        image_one_channel = Picture((image_to_attack.one_channel() - self.noise).clip(0, 255)).to_float()

        # compute the gradient
        image_gradient, loss = self._get_gradient_of_image(image_one_channel, self.target_representation,
                                                           Picture(self.noise))

        # save loss value to plot it
        self.loss_steps.append(loss)

        # normalize the gradient
        image_gradient = normalize_gradient(image_gradient, 0) * self.alpha

        # add this iteration contribution to the cumulative noise
        self.noise += image_gradient

        return self.attacked_image, loss

    @property
    def attacked_image(self):
        """
        Compute the attacked image using the original image and the cumulative noise to reduce
        rounding artifacts caused by translating the noise from one to 3 channels and vie versa multiple times,
        still this operation here is done once so some rounding error is still present.
        Use attaczked_image_monochannel to get the one channel version of the image withoud rounding errors
        :return:
        """
        return Picture(self.target_image - one_2_three_channels(self.noise)).clip(0, 255)

    @property
    def attacked_image_monochannel(self):
        """
        Compute the attacked image using the original image and the compulative noise to reduce
        rounding artifacts caused by translating the nosie from one to 3 channels and vie versa multiple times
        :return:
        """
        return Picture((self.target_image.one_channel() - self.noise / 255).clip(0, 1))

    def loss_function(self, y_pred, y_true):
        """
        Specify a loss function to drive the image we are attacking towards the target representation
        The default loss is the l2-norm
        :param y_pred: last output of the model
        :param y_true: target representation
        :return: loss value
        """

        return tf.reduce_sum(squared_difference(y_pred, y_true), [1, 2])

    def regularizer_function(self, perturbation=None):
        """
        Compute te regularization value to add to the loss function
        :param perturbation:perturbation for which to compute the regularization value
        :return: regularization value
        """

        # if no perturbation is given return 0
        if perturbation is None:
            return 0

        return tf.norm(perturbation, ord='euclidean')