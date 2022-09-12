import os
from math import ceil

import numpy as np
from tensorflow.python.ops.gen_math_ops import squared_difference
from tqdm import tqdm

import tensorflow as tf
from Attacks.Noiseprint.Lots.BaseLots4Noiseprint import BaseLots4Noiseprint
from Detectors.Noiseprint.noiseprintEngine import normalize_noiseprint, NoiseprintEngine
from Detectors.Noiseprint.utility.utility import prepare_image_noiseprint
from Utilities.Image.Picture import Picture
from Utilities.Image.functions import visuallize_matrix_values


class Lots4NoiseprintAttackGlobalMap(BaseLots4Noiseprint):

    name = "LOTS global map Attack"

    def __init__(self, steps: int, alpha: float, patch_size=(8, 8), padding_size=(0, 0, 0, 0),
                 quality_factor=None, regularization_weight=0.0, plot_interval: int = 5,
                 debug_root: str = "./Data/Debug/",
                 verbosity: int = 2):
        """
        :param steps: number of attack iterations to perform
        :param alpha: strength of the attack
        :param patch_size: the size of the patches we will split the image in for analysis
        :param padding_size: the padding along each dimension that we will apply to each of the patches
        :param quality_factor: [101,51] specify if we need to load a noiseprint model for the specific given jpeg quality
               level, if it left to None, the right model will be inferred from the file
        :param regularization_weight: [0,1] importance of the regularization factor in the loss function
        :param plot_interval: how often (# steps) should the step-visualizations be generated?
        :param debug_root: root folder insede which to create a folder to store the data produced by the pipeline
        :param verbosity: is this a test mode? In test mode visualizations and superfluous steps will be skipped in favour of a
            faster execution to test the code
        """

        super().__init__(steps, alpha, 0, quality_factor, regularization_weight, plot_interval, debug_root, verbosity)

        self.patch_size = patch_size
        self.padding_size = padding_size

        self.gradient_normalization_margin = 0

    def _on_before_attack(self):
        """
        Save parameters into the logs
        :return:
        """
        super(Lots4NoiseprintAttackGlobalMap, self)._on_before_attack()
        self.logger_module.info("Analyzing the image by patches of size:{}".format(self.patch_size))
        self.logger_module.info("Padding patches on each dimension by:{}".format(self.padding_size))

    def _compute_target_representation(self, target_representation_source_image: Picture,
                                       target_representation_source_image_mask: Picture):
        """
        Generate the target representation executing the following steps:

            1) Generate an image wise noiseprint representation on the entire image
            2) Divide this noiseprint map into patches
            3) Average these patches
            4) Create an image wide target representation by tiling these patches together

        :return: the target representation in the shape of a numpy array
        """

        pad_size = ((self.padding_size[0], self.padding_size[2]), (self.padding_size[3], self.padding_size[1]))

        # convert the image in the standard required by noiseprint
        image = prepare_image_noiseprint(target_representation_source_image)

        # generate an image wise noiseprint representation on the entire image
        original_noiseprint = Picture(self._engine.predict(image))

        # cut away section along borders
        original_noiseprint[0:self.patch_size[0], :] = 0
        original_noiseprint[-self.patch_size[0]:, :] = 0
        original_noiseprint[:, 0:self.patch_size[1]] = 0
        original_noiseprint[:, -self.patch_size[1]:] = 0

        # exctract the authentic patches from the image
        authentic_patches = original_noiseprint.get_authentic_patches(target_representation_source_image_mask,
                                                                      self.patch_size, force_shape=True,
                                                                      zero_padding=False)

        # create target patch object
        target_patch = np.zeros(self.patch_size)

        patches_map = np.zeros(image.shape)

        for patch in tqdm(authentic_patches):
            assert (patch.clean_shape == target_patch.shape)

            target_patch += patch / len(authentic_patches)

            patches_map = patch.no_paddings().add_to_image(patches_map)

        # compute the tiling factors along the X and Y axis
        repeat_factors = (ceil(image.shape[0] / target_patch.shape[0]), ceil(image.shape[1] / target_patch.shape[1]))

        # tile the target representations together
        image_target_representation = np.tile(target_patch, repeat_factors)

        # cut away "overflowing" margins
        image_target_representation = image_target_representation[:image.shape[0], :image.shape[1]]

        # save tile visualization
        visuallize_matrix_values(target_patch, os.path.join(self.debug_folder, "image-target-raw.png"))

        patches_map = Picture(normalize_noiseprint(patches_map))
        patches_map.save(os.path.join(self.debug_folder, "patches-map.png"))

        return Picture(image_target_representation)

    def _get_gradient_of_image(self, image: Picture, target: Picture, old_perturbation: Picture = None):
        """
            Add option to support padding along the border of the patch
        :return: image_gradient, cumulative_loss
        """

        assert (len(image.shape) == 2)

        image = Picture(image)
        target = Picture(target)

        pad_size = ((self.padding_size[0], self.padding_size[2]), (self.padding_size[3], self.padding_size[1]))

        image = image.pad(pad_size, mode="reflect")
        target = target.pad(pad_size, mode="reflect")

        if old_perturbation is not None:
            old_perturbation = old_perturbation.pad(pad_size, "reflect")

        image_gradient, cumulative_loss = super()._get_gradient_of_image(image, target, old_perturbation)

        if self.padding_size[0] > 0:
            image_gradient = image_gradient[self.padding_size[0]:, :]

        if self.padding_size[1] > 0:
            image_gradient = image_gradient[:, self.padding_size[1]:]

        if self.padding_size[2] > 0:
            image_gradient = image_gradient[:-self.padding_size[2], :]

        if self.padding_size[3] > 0:
            image_gradient = image_gradient[:, :-self.padding_size[3]]

        return image_gradient, cumulative_loss

    def loss_function(self, y_pred, y_true):
        """
        Specify a loss function to drive the image we are attacking towards the target representation
        The default loss is the l2-norm
        :param y_pred: last output of the model
        :param y_true: target representation
        :return: loss value
        """
        return tf.reduce_sum(squared_difference(y_pred, y_true))

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