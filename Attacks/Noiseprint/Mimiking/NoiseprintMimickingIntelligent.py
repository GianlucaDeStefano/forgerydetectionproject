import argparse
import os
from pathlib import Path

import numpy as np
from numpy.linalg import linalg
from tensorflow.python.ops.gen_math_ops import squared_difference
from tqdm import tqdm

from Attacks.Noiseprint.BaseNoiseprintAttack import BaseNoiseprintAttack
from Attacks.Noiseprint.Mimiking.BaseMimickin4Noiseprint import BaseMimicking4Noiseprint
from Datasets import get_image_and_mask, ImageNotFoundError
from Detectors.Noiseprint.noiseprintEngine import NoiseprintEngine, normalize_noiseprint
from Detectors.Noiseprint.utility.utility import prepare_image_noiseprint
from Ulitities.Image.Picture import Picture
from Ulitities.Image.functions import visuallize_matrix_values
import tensorflow as tf


class NoiseprintIntelligentMimickingAttack(BaseMimicking4Noiseprint):
    name = "Noiseprint intelligent mimicking attack"

    def __init__(self, target_image: Picture, target_image_mask: Picture, target_forgery_mask: Picture
                 , steps: int, alpha: float, patch_size=(8, 8), quality_factor=None,
                 regularization_weight=0.05, plot_interval=5, debug_root: str = "./Data/Debug/", verbosity: int = 2):
        """
        :param target_image: original image on which we should perform the attack
        :param target_image_mask: original mask of the image on which we should perform the attack
        :param target_forgery_mask: mask highlighting the section of the image that should be identified as forged after the attack
        :param steps: number of attack iterations to perform
        :param alpha: strength of the attack
        :param patch_size: the size of the patches we will split the image in for analysis
        :param quality_factor: [101,51] specify if we need to load a noiseprint model for the specific given jpeg quality
               level, if it left to None, the right model will be inferred from the file
        :param regularization_weight: [0,1] importance of the regularization factor in the loss function
        :param plot_interval: how often (# steps) should the step-visualizations be generated?
        :param debug_root: root folder insede which to create a folder to store the data produced by the pipeline
        :param verbosity: is this a test mode? In test mode visualizations and superfluous steps will be skipped in favour of a
            faster execution to test the code
        """

        super().__init__(target_image, target_image_mask, target_image, target_image_mask, steps, alpha, 0.0,
                         quality_factor,
                         regularization_weight, plot_interval, debug_root, verbosity)

        self.target_forgery_mask = target_forgery_mask

        self.patch_size = patch_size

        # for this technique no padding is needed
        self.padding_size = (8, 8, 8, 8)

        self.k = 3

    def _compute_target_representation(self, target_representation_source_image: Picture,
                                       target_representation_source_image_mask: Picture,
                                       target_forgery_mask: Picture = None):
        """
        For this technique the target reprsentation is computed in the following way:
            (1) -> Compute the Noiseprint
            (2) -> Divide the Noiseprint map into 8x8 authentic and forged patches
            (3) -> Compute the average authentic map and the average forged map
            (4) -> Compute the target representation using the target_forgery mask, applying on its authentic sections
                    the average Noiseprint patch while applying on its forged section the average forged patch
        :param target_representation_source_image: image source of the target representation
        :param target_representation_source_image_mask: mask of the image source of the target representation
        :param target_forgery_mask: mask of the forgery we want to be highlighted after the attack
        :return:
        """

        if target_forgery_mask is None:
            target_forgery_mask = self.target_forgery_mask

        # check that the passed target_forgery_mask has a valid shape
        assert (target_forgery_mask.shape[0] == target_representation_source_image.shape[0])
        assert (target_forgery_mask.shape[1] == target_representation_source_image.shape[1])

        target_representation_source_image = prepare_image_noiseprint(target_representation_source_image)

        complete_patch_size = (self.patch_size[0] + self.padding_size[1] + self.padding_size[3],
                               self.patch_size[1] + self.padding_size[0] + self.padding_size[2])

        authentic_patches = target_representation_source_image.get_authentic_patches(
            target_representation_source_image_mask, self.patch_size, self.padding_size,
            force_shape=True, zero_padding=True)

        average_authentic_patch = np.zeros(complete_patch_size)

        for base_index in tqdm(range(0, len(authentic_patches), self.batch_size)):

            patches = authentic_patches[base_index:min(len(authentic_patches), base_index + self.batch_size)]

            # compute its noiseprint
            noiseprint_patches = np.squeeze(self._engine.model(np.array(patches)[:, :, :, np.newaxis]))

            for i, noiseprint_patch in enumerate(noiseprint_patches):
                # add the noiseprint to the mean target patch object
                average_authentic_patch += noiseprint_patch / len(authentic_patches)

        t_no_padding = authentic_patches[0].no_paddings(average_authentic_patch)

        visuallize_matrix_values(t_no_padding, os.path.join(self.debug_folder, "average_authentic_patch.png"))

        if target_representation_source_image_mask.max() == 0:
            average_forged_patch = - average_authentic_patch
        else:
            forged_patches = target_representation_source_image.get_forged_patches(
                target_representation_source_image_mask, self.patch_size, self.padding_size,
                force_shape=True, zero_padding=True)

            average_forged_patch = np.zeros(complete_patch_size)

            for base_index in tqdm(range(0, len(forged_patches), self.batch_size)):

                patches = forged_patches[base_index:min(len(forged_patches), base_index + self.batch_size)]

                # compute its noiseprint
                noiseprint_patches = np.squeeze(self._engine.model(np.array(patches)[:, :, :, np.newaxis]))

                for i, noiseprint_patch in enumerate(noiseprint_patches):
                    # add the noiseprint to the mean target patch object
                    average_forged_patch += noiseprint_patch / len(forged_patches)

        t_no_padding = authentic_patches[0].no_paddings(average_forged_patch)
        visuallize_matrix_values(t_no_padding, os.path.join(self.debug_folder, "average_forged_patch.png"))

        target_mask_patches = target_forgery_mask.divide_in_patches(self.patch_size, self.padding_size,
                                                                    zero_padding=True)

        targets_list = []

        for target_mask_patch in target_mask_patches:

            if target_mask_patch.max() == 0:
                targets_list.append(average_authentic_patch*self.k)
            else:
                targets_list.append(average_forged_patch*self.k)

        return targets_list

    def _get_gradient_of_image(self, image: Picture, target: Picture, old_perturbation: Picture = None):
        """
        Compute the gradient on the entire image by executing the following steps:
            1) Divide the entire image into patches
            2) Compute the gradient of each patch with respect to the patch-target representation
            3) Recombine all the patch-gradients to obtain a image wide gradient
        :return: gradient, loss
        """

        # make sure tha image is a picture
        image = Picture(image)

        # variable to store the cumulative loss across all patches
        cumulative_loss = 0

        # image wide gradient
        image_gradient = np.zeros((image.shape[0:2]))

        # divide the image into patches
        img_patches = image.divide_in_patches(self.patch_size, self.padding_size, zero_padding=True)
        noise_patches = Picture(self.noise).divide_in_patches(self.patch_size, self.padding_size, zero_padding=True)

        assert (len(img_patches) == len(target))

        # analyze the image using batch of patches
        for base_index in tqdm(range(0, len(img_patches), self.batch_size), disable=self.clean_execution):

            # retieve this batch's patches form the list
            patches = img_patches[base_index:min(len(img_patches), base_index + self.batch_size)]

            # retrieve the noise patches from the list
            perturbations = noise_patches[base_index:min(len(img_patches), base_index + self.batch_size)]

            targets = target[base_index:min(len(img_patches), base_index + self.batch_size)]

            # check if we are on a border and therefore we have to "cut"tareget representation
            # if we are on a border, cut away the "overflowing target representation"
            for i, patch in enumerate(patches):
                if targets[i].shape != patch.shape:
                    targets[i] = targets[i][:patch.shape[0], :patch.shape[1]]

            # compute the gradient of the input w.r.t. the target representation
            patches_gradient, patch_loss = self._get_gradients_of_patches(patches, targets, perturbations)

            patch_loss = np.sum(patch_loss) / self.batch_size

            # add this patch's loss contribution
            cumulative_loss += patch_loss

            for i, patch in enumerate(patches):
                # Add the contribution of this patch to the image wide gradient removing the padding
                image_gradient = patch.add_to_image(image_gradient, patches_gradient[i])

        return image_gradient, cumulative_loss / (len(img_patches) // self.batch_size)

    @staticmethod
    def read_arguments(dataset_root) -> dict:
        """
        Read arguments from the command line or ask for them if they are not present, validate them raising
        an exception if they are invalid, it is called by the launcher script
        :param args: args dictionary containing the arguments passed while launching the program
        :return: kwargs to pass to the attack
        """
        kwarg = BaseNoiseprintAttack.read_arguments(dataset_root)

        parser = argparse.ArgumentParser()
        parser.add_argument('--target_forgery_mask', required=True,
                            help='Path of the mask highlighting the section of the image that should be identified as '
                                 'forged')
        args = parser.parse_known_args()[0]

        target_forgery_mask_path = args.target_forgery_mask

        mask_path = Path(target_forgery_mask_path)

        if mask_path.exists():
            mask = np.where(np.all(Picture(str(mask_path)) == (255, 255, 255), axis=-1), 1, 0)
        else:
            raise Exception("Target forgery mask not found")

        kwarg["target_forgery_mask"] = Picture(mask)

        return kwarg

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

        return tf.norm(perturbation, ord='euclidean', axis=[1, 2])
