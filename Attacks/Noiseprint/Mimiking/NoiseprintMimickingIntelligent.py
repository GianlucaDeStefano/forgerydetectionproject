import argparse
import os
from pathlib import Path

import numpy as np

from Attacks.Noiseprint.BaseNoiseprintAttack import BaseNoiseprintAttack
from Datasets import get_image_and_mask, ImageNotFoundError
from Detectors.Noiseprint.noiseprintEngine import NoiseprintEngine, normalize_noiseprint
from Detectors.Noiseprint.utility.utility import prepare_image_noiseprint
from Ulitities.Image.Picture import Picture
from Ulitities.Image.functions import visuallize_matrix_values


class NoiseprintIntelligentMimickingAttack(BaseNoiseprintAttack):
    name = "Noiseprint intelligent mimicking attack"

    def __init__(self, target_image: Picture, target_image_mask: Picture, target_forgery_mask: Picture
                 , steps: int, alpha: float, patch_size=(8, 8), quality_factor=None,
                 regularization_weight=0.05, plot_interval=5, debug_root: str = "./Data/Debug/", test: bool = True):
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
        :param test: is this a test mode? In test mode visualizations and superfluous steps will be skipped in favour of a
            faster execution to test the code
        """

        super().__init__(target_image, target_image_mask, target_image, target_image_mask, steps, alpha,0.9, quality_factor,
                         regularization_weight, plot_interval, debug_root, test)

        self.target_forgery_mask = target_forgery_mask

        self.patch_size = patch_size

        # for this technique no padding is needed
        self.padding_size = (0, 0, 0, 0)

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

        # prepare the image to be fed into the noiseprint model
        image = prepare_image_noiseprint(target_representation_source_image)

        # generate an image wise noiseprint representation on the entire image
        original_noiseprint = Picture(self._engine.predict(image))

        # spit the noiseprint map into authentic patches, discard the rest
        authentic_noiseprint_patches = original_noiseprint.get_authentic_patches(
            target_representation_source_image_mask, self.patch_size, self.padding_size,
            force_shape=True, zero_padding=False)

        # compute the average authentic patch
        authentic_average_patch = np.zeros(self.patch_size)
        for patch in authentic_noiseprint_patches:
            authentic_average_patch += patch / len(authentic_noiseprint_patches)

        # compute the average forged patch
        forged_average_patch = - authentic_average_patch

        # visualize the 2 target representtions
        visuallize_matrix_values(authentic_average_patch, os.path.join(self.debug_folder, "authentic-patch.png"))
        visuallize_matrix_values(forged_average_patch, os.path.join(self.debug_folder, "forged-patch.png"))

        # compute the target representation by pasting on authentic patches of the target_forgery_mask the average
        # authentic patch, and on forgered patches of the target_forgery_mask the average forged patch
        target_representation = np.zeros(original_noiseprint.shape)

        # object for storing a mask showng the distribution of authentic and forged patches in the final
        # target representatation
        target_representation_mask = np.zeros(original_noiseprint.shape)
        patches_target_forgery_mask = target_forgery_mask.divide_in_patches(self.patch_size, self.padding_size, False,
                                                                            False)
        for patch in patches_target_forgery_mask:
            if patch.no_paddings().sum() == 0:
                # this patch is authentic, apply average authentic patch
                target_representation = patch.no_paddings().add_to_image(target_representation, authentic_average_patch)
                target_representation_mask = patch.no_paddings().add_to_image(target_representation_mask,
                                                                              np.zeros(authentic_average_patch.shape))

            else:
                # this patch is forged apply average forged patch
                target_representation = patch.no_paddings().add_to_image(target_representation, forged_average_patch)
                target_representation_mask = patch.no_paddings().add_to_image(target_representation_mask,
                                                                              np.ones(forged_average_patch.shape))

        Picture(target_representation_mask).save(
            os.path.join(self.debug_folder, "target representation patch mask.png"))

        Picture(normalize_noiseprint(target_representation)).save(
            os.path.join(self.debug_folder, "target representation.png"))

        return target_representation

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
