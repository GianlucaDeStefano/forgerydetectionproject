import argparse
import os
import random
from pathlib import Path
import numpy as np
from tqdm import tqdm
from Attacks.Noiseprint.BaseNoiseprintAttack import BaseNoiseprintAttack
from Attacks.Noiseprint.Mimiking.BaseMimickin4Noiseprint import BaseMimicking4Noiseprint
from Detectors.Noiseprint.utility.utility import prepare_image_noiseprint
from Utilities.Image.Picture import Picture
from Utilities.Image.functions import visuallize_matrix_values
import tensorflow as tf


class NoiseprintGlobalIntelligentMimickingAttack(BaseMimicking4Noiseprint):
    name = "Noiseprint intelligent mimicking attack"

    def __init__(self, steps: int, alpha: float, patch_size=(8, 8), quality_factor=None,
                 regularization_weight=0, plot_interval=5, debug_root: str = "./Data/Debug/", verbosity: int = 2):
        """
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

        super().__init__(steps, alpha, 0.0, quality_factor, regularization_weight, plot_interval, debug_root, verbosity)

        self.patch_size = patch_size

        self.padding_size = (8, 8, 8, 8)

        self.k = 5

    def setup(self, target_image_path: Picture, target_image_mask: Picture, source_image_path: Picture = None,
              source_image_mask: Picture = None, target_forgery_mask: Picture = None):
        """
        :param source_image_path: image from which we will compute the target representation
        :param source_image_mask: mask of the imae from which we will compute the target representation
        :param target_forgery_mask: mask highlighting the section of the image that should be identified as forged after the attack
        :return:
        """

        super().setup(target_image_path, target_image_mask, source_image_path, source_image_mask, target_forgery_mask)

    def _compute_target_representation(self, target_representation_source_image: Picture,
                                       target_representation_source_image_mask: Picture):
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

        target_forgery_mask = self.target_forgery_mask

        assert target_forgery_mask is not None

        # check that the passed target_forgery_mask has a valid shape
        assert (target_forgery_mask.shape[0] == self.target_image.shape[0])
        assert (target_forgery_mask.shape[1] == self.target_image.shape[1])

        target_representation_source_image = prepare_image_noiseprint(target_representation_source_image)

        complete_patch_size = (self.patch_size[0] + self.padding_size[1] + self.padding_size[3],
                               self.patch_size[1] + self.padding_size[0] + self.padding_size[2])

        authentic_patches = target_representation_source_image.get_authentic_patches(
            target_representation_source_image_mask, self.patch_size, self.padding_size,
            force_shape=True, zero_padding=True)

        average_authentic_patch = np.zeros(complete_patch_size)

        for base_index in tqdm(range(0, len(authentic_patches), self.batch_size), disable=self.clean_execution):

            patches = authentic_patches[base_index:min(len(authentic_patches), base_index + self.batch_size)]

            # compute its noiseprint
            noiseprint_patches = np.squeeze(self._engine.model(np.array(patches)[:, :, :, np.newaxis]))

            for i, noiseprint_patch in enumerate(noiseprint_patches):
                # add the noiseprint to the mean target patch object
                average_authentic_patch += noiseprint_patch / len(authentic_patches)

        t_no_padding = authentic_patches[0].no_paddings(average_authentic_patch)

        visuallize_matrix_values(t_no_padding, os.path.join(self.debug_folder, "average_authentic_patch.png"))

        if target_representation_source_image_mask.max() == 0:
            average_forged_patch = average_authentic_patch
        else:
            forged_patches = target_representation_source_image.get_forged_patches(
                target_representation_source_image_mask, self.patch_size, self.padding_size,
                force_shape=True, zero_padding=True)

            average_forged_patch = np.zeros(complete_patch_size)

            for base_index in tqdm(range(0, len(forged_patches), self.batch_size), disable=self.clean_execution):

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

        average_authentic_patch = np.zeros(average_authentic_patch.shape)

        target_map = np.zeros(target_forgery_mask.shape)

        for target_mask_patch in target_mask_patches:

            if target_mask_patch.max() == 0:
                target_map = target_mask_patch.add_to_image(target_map, np.zeros(average_authentic_patch.shape))
            else:
                target_map = target_mask_patch.add_to_image(target_map, average_forged_patch * self.k)

        return target_map

    @staticmethod
    def read_arguments(dataset_root) -> tuple:
        """
        Read arguments from the command line or ask for them if they are not present, validate them raising
        an exception if they are invalid, it is called by the launcher script
        :param args: args dictionary containing the arguments passed while launching the program
        :return: kwargs to pass to the attack
        """
        attack_parameters, setup_parameters = BaseNoiseprintAttack.read_arguments(dataset_root)

        parser = argparse.ArgumentParser()
        parser.add_argument('--target_forgery_mask', required=False, default=None,
                            help='Path of the mask highlighting the section of the image that should be identified as '
                                 'forged')
        parser.add_argument('--target_forgery_id', required=False, type=int, default=None,
                            help='Id of the target_forgery type to use to autogenerate the target_forgery map')
        args = parser.parse_known_args()[0]

        target_forgery_mask_path = args.target_forgery_mask
        target_forgery_id = args.target_forgery_id

        if target_forgery_mask_path is not None:
            mask_path = Path(target_forgery_mask_path)

            if mask_path.exists():
                mask = np.where(np.all(Picture(str(mask_path)) == (255, 255, 255), axis=-1), 1, 0)
            else:
                raise Exception("Target forgery mask not found")

            setup_parameters["target_forgery_mask"] = Picture(mask)
        else:
            if input(
                    "No target_forgery_mask has been specified. Do you want to use one generated randomly? (y/n)") == 'y':
                setup_parameters["target_forgery_mask"] = None
            else:
                exit(0)
        return attack_parameters, setup_parameters
