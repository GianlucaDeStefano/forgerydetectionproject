import argparse
import os
from math import ceil

import tensorflow as tf
import numpy as np
from scipy.ndimage import binary_dilation
from tqdm import tqdm

from Attacks import LotsNoiseprint2
from Attacks.Lots.Noiseprint.Lots4Noiseprint3.Lots4Noiseprint3 import LotsNoiseprint3
from Attacks.Lots.Noiseprint.Lots4NoiseprintBase import Lots4NoiseprintBase, normalize_gradient
from Attacks.utilities.visualization import visuallize_matrix_values
from Datasets import get_image_and_mask
from Detectors.Noiseprint.noiseprintEngine import NoiseprintEngine, normalize_noiseprint
from Ulitities.Image.Patch import Patch
from Ulitities.Image.Picture import Picture


class LotsNoiseprint4(LotsNoiseprint3):
    name = "LOTS4Noiseprint4"

    def __init__(self, objective_image: Picture, objective_mask: Picture, target_representation_image: Picture = None,
                 target_representation_mask: Picture = None, qf: int = None,
                 patch_size: tuple = (32, 32), padding_size=(0, 0, 0, 0),
                 steps=50, debug_root="./Data/Debug/", alpha=1, plot_interval=1, verbose=True):
        assert (objective_image.shape[0] <= target_representation_image.shape[0])
        assert (objective_image.shape[1] <= target_representation_image.shape[1])

        super().__init__(objective_image, objective_mask, target_representation_image,
                         target_representation_mask, qf, patch_size, padding_size, steps,
                         debug_root, alpha, plot_interval, verbose)

    def _generate_target_representation(self, image: Picture, mask: Picture):
        """
        Generate the target representation executing the following steps:

            1) Generate an image wise noiseprint representation on the entire image
            2) Divide this noiseprint map into patches
            3) Average these patches
            4) Create an image wide target representation by tiling these patches together

        :return: the target representation in the shape of a numpy array
        """

        # generate an image wise noiseprint representation on the entire image
        original_noiseprint = Picture(self._engine.predict(image))

        return original_noiseprint