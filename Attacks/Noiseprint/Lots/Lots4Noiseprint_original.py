import os

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from Attacks.Noiseprint.BaseNoiseprintAttack import BaseNoiseprintAttack
from Detectors.Noiseprint.utility.utility import prepare_image_noiseprint, normalize_noiseprint_no_margins
from Ulitities.Image.Picture import Picture
from Ulitities.Image.functions import visuallize_matrix_values


class Lots4NoiseprintAttackOriginal(BaseNoiseprintAttack):

    def __init__(self, target_image: Picture, target_image_mask: Picture, steps: int, alpha: float, patch_size=(16, 16),
                 padding_size=(0, 0, 0, 0), quality_factor=None, regularization_weight=0.1, plot_interval: int = 5,
                 debug_root: str = "./Data/Debug/",
                 test: bool = True):
        """
        :param target_image: original image on which we should perform the attack
        :param target_image_mask: original mask of the image on which we should perform the attack
        :param steps: number of attack iterations to perform
        :param alpha: strength of the attack
        :param patch_size: the size of the patches we will split the image in for analysis
        :param padding_size: the padding along each dimension that we will apply to each of the patches
        :param quality_factor: [101,51] specify if we need to load a noiseprint model for the specific given jpeg quality
               level, if it left to None, the right model will be inferred from the file
        :param regularization_weight: [0,1] importance of the regularization factor in the loss function
        :param plot_interval: how often (# steps) should the step-visualizations be generated?
        :param debug_root: root folder insede which to create a folder to store the data produced by the pipeline
        :param test: is this a test mode? In test mode visualizations and superfluous steps will be skipped in favour of a
            faster execution to test the code
        """

        super().__init__(target_image, target_image_mask, target_image, target_image_mask, steps, alpha, quality_factor,
                         regularization_weight, plot_interval, debug_root, test)

        self.patch_size = patch_size
        self.padding_size = padding_size

    def _on_before_attack(self):
        """
        Save parameters into the logs
        :return:
        """
        super(Lots4NoiseprintAttackOriginal, self)._on_before_attack()

        self.write_to_logs("Analyzing the image by patches of size:{}".format(self.patch_size))
        self.write_to_logs("Padding patches on each dimension by:{}".format(self.padding_size))

    def _compute_target_representation(self, target_representation_source_image: Picture,
                                       target_representation_source_image_mask: Picture):
        """
        Generate the target representation executing the following steps:

            1) Divide the image into patches
            2) Select only the authentic patches
            3) Foreach patch compute its noiseptint
            4) Average all the noiseprint maps

        :return: the target representation in the shape of a numpy array
        """

        # format the target image in the standard that the noiseprint requires
        target_representation_source_image = prepare_image_noiseprint(target_representation_source_image)

        # spit the image into patches
        authentic_patches = target_representation_source_image.get_authentic_patches(
            target_representation_source_image_mask, self.patch_size, self.padding_size,
            force_shape=True, zero_padding=True)

        complete_patch_size = (self.patch_size[0] + self.padding_size[1] + self.padding_size[3],
                               self.patch_size[1] + self.padding_size[0] + self.padding_size[2])

        # create target patch object
        target_patch = np.zeros(complete_patch_size)

        # create a map in which to store the used patches for visualization
        patches_map = np.zeros(target_representation_source_image.shape)

        # generate authentic target representation
        self.write_to_logs("Generating target representation...")

        # foreach authentic patch
        for original_patch in tqdm(authentic_patches):
            assert (original_patch.shape == target_patch.shape)

            # compute its noiseprint
            noiseprint_patch = np.squeeze(self._engine.model(original_patch[np.newaxis, :, :, np.newaxis]))

            # add the noiseprint to the mean target patch object
            target_patch += noiseprint_patch / len(authentic_patches)

            # add the result to the map of patches
            patches_map = original_patch.no_paddings().add_to_image(patches_map)

        self.write_to_logs("Target representation generated")

        t_no_padding = authentic_patches[0].no_paddings(target_patch)

        # save a visualization of the target representation
        normalized_noiseprint = normalize_noiseprint_no_margins(t_no_padding)

        plt.imsave(fname=os.path.join(self.debug_folder, "image-target.png"), arr=normalized_noiseprint,
                   cmap='gray',
                   format='png')

        visuallize_matrix_values(t_no_padding, os.path.join(self.debug_folder, "image-target-raw.png"))

        patches_map = Picture(patches_map)
        patches_map.save(os.path.join(self.debug_folder, "patches-map.png"))

        # save target representation t in a 8x8 grid for visualization purposes
        if t_no_padding.shape[0] % 8 == 0 and t_no_padding.shape[1] % 8 == 0:

            patch_8 = np.zeros((8, 8))
            n_patches8 = (t_no_padding.shape[0] // 8) * (t_no_padding.shape[1] // 8)
            for x in range(0, t_no_padding.shape[0], 8):
                for y in range(0, t_no_padding.shape[1], 8):
                    patch_8 += t_no_padding[x:x + 8, y:y + 8] / n_patches8

            visuallize_matrix_values(patch_8, os.path.join(self.debug_folder, "clean_target_patch.png"))

        return target_patch

    def _get_gradient_of_image(self, image: Picture, target: Picture, old_perturbation: Picture = None):
        """
        Compute the gradient on the entire imabe by executing the following steps:
            1) Divide the entire image into patches
            2) Compute the gradient of each patch with respect to the patch-tirget representation
            3) Recombine all the patch-gradients to obtain a image wide gradient
            4) Apply the image-gradient to the image
            5) Convert then the image to the range of values of integers [0,255] and convert it back to the range
               [0,1]
        :return: gradient, loss
        """

        # variable to store the cumulative loss across all patches
        cumulative_loss = 0

        # image wide gradient
        image_gradient = np.zeros((image.shape[0:2]))

        # divide the image into patches
        img_patches = image.divide_in_patches(self.patch_size, self.padding_size, zero_padding=True)

        # analyze the image patch by patch
        for patch in tqdm(img_patches):

            # check if we are on a border and therefore we have to "cut"tareget representation
            target_patch_representation = target

            # if we are on a border, cut away the "overflowing target representation"
            if target_patch_representation.shape != patch.shape:
                target_patch_representation = target_patch_representation[:patch.shape[0],
                                              :patch.shape[1]]

            # compute the gradient of the input w.r.t. the target representation
            patch_gradient, patch_loss = self._get_gradient_of_patch(patch, target_patch_representation)

            # check that the retrieved gradient has the correct shape
            assert (patch_gradient.shape == patch.shape)

            # add this patch's loss contribution
            cumulative_loss += patch_loss

            # remove padding from the gradient
            patch_gradient = patch.no_paddings(patch_gradient)

            # Add the contribution of this patch to the image wide gradient removing the padding
            image_gradient = patch.add_to_image(image_gradient, patch_gradient)

        return image_gradient, cumulative_loss
