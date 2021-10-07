import os
from math import ceil

import numpy as np
from tqdm import tqdm

from Attacks.Noiseprint.Lots.BaseLots4Noiseprint import BaseLots4Noiseprint
from Detectors.Noiseprint.noiseprintEngine import normalize_noiseprint, NoiseprintEngine
from Detectors.Noiseprint.utility.utility import prepare_image_noiseprint
from Ulitities.Image.Picture import Picture
from Ulitities.Image.functions import visuallize_matrix_values


class Lots4NoiseprintAttackGlobalMap(BaseLots4Noiseprint):

    def __init__(self, target_image: Picture, target_image_mask: Picture,
                 steps: int, alpha: float, patch_size=(16, 16), padding_size=(0, 0, 0, 0),
                 quality_factor=None, regularization_weight=0.1, plot_interval: int = 5,
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

        super().__init__(target_image, target_image_mask, target_image, target_image_mask, steps, alpha,0, quality_factor,
                         regularization_weight, plot_interval, debug_root, test)

        self.patch_size = patch_size
        self.padding_size = padding_size

        self.gradient_normalization_margin = 8

    def _on_before_attack(self):
        """
        Save parameters into the logs
        :return:
        """
        super(Lots4NoiseprintAttackGlobalMap, self)._on_before_attack()
        self.write_to_logs("Analyzing the image by patches of size:{}".format(self.patch_size))
        self.write_to_logs("Padding patches on each dimension by:{}".format(self.padding_size))

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

        # conver the image in the standard required by noiseprint
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
        Perform step of the attack executing the following steps:

            1) Divide the entire image into patches
            2) Compute the gradient of each patch with respect to the patch-tirget representation
            3) Recombine all the patch-gradients to obtain a image wide gradient
            4) Apply the image-gradient to the image
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

        # variable to store the cumulative loss across all patches
        cumulative_loss = 0

        # image wide gradient
        image_gradient = np.zeros(image.shape)

        if image.shape[0] * image.shape[1] < NoiseprintEngine.large_limit:
            # the image can be processed as a single patch
            image_gradient, cumulative_loss = self._get_gradient_of_patch(image, target)

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
                    if old_perturbation is not None:
                        perturbation_patch = old_perturbation[
                                             max(x_start, 0): min(x_end, image.shape[0]),
                                             max(y_start, 0): min(y_end, image.shape[1])
                                             ]

                    patch_gradient, patch_loss = self._get_gradient_of_patch(patch, target_patch)

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

        if self.padding_size[0] > 0:
            image_gradient = image_gradient[self.padding_size[0]:, :]

        if self.padding_size[1] > 0:
            image_gradient = image_gradient[:, self.padding_size[1]]

        if self.padding_size[2] > 0:
            image_gradient = image_gradient[:-self.padding_size[2], :]

        if self.padding_size[3] > 0:
            image_gradient = image_gradient[:, :-self.padding_size[3]]

        return image_gradient, cumulative_loss
