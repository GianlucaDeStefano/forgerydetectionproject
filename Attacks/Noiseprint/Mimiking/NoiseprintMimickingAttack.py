import argparse
import os
from pathlib import Path

import numpy as np
from tqdm import tqdm

from Attacks.BaseWhiteBoxAttack import BaseWhiteBoxAttack
from Attacks.Noiseprint.BaseNoiseprintAttack import normalize_gradient, BaseNoiseprintAttack
from Datasets import get_image_and_mask, ImageNotFoundError
from Detectors.Noiseprint.noiseprintEngine import NoiseprintEngine
from Detectors.Noiseprint.utility.utility import prepare_image_noiseprint
from Ulitities.Image.Picture import Picture


class NoiseprintMimickingAttack(BaseNoiseprintAttack):
    name = "Noiseprint mimicking attack"

    def __init__(self, target_image: Picture, target_image_mask: Picture, source_image: Picture,
                 source_image_mask: Picture, steps: int, alpha: float,quality_factor=None,regularization_weight=0,
                 plot_interval: int = 5, debug_root: str = "./Data/Debug/", verbose: bool = True):
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

        super().__init__(target_image, target_image_mask, source_image, source_image_mask, steps, alpha,
                         quality_factor,regularization_weight,plot_interval,debug_root, verbose)

        self.moving_avg_gradient = 0

    def _on_before_attack(self):

        super(NoiseprintMimickingAttack, self)._on_before_attack()

        self.detector.prediction_pipeline(self.source_image.to_float(),
                                          os.path.join(self.debug_folder, "source image"),
                                          omask=self.source_image_mask, debug=True)

    def _compute_target_representation(self, target_representation_source_image: Picture,
                                       target_representation_source_image_mask: Picture):
        """
            This type of attack tries to "paste" noiseprint generated fon an authentic image on top of a
            forged one. The target representation is simply the noiseprint of the authentic image
        """
        image = prepare_image_noiseprint(target_representation_source_image)

        # generate an image wise noiseprint representation on the entire image
        original_noiseprint = Picture(self._engine.predict(image))
        return original_noiseprint

    def attack(self, image_to_attack: Picture, *args, **kwargs):
        """
        Perform step of the attack executing the following steps:

            1) Divide the entire image into patches
            2) Compute the gradient of each patch with respect to the patch-tirget representation
            3) Recombine all the patch-gradients to obtain a image wide gradient
            4) Apply the image-gradient to the image
            5) Convert then the image to the range of values of integers [0,255] and convert it back to the range
               [0,1]
        :return:
        """

        # compute the attacked image using the original image and the compulative noise to reduce
        # rounding artifacts caused by translating the nosie from one to 3 channels and vie versa multiple times
        attacked_image = self.attacked_image_monochannel

        # use nesterov momentum by adding the moving average of the gradient preventivelly
        attacked_image = Picture(attacked_image - self.moving_avg_gradient)

        # compute the gradient
        image_gradient, loss = self._get_gradient_of_image(attacked_image.to_float(), self.target_representation,
                                                           Picture(self.noise))

        # save loss value to plot it
        self.loss_steps.append(loss)

        # normalize the gradient
        image_gradient = normalize_gradient(image_gradient, 0)

        # compute the decaying alpha
        alpha = self.alpha / (1 + 0.1 * self.step_counter)

        # scale the gradient
        image_gradient = alpha * image_gradient

        # update the moving average
        self.moving_avg_gradient = self.moving_avg_gradient * 0.9 + 0.1 * image_gradient

        # add this iteration contribution to the cumulative noise
        self.noise += self.moving_avg_gradient / (1 - 0.75 ** (1 + self.step_counter))

        return self.attacked_image

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
        parser.add_argument('--noiseprint_source', required=True,
                            help='Name of the image to use as source for the noiseprint')
        args = parser.parse_known_args()[0]

        image_path = args.noiseprint_source

        try:
            image, mask = get_image_and_mask(dataset_root, image_path)
        except ImageNotFoundError:
            # the image is not present in the dataset, look if a direct reference has been given
            image_path = Path(image_path)

            if image_path.exists():
                image = Picture(str(image_path))
                mask = np.where(np.all(image == (255,255,255), axis=-1), 0, 1)
            else:
                raise

        kwarg["source_image"] = image
        kwarg["source_image_mask"] = mask
        return kwarg
