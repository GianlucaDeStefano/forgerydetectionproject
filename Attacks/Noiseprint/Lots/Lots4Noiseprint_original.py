import os

import numpy as np
from matplotlib import pyplot as plt
from tensorflow.python.ops.gen_math_ops import squared_difference
from tqdm import tqdm

import tensorflow as tf
from Attacks.Noiseprint.Lots.BaseLots4Noiseprint import BaseLots4Noiseprint
from Detectors.Noiseprint.utility.utility import prepare_image_noiseprint, normalize_noiseprint_no_margins
from Utilities.Image.Picture import Picture
from Utilities.Image.functions import visuallize_matrix_values


class Lots4NoiseprintAttackOriginal(BaseLots4Noiseprint):

    name = "LOTS Attack"

    def __init__(self, steps: int, alpha: float, patch_size=(8, 8),
                 padding_size=(0, 0, 0, 0), quality_factor=None, regularization_weight=0.0, plot_interval: int = 5,
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
        :param debug_root: root folder inside which to create a folder to store the data produced by the pipeline
        :param verbosity: is this a test mode? In test mode visualizations and superfluous steps will be skipped in favour of a
            faster execution to test the code
        """

        super().__init__(steps, alpha, 0,quality_factor, regularization_weight, plot_interval, debug_root, verbosity)

        self.patch_size = patch_size
        self.padding_size = padding_size

        self.gradient_normalization_margin = 8


    def _on_before_attack(self):
        """
        Save parameters into the logs
        :return:
        """
        super(Lots4NoiseprintAttackOriginal, self)._on_before_attack()

        self.logger_module.info("Analyzing the image by patches of size:{}".format(self.patch_size))
        self.logger_module.info("Padding patches on each dimension by:{}".format(self.padding_size))

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
        self.logger_module.info("Generating target representation...")

        # foreach authentic patch
        for base_index in tqdm(range(0, len(authentic_patches), self.batch_size)):

            patches = authentic_patches[base_index:min(len(authentic_patches), base_index + self.batch_size)]

            # compute its noiseprint
            noiseprint_patches = np.squeeze(self._engine.model(np.array(patches)[:, :, :, np.newaxis]))

            for i, noiseprint_patch in enumerate(noiseprint_patches):
                # add the noiseprint to the mean target patch object
                target_patch += noiseprint_patch / len(authentic_patches)

                # add the result to the map of patches
                patches_map = patches[i].no_paddings().add_to_image(patches_map)

        self.logger_module.info("Target representation generated")

        t_no_padding = authentic_patches[0].no_paddings(target_patch)

        # save a visualization of the target representation
        normalized_noiseprint = normalize_noiseprint_no_margins(t_no_padding)

        plt.imsave(fname=os.path.join(self.debug_folder, "image-target.png"), arr=normalized_noiseprint,
                   cmap='gray',
                   format='png')

        visuallize_matrix_values(t_no_padding, os.path.join(self.debug_folder, "image-target-raw.png"))

        patches_map = Picture(patches_map)
        patches_map.save(os.path.join(self.debug_folder, "patches-map.png"))

        return target_patch

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

        # analyze the image using batch of patches
        for base_index in tqdm(range(0, len(img_patches), self.batch_size)):

            # retieve this batch's patches form the list
            patches = img_patches[base_index:min(len(img_patches), base_index + self.batch_size)]

            # retrieve the noise patches from the list
            perturbations = noise_patches[base_index:min(len(img_patches), base_index + self.batch_size)]

            # check if we are on a border and therefore we have to "cut"tareget representation
            targets = [target for i in range(len(patches))]

            # if we are on a border, cut away the "overflowing target representation"
            for i, patch in enumerate(patches):
                if targets[i].shape != patch.shape:
                    targets[i] = targets[i][:patch.shape[0], :patch.shape[1]]

            # compute the gradient of the input w.r.t. the target representation
            patches_gradient, batch_loss = self._get_gradients_of_patches(patches, targets, perturbations)

            # add this patch's loss contribution
            cumulative_loss += np.sum(batch_loss)/(len(img_patches)//self.batch_size)

            for i, patch in enumerate(patches):
                # Add the contribution of this patch to the image wide gradient removing the padding
                image_gradient = patch.add_to_image(image_gradient, patches_gradient[i])

        return image_gradient, cumulative_loss

    def loss_function(self, y_pred, y_true):
        """
        Specify a loss function to drive the image we are attacking towards the target representation
        The default loss is the l2-norm
        :param y_pred: last output of the model
        :param y_true: target representation
        :return: loss value
        """
        return tf.norm(y_pred-y_true, ord='euclidean', axis=[1,2])

    def regularizer_function(self, perturbation=None):
        """
        Compute te regularization value to add to the loss function
        :param perturbation:perturbation for which to compute the regularization value
        :return: regularization value
        """

        # if no perturbation is given return 0
        if perturbation is None:
            return 0

        return tf.norm(perturbation, ord='euclidean', axis=[1,2])
