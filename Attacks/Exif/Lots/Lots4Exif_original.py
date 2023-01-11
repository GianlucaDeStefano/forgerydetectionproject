import os
from statistics import mean

import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

from Attacks.Exif.Lots.BaseLots4Exif import BaseLots4Exif
from Detectors.Exif.utility import prepare_image
from Utilities.Image.Picture import Picture
from Utilities.Image.functions import visuallize_matrix_values
from tqdm import tqdm


class Lots4ExifOriginal(BaseLots4Exif):


    def _get_gradient_of_image(self, image: Picture, target: list, old_perturbation: Picture = None):
        """
        Function to compute the gradient of the exif model on the entire image
        :param image: image for which we want to compute the gradient
        :param target: target representation we are trying to approximate, in this case it is a list of feature vectors
        :param old_perturbation: old perturbation that has been already applied to the image
        :return: numpy array containing the gradient, loss
        """

        # create object to store the combined gradient of all the patches
        gradient_map = np.zeros(image.shape)

        # create object to store how many patches have contributed to the gradient of a signle pixel
        count_map = np.zeros(image.shape)

        # divide the image into patches of (128,128) pixels
        patches = Picture(image).divide_in_patches((128, 128), force_shape=False, stride=self.stride)

        # verify that the number of patches and corresponding target feature vectors is the same
        assert (len(patches) == len(target))

        # object to store the cumulative loss between all the patches of all the batches
        loss = []

        # iterate through the batches to process
        for batch_idx in tqdm(range(0, (len(patches) + self.batch_size - 1) // self.batch_size, 1)):

            # compute the index of the first element in the batch
            starting_idx = self.batch_size * batch_idx

            # get a list of all the elements in the batch
            batch_patches = patches[starting_idx:min(starting_idx + self.batch_size, len(patches))]

            # prepare all the patches to be feeded into the model
            batch_patches_ready = [prepare_image(patch) for patch in batch_patches]

            # get corresponding list of target patches
            target_batch_patches = target[starting_idx:min(starting_idx + self.batch_size, len(patches))]

            # prepare patches and target vectors to be fed into the model
            x_tensor = np.array(batch_patches_ready, dtype=np.float32)
            y_tensor = np.array(target_batch_patches, dtype=np.float32)

            batch_gradients, batch_loss = self._sess.run([self.gradient_op, self.loss_op],
                                                         feed_dict={self.x: x_tensor, self.y: y_tensor})

            # add the batch loss to the cumulative loss
            loss += batch_loss.tolist()

            # construct an image wide gradient map by combining the gradients computed on sing epatches
            for i, patch_gradient in enumerate(list(batch_gradients[0])):
                gradient_map = batch_patches[i].add_to_image(gradient_map, patch_gradient)
                count_map = batch_patches[i].add_to_image(count_map, np.ones(patch_gradient.shape))

        gradient_map[np.isnan(gradient_map)] = 0

        # average the patches contribution foreach pixel in the gradient
        gradient_map = gradient_map / count_map

        return gradient_map, mean(loss)

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
        authentic_target_representation = np.zeros(4096)

        assert (target_representation_source_image.shape[0] == target_representation_source_image_mask.shape[0])
        assert (target_representation_source_image.shape[1] == target_representation_source_image_mask.shape[1])

        authentic_patches = Picture(target_representation_source_image).get_authentic_patches(
            target_representation_source_image_mask, (128, 128), force_shape=False,
            stride=self.stride)

        all_patches = Picture(target_representation_source_image).divide_in_patches((128, 128), force_shape=False, stride=self.stride)

        with self._sess.as_default():

            for batch_idx in tqdm(range(0, (len(authentic_patches) + self.batch_size - 1) // self.batch_size, 1),
                                  disable=self.clean_execution):
                starting_idx = self.batch_size * batch_idx

                batch_patches = authentic_patches[
                                starting_idx:min(starting_idx + self.batch_size, len(authentic_patches))]

                for i, patch in enumerate(batch_patches):
                    batch_patches[i] = prepare_image(patch)

                patch = np.array(batch_patches)
                tensor_patch = tf.convert_to_tensor(patch, dtype=tf.float32)

                features = self._engine.extract_features_resnet50(tensor_patch, "test", reuse=True).eval()

        mean_features = np.array(features).mean(axis=0)

        return [mean_features] * len(all_patches)


    def loss_function(self, y_pred, y_true):
        pass
