import argparse
import os
from pathlib import Path
from statistics import mean

import numpy as np
from tqdm import tqdm

from Attacks.Exif.BaseExifAttack import BaseExifAttack
from Attacks.Exif.Mimicking.BaseMimickin4Exif import BaseMimicking4Exif
from Detectors.Exif.utility import prepare_image
from Utilities.Image.Picture import Picture
import tensorflow as tf

from Utilities.Visualizers.ExifVisualizer import ExifVisualizer


class ExifIntelligentAttack(BaseMimicking4Exif):
    name = "Exif mimicking attack"

    def __init__(self, steps: int, alpha: float = 5,regularization_weight=0, plot_interval=1, patch_size=(128, 128),
                 batch_size: int = 64,
                 debug_root: str = "./Data/Debug/", verbosity: int = 2):
        """
        :param steps: number of attack iterations to perform
        :param alpha: strength of the attack
        :param regularization_weight: [0,1] importance of the regularization factor in the loss function
        :param plot_interval: how often (# steps) should the step-visualizations be generated?
        :param patch_size: Width and Height of the patches we are using to compute the Exif parameters
                            we assume that the the patch is always a square eg patch_size[0] == patch_size[1]
        :param batch_size: how many patches shall be processed in parallel
        :param debug_root: root folder inside which to create a folder to store the data produced by the pipeline
        :param verbosity: is this a test mode? In test mode visualizations and superfluous steps will be skipped in favour of a
            faster execution to test the code
        """

        super().__init__(steps, alpha, regularization_weight, plot_interval, patch_size, batch_size,
                         debug_root, verbosity)

        self.target_forgery_mask = None

        self.stride = None

        self.k = 10

    def setup(self, target_image_path: Picture, target_image_mask: Picture, source_image_path: Picture = None,
              source_image_mask: Picture = None, target_forgery_mask: Picture = None):

        self.logger_module.info("Setting up the attack ...")

        super().setup(target_image_path, target_image_mask, source_image_path, source_image_mask, target_forgery_mask)

        stride = (max(self.visualizer.metadata["sample"].shape[:2]) - self.patch_size[0]) // 30

        self.stride = (stride, stride)

        self.logger_module.info("Attack Ready")

    def _compute_target_representation(self, target_representation_source_image: Picture,
                                       target_representation_source_image_mask: Picture,
                                       target_forgery_mask: Picture = None):
        """
        Compute the target representation, being this a mimicking attack the target representation is just
        a list of target feature vectors towards which we have to drive the individual patches of our image
        :param target_representation_source_image:  image on which to compute the target feature vectors
        :param target_representation_source_image_mask: mask of the image
        :return: list of 4096-dimensional feature vectors
        """

        self.logger_module.info("Compute target representation")

        if target_forgery_mask is None:
            target_forgery_mask = self.target_forgery_mask

        authentic_target_representation = np.zeros(4096)

        print(target_representation_source_image.shape,target_representation_source_image_mask.shape)

        assert (target_representation_source_image.shape[0] == target_representation_source_image_mask.shape[0])
        assert (target_representation_source_image.shape[1] == target_representation_source_image_mask.shape[1])

        authentic_patches = Picture(target_representation_source_image).get_authentic_patches(
            target_representation_source_image_mask, (128, 128), force_shape=False,
            stride=self.stride)

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

                for authentic_feature in features:
                    authentic_target_representation += np.array(authentic_feature) / len(authentic_patches)

        forged_target_representation = -authentic_target_representation * self.k

        patches = Picture(target_forgery_mask).divide_in_patches((128, 128), force_shape=False, stride=self.stride)

        target_representations = []

        for patch in patches:
            if patch.max() == 0:
                target_representations += [authentic_target_representation]
            else:
                target_representations += [forged_target_representation]

        return target_representations


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
        for batch_idx in tqdm(range(0, (len(patches) + self.batch_size - 1) // self.batch_size, 1),
                              disable=self.clean_execution):

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

            # construct an image wide gradient map by combining the gradients computed on singe patches
            for i, patch_gradient in enumerate(list(batch_gradients[0])):
                gradient_map = batch_patches[i].add_to_image(gradient_map, patch_gradient)
                count_map = batch_patches[i].add_to_image(count_map, np.ones(patch_gradient.shape))

        gradient_map[np.isnan(gradient_map)] = 0

        # average the patches contribution foreach pixel in the gradient
        gradient_map = gradient_map / count_map

        return gradient_map, mean(loss)

    @staticmethod
    def read_arguments(dataset_root) -> tuple:
        """
        Read arguments from the command line or ask for them if they are not present, validate them raising
        an exception if they are invalid, it is called by the launcher script
        :param args: args dictionary containing the arguments passed while launching the program
        :return: kwargs to pass to the attack
        """
        attack_parameters, setup_parameters = BaseExifAttack.read_arguments(dataset_root)

        parser = argparse.ArgumentParser()
        parser.add_argument('--target_forgery_mask', required=False, default=None,
                            help='Path of the mask highlighting the section of the image that should be identified as '
                                 'forged')
        parser.add_argument('--target_forgery_id', required=False, type=int, default=None,
                            help='Id of the target_forgery type to use to autogenerate the target_forgery map')
        args = parser.parse_known_args()[0]

        target_forgery_mask_path = args.target_forgery_mask

        if target_forgery_mask_path is not None:
            mask_path = Path(target_forgery_mask_path)

            if mask_path.exists():
                mask = np.where(np.all(Picture(str(mask_path)) == (255, 255, 255), axis=-1), 1, 0)
            else:
                raise Exception("Target forgery mask not found")

            setup_parameters["target_forgery_mask"] = Picture(mask)

        else:

                if input("No target_forgery_mask has been specified. Do you want to use one generated randomly? (y/n)") == 'y':
                    setup_parameters["target_forgery_mask"] = None
                else:
                    exit(0)

        return attack_parameters, setup_parameters
