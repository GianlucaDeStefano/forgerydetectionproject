import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from Attacks.Exif.BaseExifAttack import BaseExifAttack
from Datasets import get_image_and_mask, ImageNotFoundError
from Detectors.Exif.utility import prepare_image
from Ulitities.Image.Picture import Picture
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


class ExifMimickingAttack(BaseExifAttack):
    name = "Exif mimicking attack"

    def __init__(self, target_image: Picture, target_image_mask: Picture, source_image: Picture,
                 source_image_mask: Picture, steps: int, alpha: float = 1,
                 regularization_weight=0.05, plot_interval=1, patch_size=(128, 128), batch_size: int = 128,
                 debug_root: str = "./Data/Debug/", test: bool = True):
        """
        :param target_image: original image on which we should perform the attack
        :param target_image_mask: original mask of the image on which we should perform the attack
        :param source_image: image from which we will compute the target representation
        :param source_image_mask: mask of the imae from which we will compute the target representation
        :param steps: number of attack iterations to perform
        :param alpha: strength of the attack
        :param regularization_weight: [0,1] importance of the regularization factor in the loss function
        :param plot_interval: how often (# steps) should the step-visualizations be generated?
        :param patch_size: Width and Height of the patches we are using to compute the Exif parameters
                            we assume that the the patch is always a square eg patch_size[0] == patch_size[1]
        :param batch_size: how many patches shall be processed in parallel
        :param debug_root: root folder inside which to create a folder to store the data produced by the pipeline
        :param test: is this a test mode? In test mode visualizations and superfluous steps will be skipped in favour of a
            faster execution to test the code
        """

        assert (target_image.shape == source_image.shape)

        super().__init__(target_image, target_image_mask, source_image, source_image_mask, steps, alpha,
                         regularization_weight, plot_interval, patch_size, batch_size, debug_root, test)

        stride = (max(source_image.shape[0], source_image.shape[1]) - self.patch_size[0]) // 30

        self.stride = (stride, stride)

    def _compute_target_representation(self, target_representation_source_image: Picture,
                                       target_representation_source_image_mask: Picture):
        """
        Compute the target representation, being this a mimicking attack the target representation is just
        a list of target feature vectors towards which we have to drive the individual patches of our image
        :param target_representation_source_image:  image on which to compute the target feature vectors
        :param target_representation_source_image_mask: mask of the image
        :return: list of 4096-dimensional feature vectors
        """
        target_representation = []

        patches = Picture(target_representation_source_image).divide_in_patches((128, 128), force_shape=False,
                                                                                stride=self.stride)

        with self._sess.as_default():
            for batch_idx in tqdm(range(0, (len(patches) + self.batch_size - 1) // self.batch_size, 1)):
                starting_idx = self.batch_size * batch_idx

                batch_patches = patches[starting_idx:min(starting_idx + self.batch_size, len(patches))]

                for i, patch in enumerate(batch_patches):
                    batch_patches[i] = prepare_image(patch)

                patch = np.array(batch_patches)
                tensor_patch = tf.convert_to_tensor(patch, dtype=tf.float32)

                features = self._engine.extract_features_resnet50(tensor_patch, "test", reuse=True).eval()

                target_representation = target_representation + [np.array(t) for t in features.tolist()]

            return target_representation

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
        loss = 0

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

            batch_gradients,batch_loss = self._sess.run([self.gradient_op,self.loss_op],feed_dict={self.x: x_tensor, self.y: y_tensor})

            # add the batch loss to the cumulative loss
            loss += batch_loss

            # construct an image wide gradient map by combining the gradients computed on sing epatches
            for i, patch_gradient in enumerate(list(batch_gradients[0])):
                gradient_map = batch_patches[i].add_to_image(gradient_map, patch_gradient)
                count_map = batch_patches[i].add_to_image(count_map, np.ones(patch_gradient.shape))

        gradient_map[np.isnan(gradient_map)] = 0

        # average the patches contribution foreach pixel in the gradient
        gradient_map = gradient_map / count_map

        return gradient_map, loss

    @staticmethod
    def read_arguments(dataset_root) -> dict:
        """
        Read arguments from the command line or ask for them if they are not present, validate them raising
        an exception if they are invalid, it is called by the launcher script
        :param args: args dictionary containing the arguments passed while launching the program
        :return: kwargs to pass to the attack
        """
        kwarg = BaseExifAttack.read_arguments(dataset_root)

        parser = argparse.ArgumentParser()
        parser.add_argument('--source_image', required=True,
                            help='Name of the image to use as source')

        args = parser.parse_known_args()[0]

        image_path = args.source_image

        try:
            image, mask = get_image_and_mask(dataset_root, image_path)
        except ImageNotFoundError:
            # the image is not present in the dataset, look if a direct reference has been given
            image_path = Path(image_path)

            if image_path.exists():
                image = Picture(str(image_path))
                mask = np.where(np.all(image == (255, 255, 255), axis=-1), 1, 0)
            else:
                raise

        kwarg["source_image"] = image
        kwarg["source_image_mask"] = mask
        return kwarg
