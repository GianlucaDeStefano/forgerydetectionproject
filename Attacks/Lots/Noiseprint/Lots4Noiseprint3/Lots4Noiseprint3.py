import argparse
import os
import tensorflow as tf
import numpy as np
from scipy.ndimage import binary_dilation
from tqdm import tqdm

from Attacks import LotsNoiseprint2
from Attacks.Lots.Noiseprint.Lots4NoiseprintBase import Lots4NoiseprintBase, normalize_gradient
from Attacks.utilities.visualization import visuallize_matrix_values
from Datasets import get_image_and_mask
from Detectors.Noiseprint.noiseprintEngine import NoiseprintEngine, normalize_noiseprint
from Ulitities.Image.Patch import Patch
from Ulitities.Image.Picture import Picture


class LotsNoiseprint3(LotsNoiseprint2):
    name = "LOTS4Noiseprint3"

    def __init__(self, objective_image: Picture, objective_mask: Picture, target_representation_image: Picture = None,
                 target_representation_mask: Picture = None, qf: int = None,
                 patch_size: tuple = (8, 8), padding_size=(0, 0, 0, 0),
                 steps=50, debug_root="./Data/Debug/", alpha=1, plot_interval=1, verbose=True):
        assert (objective_image.shape[0] <= target_representation_image.shape[0])
        assert (objective_image.shape[1] <= target_representation_image.shape[1])

        self.moving_avg_gradient = np.zeros(objective_image.one_channel().shape)

        super().__init__(objective_image, objective_mask, target_representation_image,
                         target_representation_mask, qf, patch_size, padding_size, steps,
                         debug_root, alpha, plot_interval, verbose)

    def _on_before_attack(self):
        self.visualizer.prediction_pipeline(self.original_objective_image.to_float(),
                                            os.path.join(self.debug_folder, "forged image"),
                                            omask=self.objective_image_mask, debug=True)
        self.visualizer.prediction_pipeline(self.target_representation_image.to_float(),
                                            os.path.join(self.debug_folder, "authentic image"),
                                            omask=self.target_representation_mask, debug=True)
        super(LotsNoiseprint3, self)._on_before_attack()


    def _generate_target_representation(self, image: Picture, mask: Picture):
        """
            This type of attack tries to "paste" noiseprint generated fon an authentic image on top of a
            forged one. The target representation is simply the noiseprint of the authentic image
        """

        # generate an image wise noiseprint representation on the entire image
        original_noiseprint = Picture(self._engine.predict(image))
        return original_noiseprint

    @staticmethod
    def read_arguments(dataset_root) -> dict:
        """
        Read arguments from the command line or ask for them if they are not present, validate them raising
        an exception if they are invalid, it is called by the launcher script
        :param args: args dictionary containing the arguments passed while launching the program
        :return: kwargs to pass to the attack
        """
        kwarg = LotsNoiseprint2.read_arguments(dataset_root)

        parser = argparse.ArgumentParser()
        parser.add_argument("-n", '--noiseprint_source', required=True,
                            help='Name of the image to use as source for the noiseprint')
        args = parser.parse_known_args()[0]

        image_path = args.noiseprint_source

        image, mask = get_image_and_mask(dataset_root, image_path)

        kwarg["target_representation_image"] = image
        kwarg["target_representation_mask"] = mask
        return kwarg

    def _attack(self):
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
        image_gradient, loss = self._get_gradient_of_image(attacked_image.to_float(), self.target_representation)

        # save loss value to plot it
        self.loss_steps.append(loss)

        # normalize the gradient
        image_gradient = normalize_gradient(image_gradient, 0)

        # compute the decaying alpha
        alpha = self.alpha / (1 + 0.1 * self.attack_iteration)

        # scale the gradient
        image_gradient = alpha * image_gradient

        # update the moving average
        self.moving_avg_gradient = self.moving_avg_gradient * 0.9 + 0.1 * image_gradient

        # add this iteration contribution to the cumulative noise
        self.noise += self.moving_avg_gradient / (1-0.5**(1+self.attack_iteration))
