import argparse
import os
from abc import ABC, abstractmethod

import numpy as np
from cv2 import PSNR

from Attacks.BaseIterativeAttack import BaseIterativeAttack
from Ulitities.Image.Picture import Picture


class BaseWhiteBoxAttack(BaseIterativeAttack, ABC):
    """
    This class in used as a base to implement attacks whose aim is to reproduce some kind of "target representation"
    """

    name = "Base Mimicking Attack"

    def __init__(self, target_image: Picture, target_image_mask: Picture, source_image: Picture,
                 source_image_mask: Picture, detector: str, steps: int, alpha: float, regularization_weight=0.05,
                 plot_interval=5, additive_attack=True,
                 debug_root: str = "./Data/Debug/", verbose: bool = True):
        """
        :param target_image: original image on which we should perform the attack
        :param target_image_mask: original mask of the image on which we should perform the attack
        :param source_image: image from which we will compute the target representation
        :param source_image_mask: mask of the imae from which we will compute the target representation
        :param detector: name of the detector to be used to visualize the results
        :param steps: number of attack iterations to perform
        :param alpha: strength of the attack
        :param regularization_weight: [0,1] importance of the regularization factor in the loss function
        :param plot_interval: how often (# steps) should the step-visualizations be generated?
        :param additive_attack: show we feed the result of the iteration i as the input of the iteration 1+1?
        :param debug_root: root folder inside which to create a folder to store the data produced by the pipeline
        :param verbose: verbosity of the logs printed in the console
        """

        super(BaseWhiteBoxAttack, self).__init__(target_image, target_image_mask, detector, steps, plot_interval,
                                                 additive_attack,
                                                 debug_root, verbose)

        # save the source image and its mask
        self.source_image = source_image
        self.source_image_mask = source_image_mask

        # save the regularization weight
        self.regularization_weight = regularization_weight

        # save the strength of the attack
        self.alpha = alpha

        # variable for storing the target representation of the attack
        self.target_representation = None

        # create list for tracking the loss during iterations
        self.loss_steps = [9999]

        # create list for tracking the PSNR during iterations
        self.psnr_steps = [999]

    def _on_before_attack(self):
        """
        Write the input parameters into the logs, generate target representation
        :return: None
        """
        super(BaseWhiteBoxAttack, self)._on_before_attack()

        self.write_to_logs("Source image: {}".format(self.source_image.path))
        self.write_to_logs("Alpha: {}".format(self.alpha))
        self.write_to_logs("Regularization weight:{}".format(self.regularization_weight))

        # compute the target representation
        self.target_representation = self._compute_target_representation(self.source_image, self.source_image_mask)

    def _on_after_attack_step(self, attacked_image: Picture, *args, **kwargs):
        """
        At each step update the graph of the loss and of the PSNR
        :param attacked_image:
        :param args:
        :param kwargs:
        :return:
        """

        # compute the PSNR between the initial image
        psnr = PSNR(self.target_image, np.array(attacked_image, np.int))
        self.psnr_steps.append(psnr)

        super()._on_after_attack_step(attacked_image)

        self.detector.plot_graph(self.loss_steps[1:], "Loss", "Attack iteration",
                                 os.path.join(self.debug_folder, "loss"))
        self.detector.plot_graph(self.psnr_steps[1:], "PSNR", "Attack iteration",
                                 os.path.join(self.debug_folder, "psnr"))

    def step_note(self):
        return "Step:{} Loss:{:.2f} PSNR:{:.2f}".format(self.step_counter + 1, self.loss_steps[-1], self.psnr_steps[-1])

    @abstractmethod
    def _compute_target_representation(self, target_representation_source_image: Picture,
                                       target_representation_source_image_mask: Picture):
        """
        Function to compute the target representation our attack are going to try to mimic
        :param target_representation_source_image: image source of the target representation
        :param target_representation_source_image_mask: mask of the image source of the target representation
        :return:
        """
        raise NotImplementedError

    @staticmethod
    def read_arguments(dataset_root) -> dict:
        """
        Read arguments from the command line or ask for them if they are not present, validate them raising
        an exception if they are invalid, it is called by the launcher script
        :param args: args dictionary containing the arguments passed while launching the program
        :return: kwargs to pass to the attack
        """
        kwarg = BaseIterativeAttack.read_arguments(dataset_root)
        parser = argparse.ArgumentParser()
        parser.add_argument("-a", '--alpha', default=5, type=float, help='Strength of the attack')
        args = parser.parse_known_args()[0]

        kwarg["alpha"] = int(args.alpha)

        return kwarg
