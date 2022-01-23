import argparse
import logging
import os
from abc import abstractmethod, ABC
from datetime import datetime
from pathlib import Path

import numpy as np
from cv2 import PSNR

from Datasets import get_image_and_mask
from Detectors.DetectorEngine import DeterctorEngine
from Utilities.Image.Picture import Picture
from Utilities.Logger.Logger import Logger

from Utilities.io.folders import create_debug_folder


class BaseAttack(ABC, Logger):
    attack_name = "Base Attack"

    def __init__(self, detector: DeterctorEngine, debug_root: str = "./Data/Debug/", verbosity: int = 2):
        """
        :param detector: name of the detector to be used to visualize the results
        :param debug_root: root folder insede which to create a folder to store the data produced by the pipeline
        :param verbosity: modality to use to run the attack:
                0 -> the attack will not output any log in the console safe for error crashes, no visualizer is used before,during or after the attack
                1 -> the attack output logs in the console, visualizers will be used only before and after the attack
                2 -> the attack outputs logs in the console, visualizers will be used before,during and after the attack
        """
        # save the input image and its mask
        self.target_image = None
        self.target_image_mask = None

        # load the desired detector into memory
        self.detector = detector

        # save the verbosity level (0 -> no logs,1 -> quick logs, 2-> full logs)
        self.verbosity = verbosity

        # save debug root folder
        self.debug_root = debug_root

        # create debug folder
        self.debug_folder = None

        self.start_time = datetime.now()

    def setup(self, target_image: Picture, target_image_mask: Picture, source_image: Picture = None,
              source_image_mask: Picture = None,target_forgery_mask : Picture = None):
        """
        Function to load the iteration-dependent variable into the pipeline :param target_image: original image on
        which we should perform the attack :param target_image_mask: original mask of the image on which we should
        perform the attack
        :param source_image: image from which we will compute the target representation  (to be
        ignored for classes not extending BaseWhiteBoxAttack)
        :param source_image_mask: mask of the image from which we will compute the target representation (to be
        ignored for classes not extending BaseWhiteBoxAttack)
        :return:
        """
        # check if the target image is in the desired format (integer [0,255])
        assert (isinstance(target_image[0], (int, np.uint)), np.amax(target_image) < 255, np.amin(target_image) > -1)

        # check that the mask is in the desired format (integer [0,1]) and has the same shape of the input image
        # along the x and y axis
        assert (isinstance(target_image[0], (int, np.uint)), np.amax(target_image) < 255, np.amin(target_image) > -1,
                target_image.shape[0] ==
                target_image_mask.shape[0], target_image.shape[1] == target_image_mask.shape[1])

        # save the input image and its mask
        self.target_image = target_image
        self.target_image_mask = target_image_mask

        self.debug_folder = create_debug_folder(self.debug_root)

    @property
    def is_ready(self):
        """
        :return: returns True if the attack is ready to be executed otherwise False
        """
        return self.target_image is not None and self.target_image_mask is not None

    def execute(self) -> Picture:
        """
        Start the attack pipeline using the data passed in the initialization
        :return: attacked image
        """
        # execute pre-attack operations
        pristine_image = self.target_image
        self._on_before_attack()

        # execute the attack
        attacked_image = self.attack(pristine_image)

        # execute post-attack operations
        self._on_after_attack(attacked_image)

        return attacked_image

    def _on_before_attack(self):
        """
        Instructions executed before performing the attack, writing logs
        :return:
        """

        if not self.is_ready:
            raise Exception("The attack is not ready to be executed")

        self.logger_module.info("Verbosity: {}\n".format(str(self.verbosity)))

        self.logger_module.info("Attack name: {}".format(self.name))
        self.logger_module.info("Target image: {}".format(self.target_image.path))

        if not self.test:
            self.detector.prediction_pipeline(self.target_image, os.path.join(self.debug_folder, "initial result"),
                                              omask=self.target_image_mask)

        self.start_time = datetime.now()

    def _on_after_attack(self, attacked_image: Picture):
        """
        Instructions executed after performing the attack
        :return:
        """

        psnr = PSNR(self.target_image, np.array(attacked_image, np.int))

        path = os.path.join(self.debug_folder, "attacked_image.png")
        attacked_image.save(path)

        attacked_image = Picture(path=path)

        if not self.test:
            heatmap,mask = self.detector.prediction_pipeline(attacked_image, os.path.join(self.debug_folder, "final result"),
                                              original_picture=self.target_image, omask=self.target_image_mask,
                                              note="final result PSNR:{:.2f}".format(psnr))

            self.detector.save_heatmap(heatmap,os.path.join(self.debug_folder, "final heatmap.png"))

        end_time = datetime.now()
        timedelta = end_time - self.start_time

        self.logger_module.info("Attack pipeline terminated in {}".format(timedelta))

    @abstractmethod
    def attack(self, image_to_attack: Picture, *args, **kwargs):
        """
        Apply attack to the passed image
        :param image_to_attack: the image on which we are going to apply the attack
        :param args:
        :param kwargs:
        :return: the attacked image
        """
        raise NotImplementedError

    @staticmethod
    def read_arguments(dataset_root) -> tuple:
        """
        Read arguments from the command line or ask for them if they are not present, validate them raising
        an exception if they are invalid, it is called by the launcher script
        :param args: args dictionary containing the arguments passed while launching the program
        :return: kwargs to pass to the attack
        """
        parser = argparse.ArgumentParser()
        parser.add_argument('--image', required=True, help='Name of the input image, or its path')
        parser.add_argument('--mask', default=None, help='Path to the binary mask of the image')
        parser.add_argument('--test', default=False, action='store_true',
                            help='Should the algorithm be executed in test mode?')
        args = parser.parse_known_args()[0]

        image_path = args.image
        if not image_path:
            image_path = str(input("Input a reference of the image to attack (path or name)"))

        mask_path = args.mask
        if Path(image_path).exists() and not mask_path:
            mask_path = str(input("Input the path to the mask of the image"))

        image, mask = get_image_and_mask(dataset_root, image_path, mask_path)

        verbosity = 2
        if args.test:
            verbosity = 1

        attack_parameters = dict()
        attack_parameters["verbosity"] = verbosity

        setup_parameters = dict()
        setup_parameters["target_image"] = image
        setup_parameters["target_image_mask"] = mask
        setup_parameters["source_image"] = None
        setup_parameters["source_image_mask"] = None
        setup_parameters["target_forgery_mask"] = None

        return attack_parameters,setup_parameters

    @classmethod
    def name(cls):
        return cls.attack_name

    @property
    def test(self):
        return self.verbosity <= 1

    @property
    def clean_execution(self):
        return self.verbosity == 0
