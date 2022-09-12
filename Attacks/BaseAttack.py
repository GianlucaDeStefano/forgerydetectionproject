import argparse
import logging
import os
from abc import abstractmethod, ABC
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from cv2 import PSNR

from Datasets import get_image_and_mask
from Utilities.Image.Picture import Picture
from Utilities.Logger.Logger import Logger
from Utilities.Visualizers.BaseVisualizer import BaseVisualizer

from Utilities.io.folders import create_debug_folder


class BaseAttack(ABC, Logger):
    attack_name = "Base Attack"

    def __init__(self, visualizer: BaseVisualizer, debug_root: str = "./Data/Debug/", verbosity: int = 2):
        """
        :param visualizer: instance of the visualizer class wrapping the functionalities of the targeted detector
        :param debug_root: root folder insede which to create a folder to store the data produced by the pipeline
        :param verbosity: modality to use to run the attack:
                0 -> the attack will not output any log in the console safe for error crashes, no visualizer is used before,during or after the attack
                1 -> the attack output logs in the console, visualizers will be used only before and after the attack
                2 -> the attack outputs logs in the console, visualizers will be used before,during and after the attack
        """
        # save the input image and its mask
        self.target_image_path = None
        self.target_image = None
        self.target_image_mask = None

        # load the desired detector into memory
        self.visualizer = visualizer

        # save the verbosity level (0 -> no logs,1 -> quick logs, 2-> full logs)
        self.verbosity = verbosity

        self._debug_root = debug_root

        # create debug folder
        self.debug_folder = None

        self.start_time = datetime.now()

    def setup(self, target_image_path: str, target_image_mask: Picture):
        """
        Setup the pipeline for execution
        @param target_image_path: path fo the sample to process
        @param target_image_mask: np.array containing a binary mask where 0 -> pristine 1-> forged pixel
        @return:
        """

        print("\nSETUP \n")
        self.debug_folder = create_debug_folder(self._debug_root)

        # load the sample in the visualizer
        self.visualizer.initialize(target_image_path, None, True, True)

        # prepare the instance of the input image and its mask
        self.target_image = Picture(self.visualizer.metadata["sample"])
        self.target_image_path = self.visualizer.metadata["sample_path"]
        self.target_image_mask = target_image_mask


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
        self.logger_module.info("Target image: {}".format(self.target_image_path))

        if not self.test:
            self.visualizer.save_prediction_pipeline(os.path.join(self.debug_folder, "initial result"))

        self.start_time = datetime.now()

    def _on_after_attack(self, attacked_image: Picture):
        """
        Instructions executed after performing the attack
        :return:
        """
        print("FINAL metadata")

        path = os.path.join(self.debug_folder, "attacked_image.png")
        attacked_image.save(path)

        pristine_image = np.asarray(cv2.imread(self.target_image_path), dtype=np.uint8)
        attacked_image_n = np.asarray(cv2.imread(path), dtype=np.uint8)

        psnr = PSNR(pristine_image, attacked_image_n)

        self.logger_module.info(f"FINAL PSNR:{psnr:.02f}")

        if not self.test:
            self.visualizer.initialize(sample_path=path)
            self.visualizer.save_prediction_pipeline(os.path.join(self.debug_folder, "final result"))

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
        setup_parameters["target_image_path"] = image.path
        setup_parameters["target_image_mask"] = mask
        setup_parameters["source_image_path"] = None
        setup_parameters["source_image_mask"] = None
        setup_parameters["target_forgery_mask"] = None

        return attack_parameters, setup_parameters

    @classmethod
    def name(cls):
        return cls.attack_name

    @property
    def test(self):
        return self.verbosity <= 1

    @property
    def clean_execution(self):
        return self.verbosity == 0
