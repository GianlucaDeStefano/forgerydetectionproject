import argparse
import logging
import os
from abc import abstractmethod, ABC
from datetime import datetime
from pathlib import Path

import numpy as np
from cv2 import PSNR

from Datasets import get_image_and_mask
from Ulitities.Image.Picture import Picture
from Ulitities.Visualizers.ExifVisualizer import ExifVisualizer
from Ulitities.Visualizers.NoiseprintVisualizer import NoiseprintVisualizer
from Ulitities.io.folders import create_debug_folder


class BaseAttack(ABC):
    attack_name = "Base Attack"

    def __init__(self, target_image: Picture, target_image_mask: Picture, detector_name: str,
                 debug_root: str = "./Data/Debug/",
                 test: bool = True):
        """
        :param target_image: original image on which we should perform the attack
        :param target_image_mask: original mask of the image on which we should perform the attack
        :param detector: name of the detector to be used to visualize the results
        :param debug_root: root folder insede which to create a folder to store the data produced by the pipeline
        :param test: is this a test mode? In test mode visualizations and superfluous steps will be skipped in favour of a
            faster execution to test the code
        """

        # check if the target image is in the desired format (integer [0,255])
        assert (isinstance(target_image[0], (int, np.uint)), np.amax(target_image) < 255, np.amax(target_image) > -1)

        # check that the mask is in the desired format (integer [0,1]) and has the same shape of the input image
        # along the x and y axis
        assert (isinstance(target_image[0], (int, np.uint)), np.amax(target_image) < 255, np.amax(target_image) > -1,
                target_image.shape[0] ==
                target_image_mask.shape[0], target_image.shape[1] == target_image_mask.shape[1])

        # save the input image and its mask
        self.target_image = target_image
        self.target_image_mask = target_image_mask

        # load the desired detector into memory
        self.detector = None
        if detector_name.lower() == "noiseprint":
            self.detector = NoiseprintVisualizer()
        elif detector_name.lower() == "exif":
            self.detector = ExifVisualizer()
        else:
            raise Exception("Unknown detector: {}".format(detector_name))

        # save the verbosity level (0 -> short logs, 1-> full lofs)
        self.test = test

        # create debug folder
        self.debug_folder = create_debug_folder(debug_root)

        # Remove all handlers associated with the root logger object.
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        logging.basicConfig(format='%(message)s', filename=os.path.join(self.debug_folder, "logs.txt"),
                            level=logging.DEBUG)

        for name in logging.root.manager.loggerDict:
            logging.getLogger(name).disabled = True

        self.start_time = datetime.now()

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
        self.write_to_logs("Test mode: {}\n".format(str(self.test)))

        self.write_to_logs("Attack name: {}".format(self.name))
        self.write_to_logs("Target image: {}".format(self.target_image.path))

        if not self.test:
            self.detector.prediction_pipeline(self.target_image, os.path.join(self.debug_folder, "initial result"),omask=self.target_image_mask)

        self.start_time = datetime.now()


    def _on_after_attack(self, attacked_image: Picture):
        """
        Instructions executed after performing the attack
        :return:
        """

        psnr = PSNR(self.target_image,np.array(attacked_image,np.int))

        path = os.path.join(self.debug_folder,"attacked_image.png")
        attacked_image.save(path)

        attacked_image = Picture(path=path)

        self.detector.prediction_pipeline(attacked_image, os.path.join(self.debug_folder, "final result"),original_picture=self.target_image,omask=self.target_image_mask,note="final result PSNR:{:.2f}".format(psnr))

        end_time = datetime.now()
        timedelta = end_time - self.start_time

        self.write_to_logs("Attack pipeline terminated in {}".format(timedelta))

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

    def write_to_logs(self, message, force_print=True):
        """
        Add a new line to this attack's log file IF it exists
        :param force_print:
        :param message: message to write to the file
        :return:
        """

        if force_print or self.test:
            print(message)

        if not self.debug_folder:
            return

        logging.info(message)

    @staticmethod
    def read_arguments(dataset_root) -> dict:
        """
        Read arguments from the command line or ask for them if they are not present, validate them raising
        an exception if they are invalid, it is called by the launcher script
        :param args: args dictionary containing the arguments passed while launching the program
        :return: kwargs to pass to the attack
        """
        parser = argparse.ArgumentParser()
        parser.add_argument('--image', required=True, help='Name of the input image, or its path')
        parser.add_argument('--mask', default=None, help='Path to the binary mask of the image')
        parser.add_argument('--test',default=False,action='store_true', help='Should the algorithm be executed in test mode?')
        args = parser.parse_known_args()[0]

        image_path = args.image
        if not image_path:
            image_path = str(input("Input a reference of the image to attack (path or name)"))

        mask_path = args.mask
        if Path(image_path).exists() and not mask_path:
            mask_path = str(input("Input the path to the mask of the image"))

        image, mask = get_image_and_mask(dataset_root, image_path, mask_path)

        kwarg = dict()
        kwarg["target_image"] = image
        kwarg["target_image_mask"] = mask
        kwarg["test"] = args.test
        return kwarg

    @classmethod
    def name(cls):
        return cls.attack_name
