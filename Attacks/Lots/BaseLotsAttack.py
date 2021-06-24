import os.path
from abc import ABC, abstractmethod
import numpy as np
from PIL import Image
from datetime import datetime
from Attacks.BaseAttack import BaseAttack
from Attacks.utilities.image import one_2_three_channel
from Ulitities.Image import Picture


class InvalidPatchSize(Exception):

    def __init__(self, reason):
        super().__init__("The passed patch size is invalid for the following reason: {}".format(reason))


class MissingTargetRepresentation(Exception):
    def __init__(self, image_name):
        super().__init__("No image found with name: {}".format(image_name))


def check_patch_size(patch_size):
    """
    Function used to check if the givedn patch sieze is valid or not
    :param patch_size:
    :return:
    """
    axis = 0
    for element in list(patch_size):
        if element < 1:
            raise InvalidPatchSize("The axis {} has a 0 or negative size: {}".format(axis, element))
        axis += 1

    return True


class BaseLotsAttack(BaseAttack, ABC):

    def __init__(self, target_image: Picture, mask: Picture, name: str, image_path, mask_path, patch_size: tuple,
                 steps=50, debug_root= "./Data/Debug/",alpha=5,plot_interval=3):
        """
        Base class to implement various attacks
        :param target_image: image to attack
        :param mask: binary mask of the image to attack, 0 = authentic, 1 = forged
        :param name: name to identify the attack
        :param patch_size: size of the patch ot use to generate the target representation
        :param steps: total number of steps of the attack
        :param debug_root: root dolder in which save debug data generated by the attack
        """

        assert (check_patch_size(patch_size))
        self.patch_size = patch_size

        # Define object to contain the patch wise target representation
        self.target_representation = None

        self.alpha = alpha
        super().__init__(target_image, mask, name, image_path, mask_path, steps, debug_root,plot_interval)

    def _on_before_attack(self):
        super()._on_before_attack()

        # Before beginning the attack, generate the target representation
        if self.target_representation is None:
            self._generate_target_representation()

        self.write_to_logs("Patch size:{}".format(str(self.patch_size)))
        self.write_to_logs("Image shape:{}".format(str(self.original_image.shape)))
        self.write_to_logs("Target representation shape:{}".format(str(self.target_representation.shape)))
        self.write_to_logs("Quality factor:{}".format(str(self.qf)))
        self.write_to_logs("Alpha:{}".format(str(self.alpha)))

    @abstractmethod
    def _generate_target_representation(self):
        raise NotImplemented

