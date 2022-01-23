from abc import ABC, abstractmethod

import numpy as np


class NoMaskAvailableException(Exception):
    def __init__(self):
        super().__init__("This Dataset has no masks")


class ImageNotFoundException(Exception):
    def __init__(self, image_name):
        super().__init__("No image found with name: {}".format(image_name))


def mask_2_binary(mask: np.array, threshold: float = 0.5, flip=False):
    """
    Load the groundtruth and be sure the mask is only composd by 0 or ones
    :param mask: an np array containing the mask data
    :param threshold: thereshold to define a positive or negative pixel
    :param flip: flipl positive pixels to negative and negative to positive
    :return:numpy array containing the mask
    """

    # convert the mask to a binary mask
    if len(mask.shape) == 2:
        mask = np.where(mask < threshold, 0, 1)
    else:
        mask = np.where(np.all(mask == (0, 0, 0), axis=-1), 0, 1)

    if flip:
        mask = 1 - mask

    return np.squeeze(mask)


class Dataset(ABC):

    def __init__(self, root, has_masks, name: str, supported_formats=None):
        """
        :param root: root folder of the dataset
        :param has_masks: does the dataset provide masks
        :param name: name of the dataset
        :param supported_formats: list of formats to accept
        """
        if supported_formats is None:
            supported_formats = ["jpg", "jpeg", "png", "tif", "bmp"]
        self.supported_formats = supported_formats
        self.root = root

        print(name)
        if not name or name == "":
            raise Exception("A database needs a name")

        self.name = name
        self.has_masks = has_masks

    @abstractmethod
    def get_authentic_images(self, target_shape=None):
        """
        Return a list of paths corresponding to all the authentic images in the dataset
        :param target_shape: require the returned images to have this target shape
        :return: list of paths
        """
        raise NotImplemented

    @abstractmethod
    def get_forged_images(self, target_shape=None):
        """
        Return a list of paths corresponding to all the authentic forged in the dataset
        :param target_shape: require the returned images to have this target shape
        :return: list of paths
        """
        raise NotImplemented

    @abstractmethod
    def get_image(self, image_name):
        """
        Return the path on an image, given its name
        :param image_name: name og the image
        :return: path to the image
        """
        raise NotImplemented

    def get_mask_of_image(self, image_path):
        """
        Given the path of an image, return np array containing its binary mask
        :param image_path: the path of the image of which we want the mask of
        :return:
        """
        raise NoMaskAvailableException