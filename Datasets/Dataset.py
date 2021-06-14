from abc import ABC, abstractmethod


class NoMaskAvailableException(Exception):
    def __init__(self):
        super().__init__("This Dataset has no masks")


class ImageNotFoundException(Exception):
    def __init__(self, image_name):
        super().__init__("No image found with name: {}".format(image_name))


class Dataset(ABC):

    def __init__(self, root, has_masks, supported_formats=None):
        """
        :param root: root folder of the dataset
        :param has_masks: does the dataset provide masks
        :param supported_formats: list of formats to accept
        """
        if supported_formats is None:
            supported_formats = ["jpg", "jpeg", "png", "tif", "bmp"]
        self.supported_formats = supported_formats
        self.root = root
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
