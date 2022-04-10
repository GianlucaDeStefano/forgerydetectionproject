import os
from pathlib import Path

import numpy as np
from PIL import Image

from Datasets.Dataset import Dataset, ImageNotFoundException


def mask_2_binary(mask: np.array):
    """
    Convert a mask of the Columbia Uncompressed dataset to a Binary mask
    :param mask:
    :return: binary mask
    """

    mask = np.asarray(mask, dtype="float")

    # transform the mask into a binary mask:

    mask = mask[:, :, 0]
    return np.array(np.where(mask >= 200, 0, 1), int)


class ColumbiaUncompressedDataset(Dataset):

    def __init__(self, root):
        super(ColumbiaUncompressedDataset, self).__init__(os.path.join(root, "ColumbiaUncompressed"), False,
                                                          "Columbia uncompressed", ["tif"])

    def get_authentic_images(self, target_shape=None):
        """
        Return a list of paths to all the authentic images
        :param root: root folder of the dataset
        :return: list of Paths
        """

        paths = []

        for image_format in self.supported_formats:
            paths += Path(os.path.join(self.root, "4cam_auth")).glob("*.{}".format(image_format))

        return paths

    def get_forged_images(self, target_shape=None):
        """
        Return a list of paths to all the authentic images
        :param root: root folder of the dataset
        :return: list of Paths
        """

        paths = []

        for image_format in self.supported_formats:
            paths += Path(os.path.join(self.root, "4cam_splc")).glob("*.{}".format(image_format))

        return paths

    def get_mask_of_image(self, image_path: str):
        image_name = os.path.basename(image_path)
        image_name = os.path.splitext(image_name)[0] + "_edgemask.jpg"

        path = None
        mask = False

        if Path(os.path.join(self.root, "4cam_splc", 'edgemask', image_name)).exists():
            path = str(os.path.join(self.root, "4cam_splc", 'edgemask', image_name))
            mask = Image.open(path)
            mask.load()
            mask = mask_2_binary(np.array(mask))
            return mask, path

        elif Path(os.path.join(self.root, "4cam_auth", 'edgemask', image_name)).exists():
            # the image has no forgery, return a nupy array of zeroes
            image = Image.open(image_path)
            image.load()
            image = np.array(image)

            mask = np.zeros((image.shape[0], image.shape[1]))
            path = ""

        else:
            raise ImageNotFoundException(image_name)

        return mask, path

    def get_image(self, image_name):

        fixed_name = str(image_name).split(".")[0] + ".tif"

        if Path(os.path.join(self.root, "4cam_splc", fixed_name)).exists():
            return str(os.path.join(self.root, "4cam_splc", fixed_name))
        elif Path(os.path.join(self.root, "4cam_auth", fixed_name)).exists():
            return str(os.path.join(self.root, "4cam_auth", fixed_name))
        else:
            raise ImageNotFoundException(image_name)
