import os
from pathlib import Path

import numpy as np
from PIL import Image

from Datasets.Dataset import Dataset, ImageNotFoundException, mask_2_binary


class DsoDatasetDataset(Dataset):

    def __init__(self, root):
        super(DsoDatasetDataset, self).__init__(os.path.join(root,"DSO"), False, "DSO dataset", ["png"])

    def get_authentic_images(self, target_shape=None):
        """
        Return a list of paths to all the authentic images
        :param root: root folder of the dataset
        :return: list of Paths
        """

        paths = []

        for image_format in self.supported_formats:
            paths += Path(os.path.join(self.root, "images")).glob("*.{}".format(image_format))

        return paths

    def get_forged_images(self, target_shape=None):
        """
        Return a list of paths to all the authentic images
        :param root: root folder of the dataset
        :return: list of Paths
        """

        paths = []

        for image_format in self.supported_formats:
            Path(os.path.join(self.root, "masks")).glob("*.{}".format(image_format))

        return paths

    def get_mask_of_image(self, image_path: str):

        filename = os.path.basename(image_path)

        if "normal" in filename:
            # the image is an authentic one, return an all 0 mask
            image = Image.open(image_path)
            image.load()
            image = np.array(image)
            return np.zeros(image.shape)

        # return the mask
        path = os.path.join(self.root, "masks", filename)
        mask = Image.open(path)
        mask.load()
        mask = np.squeeze(np.array(mask))[:, :, 0]
        return mask_2_binary(mask, flip=True), path

    def get_image(self, image_name):

        path = Path(os.path.join(self.root, "images", image_name))
        if path.exists():
            return str(path)
        else:
            raise ImageNotFoundException(image_name)
