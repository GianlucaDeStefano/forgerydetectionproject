import os
from pathlib import Path

from Datasets.Dataset import Dataset


class ColumbiaDataset(Dataset):

    def __init__(self,root = os.path.dirname(__file__) + "/Data/"):
        super(ColumbiaDataset, self).__init__(root, False, supported_formats_mask=None)

    def get_authentic_images(self, target_shape=None):
        """
        Return a list of paths to all the authentic images
        :param root: root folder of the dataset
        :return: list of Paths
        """
        paths = []

        for root, dirs, files in os.walk(self.root, topdown=False):
            for name in dirs:
                if name.startswith("Au"):
                    paths += Path(os.path.join(root, name)).glob("*.bmp")
        return paths

    def get_forged_images(self, target_shape=None):
        """
        Return a list of paths to all the authentic images
        :param root: root folder of the dataset
        :return: list of Paths
        """
        paths = []

        for root, dirs, files in os.walk(self.root, topdown=False):
            for name in dirs:
                if name.startswith("Sp"):
                    paths += Path(os.path.join(root, name)).glob("*.bmp")

        return paths
