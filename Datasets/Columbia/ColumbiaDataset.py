import os
from pathlib import Path

from Datasets.Dataset import Dataset, ImageNotFoundException


class ColumbiaDataset(Dataset):

    def __init__(self,root = os.path.dirname(__file__) + "/Data/"):
        super(ColumbiaDataset, self).__init__(root, False,"Columbia", supported_formats=["bmp"])

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
                    for image_format in self.supported_formats:
                        paths += Path(os.path.join(root, name)).glob("*.{}".format(image_format))
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

    def get_image(self, image_name):
        for root, dirs, files in os.walk(self.root, topdown=False):
            for dir_name in dirs:
                for file_name in  os.walk(os.path.join(root, dir_name), topdown=False):
                    if file_name == image_name:
                        return os.path.join(root, dir_name,image_name)

        raise ImageNotFoundException(image_name)

