import os

from PIL.Image import Image

from Datasets.Dataset import Dataset


class Casia2Dataset(Dataset):

    def __init__(self,root = os.path.dirname(__file__) + "/Data/"):
        super(Casia2Dataset, self).__init__(root, True, supported_formats_mask=None)

    def get_authentic_images(self, target_shape=None):
        """
            Return a list of paths to all the authentic images
            :param target_shape: The shape the desired images must have
            :param root: root folder of the dataset
            :return: list of Paths
            """
        paths = []

        for filename in os.listdir(os.path.join(self.root, "Au")):

            if not filename.endswith("jpg"):
                continue

            if target_shape:

                im = Image.open(os.path.join(self.root, "Au", filename))

                if im.size != target_shape:
                    continue

            paths.append(os.path.join(self.root, "Au", filename))

        return paths

    def get_forged_images(self, target_shape=None):
        """
            Return a list of paths to all the authentic images
            :param target_shape: The shape the desired images must have
            :param root: root folder of the dataset
            :return: list of Paths
            """
        paths = []

        for filename in os.listdir(os.path.join(self.root, "Tp")):

            if not filename.endswith("jpg"):
                continue

            if target_shape:

                im = Image.open(os.path.join(self.root, "Tp", filename))

                if im.size != target_shape:
                    continue

            paths.append(os.path.join(self.root, "Tp", filename))
        return paths

    def get_mask_of_image(self, image_path):
        """
        Given the path of an image belonging to the Casia 2 dataset return the path to the mas of said image
        :param image_path: path to the image in question
        :param root:  root folder of the dataset
        :return: path to the mask
        """

        filename = os.path.basename(image_path)
        filename = os.path.splitext(filename)[0] + "_gt.png"
        return os.path.join(self.root, "Gt", filename)
