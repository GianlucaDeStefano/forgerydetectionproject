import os

import numpy as np

from Datasets.Dataset import Dataset, ImageNotFoundException, mask_2_binary
from Detectors.Noiseprint.Noiseprint.utility.utilityRead import imread2f


class RitDataset(Dataset):

    def __init__(self, root=os.path.dirname(__file__) + "/Data/"):
        super(RitDataset, self).__init__(root, True,"Realistic image tampering")

        self.camera_folders = ["Canon_60D", "Nikon_D90", "Nikon_D7000", "Sony_A57"]

    def get_authentic_images(self, target_shape=None):

        authentic_images = []

        for folder in self.camera_folders:

            folder_path = os.path.join(self.root, folder, "pristine")

            for filename in os.listdir(folder_path):

                if filename.lower().endswith(tuple(self.supported_formats)):
                    authentic_images.append(os.path.join(folder_path, filename))

        return authentic_images

    def get_forged_images(self, target_shape=None):
        forged_images = []

        for folder in self.camera_folders:

            folder_path = os.path.join(self.root, folder, "tampered-realistic")

            for filename in os.listdir(folder_path):

                if filename.lower().endswith(tuple(self.supported_formats)):
                    forged_images.append(os.path.join(folder_path, filename))

        return forged_images

    def get_mask_of_image(self, image_path:str):
        path = image_path.replace("tampered-realistic", "ground-truth")
        mask, mode = imread2f()
        return mask_2_binary(mask,0.5),path

    def get_image(self, image_name):

        for folder in self.camera_folders:

            folder_path = os.path.join(self.root, folder, "pristine")

            for filename in os.listdir(folder_path):

                if filename == image_name:

                    return os.path.join(folder_path,image_name)

        raise ImageNotFoundException(image_name)

