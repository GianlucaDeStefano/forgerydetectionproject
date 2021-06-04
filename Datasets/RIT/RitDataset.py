import os

from Datasets.Dataset import Dataset
from Ulitities.Images import load_mask


class RitDataset(Dataset):

    def __init__(self, root=os.path.dirname(__file__) + "/Data/"):
        super(RitDataset, self).__init__(root, True)

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
        return load_mask(image_path.replace("tampered-realistic", "ground-truth"))
