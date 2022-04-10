import os
from abc import abstractmethod

from tqdm import tqdm

from Datasets import mask_2_binary, Dataset
from Utilities.Image.Picture import Picture
from Utilities.Logger.Logger import Logger


class TransferabilityExperiment(Logger):
    """
    This class is made to test the actual transferability of the applied forgeries between detectors
    """

    def __init__(self, debug_root, data_root_folder, original_dataset: Dataset, visualizer):
        """
        :param debug_root: str
            debug root folder where to save the logs
        :param data_root_folder: str
            folder containing the data to process
        :param original_dataset: Dataset
            original dataset of the samples to test (used to retrieve the original mask)
        :param visualizer: Visualizer
            The visualizer class of the detector we want to use to test the samples
        """

        self.debug_root = debug_root

        self.original_results_dir = os.path.join(self.debug_root, "original")
        os.makedirs(self.original_results_dir)

        self.attacked_results_dir = os.path.join(self.debug_root, "attacked other detector")
        os.makedirs(self.attacked_results_dir)

        self.data_root_folder = data_root_folder
        self.original_dataset = original_dataset
        self.visualizer = visualizer

    @property
    def target_masks_folder(self):
        """
        Returns the path of the folder containing all the target masks
        """
        return os.path.join(self.data_root_folder, "masks")

    @property
    def attacked_images_folder(self):
        """
        Returns the path of the folder containing all the attacked images
        """
        return os.path.join(self.data_root_folder, "output")

    @abstractmethod
    def _compute_scores(self, sample_name, original_image, attacked_image, original_forgery_mask, target_forgery_mask):
        raise NotImplementedError

    def execute(self):
        """
        Execute the experiment
        """

        for sample_name in tqdm(os.listdir(self.attacked_images_folder)):

            if "png" not in sample_name or "tmp" in sample_name or "output" in sample_name:
                continue

            # load sample
            attacked_image = Picture(path=os.path.join(self.attacked_images_folder, sample_name))

            # load target forgery mask
            target_forgery_mask = mask_2_binary(Picture(os.path.join(self.target_masks_folder, sample_name)))

            # load original forgery mask
            original_sample_path = self.original_dataset.get_image(sample_name)
            original_image = Picture(path=original_sample_path)

            original_forgery_mask, _ = self.original_dataset.get_mask_of_image(original_sample_path)

            # obtain predicted heatmap and mask
            self._compute_scores(sample_name, original_image, attacked_image, original_forgery_mask,
                                 target_forgery_mask)
