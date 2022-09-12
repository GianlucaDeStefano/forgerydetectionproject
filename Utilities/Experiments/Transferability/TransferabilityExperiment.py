import os.path
import traceback
from abc import abstractmethod
from collections import defaultdict
from os import listdir
from os.path import join, isfile, basename
from statistics import mean

import numpy as np

from Datasets.Dataset import resize_mask
from Utilities.Experiments.BaseExperiment import BaseExperiment
from Utilities.Image.Picture import Picture
from Utilities.Visualizers.BaseVisualizer import BaseVisualizer


class TransferabilityExperiment(BaseExperiment):

    def __init__(self, visualizer: BaseVisualizer, debug_root, dataset, attacked_samples_folder_path):
        self.attacked_samples_folder_path = attacked_samples_folder_path
        samples = [os.path.join(self.attacked_samples_folder_path, f) for f in listdir(attacked_samples_folder_path) if
                   isfile(join(attacked_samples_folder_path, f))]

        super().__init__(visualizer, debug_root, samples, dataset)

        self.target_masks_folder_old = os.path.join(self.attacked_samples_folder_path, "targetForgeryMasks")

        # create folder to store the detector's result on the pristine sample
        self.pristine_results_folder = os.path.join(self.debug_root, "pristine")
        os.makedirs(self.pristine_results_folder)

        # create folder to store the computed detector's result
        self.pristine_visualizations = os.path.join(self.pristine_results_folder, "visualizations")
        os.makedirs(self.pristine_visualizations)

        # create folder to store the computed detector's heatmaps
        self.pristine_heatmaps = os.path.join(self.pristine_results_folder, "heatmaps")
        os.makedirs(self.pristine_heatmaps)

        # create folder to store the attacked samples
        self.attacked_samples_folder = os.path.join(self.debug_root, "attackedSamples")
        os.makedirs(self.attacked_samples_folder)

        # create folder to store the computed random target forgery masks
        self.target_masks_folder = os.path.join(self.attacked_samples_folder, "targetForgeryMasks")
        os.makedirs(self.target_masks_folder)

    def process_sample(self, sample_path, gt_mask):
        filename = basename(sample_path)

        sample_name = basename(sample_path)

        pristine_sample = Picture(path=sample_path)

        original_forgery_mask = Picture(resize_mask(gt_mask, (pristine_sample.shape[1], pristine_sample.shape[0])))

        target_forgery_mask_path = os.path.join(self.target_masks_folder_old, sample_name)

        target_forgery_mask = Picture(target_forgery_mask_path)
        target_forgery_mask = np.rint(Picture(target_forgery_mask[:, :, 0]) / 255)

        self.visualizer.process_sample(sample_path)
        heatmap = self.visualizer.metadata["heatmap"]

        self.visualizer.save_prediction_pipeline(os.path.join(self.pristine_visualizations, sample_name),
                                                 mask=original_forgery_mask)

        # save pristine heatmap
        pristine_heatmap_path = os.path.join(self.pristine_heatmaps, filename.split(".")[0] + ".npy")
        np.save(pristine_heatmap_path, np.array(heatmap))

        target_forgery_mask_path = os.path.join(self.target_masks_folder, basename(sample_path))
        Picture(target_forgery_mask * 255).save(target_forgery_mask_path)

        del heatmap

