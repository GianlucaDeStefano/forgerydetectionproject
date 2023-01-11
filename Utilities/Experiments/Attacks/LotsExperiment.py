import os.path
import traceback
from os.path import basename
from statistics import mean
import cv2
import numpy as np
from Attacks.BaseWhiteBoxAttack import BaseWhiteBoxAttack
from Datasets.Dataset import resize_mask
from Detectors.DetectorEngine import find_optimal_mask
from Utilities.Experiments.BaseExperiment import BaseExperiment
from Utilities.Image.Picture import Picture
from Utilities.Image.functions import create_random_nonoverlapping_mask
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef as mcc


class LotsExperiment(BaseExperiment):

    def __init__(self, attack: BaseWhiteBoxAttack, debug_root, samples, dataset):
        super().__init__(attack.visualizer, debug_root, samples, dataset)
        self.attack = attack

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

        # create folder to store the computed detector's result
        self.attacked_visualizations = os.path.join(self.attacked_samples_folder, "visualizations")
        os.makedirs(self.attacked_visualizations)

        # create folder to store the computed detector's heatmaps
        self.attacked_heatmaps = os.path.join(self.attacked_samples_folder, "heatmaps")
        os.makedirs(self.attacked_heatmaps)

    def process_sample(self, sample_path, gt_mask):
        try:
            filename = basename(sample_path)

            # establish original and target forgery masks
            pristine_sample = Picture(path=sample_path)

            # load the original forgery mask and enforce that its width and height are the same of the given sample by resizing it
            original_forgery_mask = Picture(resize_mask(gt_mask, (pristine_sample.shape[1], pristine_sample.shape[0])))

            # setup the attack
            self.attack.setup(sample_path, original_forgery_mask, sample_path, original_forgery_mask)

            # save result of the detector on the pristine image
            self.visualizer.save_prediction_pipeline(os.path.join(self.pristine_visualizations, filename),
                                                     original_forgery_mask)

            # compute the pristine heatmap
            heatmap_pristine = self.visualizer.metadata["heatmap"]

            # save pristine heatmap
            pristine_heatmap_path = os.path.join(self.pristine_heatmaps, filename.split(".")[0] + ".npy")
            np.save(pristine_heatmap_path, np.array(heatmap_pristine))

            print("Executing the attack ...")

            # execute the attack
            _, attacked_sample = self.attack.execute()

            # save the attacked sample in the file system
            attacked_sample_path = os.path.join(self.attacked_samples_folder, filename)
            attacked_sample.save(attacked_sample_path)

            # compute the attacked heatmap
            self.visualizer.process_sample(attacked_sample_path)
            heatmap_attacked = self.visualizer.metadata["heatmap"]

            # save attacked heatmap
            attacked_heatmap_path = os.path.join(self.attacked_heatmaps, filename.split(".")[0] + ".npy")
            np.save(attacked_heatmap_path, np.array(heatmap_attacked))

            # save result of the detector on the attacked image
            self.visualizer.save_prediction_pipeline(os.path.join(self.attacked_visualizations, filename),original_forgery_mask)

            del heatmap_attacked, heatmap_pristine

        except:
            pass