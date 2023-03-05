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


class MimicryExperiment(BaseExperiment):

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

        # create folder to store the computed random target forgery masks
        self.target_masks_folder = os.path.join(self.attacked_samples_folder, "targetForgeryMasks")
        os.makedirs(self.target_masks_folder)

        # create folder to store the computed detector's result
        self.attacked_visualizations = os.path.join(self.attacked_samples_folder, "visualizations")
        os.makedirs(self.attacked_visualizations)

        # create folder to store the computed detector's heatmaps
        self.attacked_heatmaps = os.path.join(self.attacked_samples_folder, "heatmaps")
        os.makedirs(self.attacked_heatmaps)

    def process_sample(self, sample_path, gt_mask):
        filename = basename(sample_path)

        # establish original and target forgery masks
        pristine_sample = Picture(path=sample_path)

        # load the original forgery mask and enforce that its width and height are the same of the given sample by resizing it
        original_forgery_mask = Picture(resize_mask(gt_mask, (pristine_sample.shape[1], pristine_sample.shape[0])))

        target_forgery_mask = Picture(create_random_nonoverlapping_mask(original_forgery_mask))

        # save the target_forgery_mask to use it in the transferability experiments
        target_forgery_mask_path = os.path.join(self.target_masks_folder, basename(sample_path))
        Picture(target_forgery_mask * 255).save(target_forgery_mask_path)

        # setup the attack
        self.attack.setup(sample_path, original_forgery_mask, sample_path, original_forgery_mask,
                          target_forgery_mask)

        # save result of the detector on the pristine image
        self.visualizer.save_prediction_pipeline(os.path.join(self.pristine_visualizations, filename),
                                                 original_forgery_mask)

        # compute the pristine heatmap
        heatmap_pristine = self.visualizer.metadata["heatmap"]

        # save pristine heatmap
        pristine_heatmap_path = os.path.join(self.pristine_heatmaps, filename.split(".")[0] + ".npy")
        np.save(pristine_heatmap_path, np.array(heatmap_pristine))

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
        self.visualizer.save_prediction_pipeline(os.path.join(self.attacked_visualizations, filename),
                                                 target_forgery_mask)

        del heatmap_attacked, heatmap_pristine
        del original_forgery_mask, target_forgery_mask
        del pristine_sample, attacked_sample

        self.visualizer.reset_metadata()

def compute_score(heatmap, target_mask, metric, second_mask=None, threshold=None, test_flipped=True):
    """
    Given an heatmap a target mask, and a metric find the theshold that when used on the heatmap produces the optimal mask
    w.r.t the metric and the target mask. Return the score computed w.r.t the target mask and also the secondary mask (if given)
    @param heatmap: heatmap to analyze
    @param target_mask: target mask to use in the threshold selection process
    @param metric: metric to use to compute the scores
    @param second_mask: second mask to use to compute the score
    @param threshold: default None, by default the threshold used is the one maximizing the metric
    @param test_flipped: use also the flipped heatmap to select the optimal mask
    @return: tuple (metric(target_mask),metric(second_mask),used_threshold)
    """
    assert heatmap.min() == 0 and heatmap.max() == 1

    first_score = None
    second_score = None

    if threshold is not None:
        mask = np.where(heatmap > threshold, 1, 0)
        first_score = metric(mask.flatten(), target_mask.flatten())
        if second_mask is not None:
            second_score = metric(mask.flatten(), second_mask.flatten())
    else:

        optimal_mask, first_score, threshold = find_optimal_mask(heatmap, target_mask, metric,
                                                                 test_flipped=test_flipped)
        if second_mask is not None:
            second_score = metric(optimal_mask.flatten(), second_mask.flatten())

    return float(first_score), float(second_score), threshold


def compute_PSNR(sample1_path, sample2_path):
    img1 = cv2.imread(str(sample1_path))
    img2 = cv2.imread(str(sample2_path))
    return cv2.PSNR(img1, img2)
