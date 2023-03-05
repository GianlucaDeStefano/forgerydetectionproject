import traceback
from statistics import mean

import cv2
import matplotlib
import sklearn as sklearn
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef as mcc
from tqdm import tqdm

from Detectors.DetectorEngine import find_optimal_mask

# matplotlib.use('TKAgg')
from matplotlib import pyplot as plt
from collections import defaultdict
from os import listdir
from os.path import join, basename, isfile

import numpy
import numpy as np

from Datasets.ColumbiaUncompressed.ColumbiaUncompressedDataset import ColumbiaUncompressedDataset
from Datasets.DSO.DsoDataset import DsoDataset
from Datasets.Dataset import resize_mask
from Utilities.Experiments.MetricGeneration import compute_score, normalize_heatmap
from Utilities.Logger.Logger import Logger


class MetricGeneratorLots(Logger):

    def __init__(self, execution_data_root, configs, dataset=None):
        self.execution_data_root = execution_data_root

        self.logger_module.info(f"Execution data root: {execution_data_root}")

        if not dataset:
            if "DSO" in self.execution_data_root:
                # load dataset DSO
                dataset = DsoDataset(configs["global"]["datasets"]["root"])
            elif "Columbia" in self.execution_data_root:
                # load datset columbia
                dataset = ColumbiaUncompressedDataset(configs["global"]["datasets"]["root"])
            else:
                raise Exception("Unknown dataset")

        self.dataset = dataset

        self.metrics = dict()

        # media metrics
        self.metrics["median-bg"] = []
        self.metrics["median-gt"] = []
        self.metrics["visibility-gt"] = []

        self.metrics["median-bg-attacked"] = []
        self.metrics["median-gt-attacked"] = []
        self.metrics["visibility-gt-attacked"] = []

        # EXPERIMENT 0: metrics computed on the pristine sample

        # F1 and MCC values of the original forgery mask before the attack using the original thresholds
        self.metrics["original_forgery_f1s_0"] = []
        self.metrics["original_forgery_mcc_0"] = []

        self.metrics["dr_gt_f1_0"] = []
        self.metrics["dr_bg_f1_0"] = []

        self.metrics["dr_gt_mcc_0"] = []
        self.metrics["dr_bg_mcc_0"] = []

        # EXPERIMENT 1: metrics computed on the attacked sample using the same thresholds as in the experiment 0

        # F1 and MCC values of the original forgery mask after the attack using the original thresholds
        self.metrics["original_forgery_f1s_1"] = []
        self.metrics["original_forgery_mccs_1"] = []

        self.metrics["dr_gt_f1_1"] = []
        self.metrics["dr_bg_f1_1"] = []

        self.metrics["dr_gt_mcc_1"] = []
        self.metrics["dr_bg_mcc_1"] = []

        # EXPERIMENT 2: metrics computed on the attacked sample computing the threshold using the original forgery mask
        # as target

        # F1 and MCC values of the original forgery mask after the attack using thresholds computed on the attacked
        # heatmap with as target the original forgery
        self.metrics["original_forgery_f1s_2"] = []
        self.metrics["original_forgery_mccs_2"] = []

        self.metrics["dr_gt_f1_2"] = []
        self.metrics["dr_bg_f1_2"] = []

        self.metrics["dr_gt_mcc_2"] = []
        self.metrics["dr_bg_mcc_2"] = []

        # EXPERIMENT 4: metric computed on the attacked sample computing the thresholds using the OTSU metric

        # F1 and MCC values of the original forgery mask after the attack using thresholds computed using OTSU
        self.metrics["original_forgery_f1s_4"] = []
        self.metrics["original_forgery_mccs_4"] = []

        self.metrics["dr_gt_4"] = []
        self.metrics["dr_bg_4"] = []

        # EXPERIMENT 5: metric computed on the attacked sample a threshold of 0.5

        # F1 and MCC values of the original forgery mask after the attack using a threshold of 0.5
        self.metrics["original_forgery_f1s_5"] = []
        self.metrics["original_forgery_mccs_5"] = []

        self.metrics["dr_gt_5"] = []
        self.metrics["dr_bg_5"] = []

        # EXPERIMENT 7: metric computed on the attacked sample computing the thresholds using the value at the 0.8 percentile

        # F1 and MCC values of the original forgery mask after the attack using as threshold a percentile of 0.8
        self.metrics["original_forgery_f1s_7"] = []
        self.metrics["original_forgery_mccs_7"] = []

        self.metrics["dr_gt_7"] = []
        self.metrics["dr_bg_7"] = []

        # AUC computed w.r.t the gt forgery on the pristine heatmap
        self.metrics["auc_gt_pre"] = []

        # AUC computed w.r.t the gt forgery on the attacked heatmap
        self.metrics["auc_gt_post"] = []


    def process(self):

        for sample_path in tqdm(self.get_samples_to_process()):

            try:
                sample_name = basename(sample_path)

                original_forgery_mask = self.get_original_forgery_mask(sample_path)

                original_heatmap = normalize_heatmap(self.get_pristine_heatmap(sample_name))
                attacked_heatmap = normalize_heatmap(self.get_attacked_heatmap(sample_name))

                # compute metric on the pristine heatmap
                median_bg, median_gt, visibility_gt = self.compute_metric_of_heatmap(
                    original_heatmap, original_forgery_mask)

                self.metrics["median-bg"].append(median_bg)
                self.metrics["median-gt"].append(median_gt)
                self.metrics["visibility-gt"].append(visibility_gt)

                # compute metric on the attacked heatmap
                median_bg_attacked, median_gt_attacked, visibility_gt_attacked = self.compute_metric_of_heatmap(
                    attacked_heatmap, original_forgery_mask)

                self.metrics["median-bg-attacked"].append(median_bg_attacked)
                self.metrics["median-gt-attacked"].append(median_gt_attacked)
                self.metrics["visibility-gt-attacked"].append(visibility_gt_attacked)

                self.compute_thresholding_metrics(original_heatmap, attacked_heatmap, original_forgery_mask)
            except Exception as e:
                print("Exception: e")
                print(traceback.format_exc())

        for key, values in self.metrics.items():

            if not values:
                continue

            self.logger_module.info(f"key: {key}: {mean(values):.4f}")


    def compute_metric_of_heatmap(self, heatmap, original_forgery_mask):

        assert (original_forgery_mask.min() == 0 and original_forgery_mask.max() == 1)

        # compute properties on the rpistine heatmap
        median_bg = numpy.ma.median(np.ma.masked_where(original_forgery_mask == 1, heatmap))
        median_gt = numpy.ma.median(np.ma.masked_where(original_forgery_mask == 0, heatmap))

        visibility_gt = median_gt - median_bg

        return median_bg, median_gt, visibility_gt

    def compute_detection_rates(self, gt_mask, predicted_mask):

        dr_gt = np.ma.masked_where(gt_mask == 0, predicted_mask).sum() / gt_mask.sum()

        dr_bg = np.ma.masked_where(gt_mask == 1, predicted_mask).sum() / (gt_mask == 0).sum()

        return dr_gt, dr_bg

    def compute_thresholding_metrics(self, heatmap_pristine, heatmap_attacked, original_forgery_mask):

        # EXPERIMENT 0:
        # compute the base performance of the detector on the pristine sample saving the threshold for later use
        pristine_f1_original_mask, _, original_f1_threshold, mask_f1_0 = compute_score(
            heatmap_pristine, original_forgery_mask, f1_score)

        pristine_mcc_original_mask, _, original_mcc_threshold, mask_mcc_0 = compute_score(
            heatmap_pristine, original_forgery_mask, mcc,)

        # save results w.r.t original forgery
        self.metrics["original_forgery_f1s_0"].append(pristine_f1_original_mask)
        self.metrics["original_forgery_mcc_0"].append(pristine_mcc_original_mask)

        dr_gt_f1_0, dr_bg_f1_0 = self.compute_detection_rates(original_forgery_mask, mask_f1_0)

        self.metrics["dr_gt_f1_0"].append(dr_gt_f1_0)
        self.metrics["dr_bg_f1_0"].append(dr_bg_f1_0)

        dr_gt_mcc_0, dr_bg_mcc_0 = self.compute_detection_rates(original_forgery_mask, mask_mcc_0)
        self.metrics["dr_gt_mcc_0"].append(dr_gt_mcc_0)
        self.metrics["dr_bg_mcc_0"].append(dr_bg_mcc_0)

        # EXPERIMENT 1
        attacked_f1_original_mask_original_t, attacked_f1_target_mask_original_t, _, mask_f1_1 = compute_score(
            heatmap_attacked, original_forgery_mask, f1_score, None, original_f1_threshold,
            test_flipped=False)
        attacked_mcc_original_mask_original_t, attacked_mcc_target_mask_original_t, _, mask_mcc_1 = compute_score(
            heatmap_attacked, original_forgery_mask, mcc, None, original_f1_threshold,
            test_flipped=False)

        self.metrics["original_forgery_f1s_1"].append(attacked_f1_original_mask_original_t)
        self.metrics["original_forgery_mccs_1"].append(attacked_mcc_original_mask_original_t)

        dr_gt_f1_1, dr_bg_f1_1 = self.compute_detection_rates(original_forgery_mask, mask_f1_1)

        self.metrics["dr_gt_f1_1"].append(dr_gt_f1_1)
        self.metrics["dr_bg_f1_1"].append(dr_bg_f1_1)

        dr_gt_mcc_1, dr_bg_mcc_1 = self.compute_detection_rates(original_forgery_mask, mask_mcc_1)
        self.metrics["dr_gt_mcc_1"].append(dr_gt_mcc_1)
        self.metrics["dr_bg_mcc_1"].append(dr_bg_mcc_1)

        # EXPERIMENT 2
        attacked_f1_original_mask_original_recomputed_t, _, _, mask_f1_2 = compute_score(
            heatmap_attacked, original_forgery_mask, f1_score, test_flipped=False)
        attacked_mcc_original_mask_original_recomputed_t, _, _, mask_mcc_2 = compute_score(
            heatmap_attacked, original_forgery_mask, mcc, test_flipped=False)

        self.metrics["original_forgery_f1s_2"].append(attacked_f1_original_mask_original_recomputed_t)
        self.metrics["original_forgery_mccs_2"].append(attacked_mcc_original_mask_original_recomputed_t)

        dr_gt_f1_2, dr_bg_f1_2 = self.compute_detection_rates(original_forgery_mask, mask_f1_2)

        self.metrics["dr_gt_f1_2"].append(dr_gt_f1_2)
        self.metrics["dr_bg_f1_2"].append(dr_bg_f1_2)

        dr_gt_mcc_2, dr_bg_mcc_2 = self.compute_detection_rates(original_forgery_mask, mask_mcc_2)

        self.metrics["dr_gt_mcc_2"].append(dr_gt_mcc_2)
        self.metrics["dr_bg_mcc_2"].append(dr_bg_mcc_2)

        # EXPERIMENT 4:
        otsu_threshold, otsu_mask = cv2.threshold(np.array(np.rint(heatmap_attacked * 255), dtype=np.uint8), 0, 255,
                                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        otsu_mask = np.rint(otsu_mask / 255)
        # F1 and MCC values of the original forgery mask after the attack using thresholds computed using OTSU
        self.metrics["original_forgery_f1s_4"].append(f1_score(original_forgery_mask.flatten(), otsu_mask.flatten()))
        self.metrics["original_forgery_mccs_4"].append(mcc(original_forgery_mask.flatten(), otsu_mask.flatten()))

        dr_gt_mcc_4, dr_bg_mcc_4 = self.compute_detection_rates(original_forgery_mask, otsu_mask)

        self.metrics["dr_gt_4"].append(dr_gt_mcc_4)
        self.metrics["dr_bg_4"].append(dr_bg_mcc_4)

        # EXPERIMENT 5:
        # Use a threshold of 0.5 to segment the attacked heatmap

        mask_5 = np.where(heatmap_attacked > 0.5, 1, 0)

        # F1 and MCC values of the original forgery mask after the attack using thresholds computed using OTSU
        self.metrics["original_forgery_f1s_5"].append(f1_score(original_forgery_mask.flatten(), mask_5.flatten()))
        self.metrics["original_forgery_mccs_5"].append(mcc(original_forgery_mask.flatten(), mask_5.flatten()))

        dr_gt_mcc_5, dr_bg_mcc_5 = self.compute_detection_rates(original_forgery_mask, mask_5)

        self.metrics["dr_gt_5"].append(dr_gt_mcc_5)
        self.metrics["dr_bg_5"].append(dr_bg_mcc_5)

        # EXPERIMENT 7
        # Use as threshold the value of the quantile 0.8
        percentile = np.quantile(np.array(heatmap_attacked), 0.8)
        mask_7 = np.where(heatmap_attacked > percentile, 1, 0)
        # F1 and MCC values of the original forgery mask after the attack using thresholds computed using OTSU
        self.metrics["original_forgery_f1s_7"].append(f1_score(original_forgery_mask.flatten(), mask_7.flatten()))
        self.metrics["original_forgery_mccs_7"].append(mcc(original_forgery_mask.flatten(), mask_7.flatten()))

        dr_gt_mcc_7, dr_bg_mcc_7 = self.compute_detection_rates(original_forgery_mask, mask_7)

        self.metrics["dr_gt_7"].append(dr_gt_mcc_7)
        self.metrics["dr_bg_7"].append(dr_bg_mcc_7)

        print(original_forgery_mask.shape, original_forgery_mask.min(), original_forgery_mask.max())
        print(heatmap_pristine.shape, heatmap_pristine.min(), heatmap_pristine.max())
        print(heatmap_attacked.shape, heatmap_attacked.min(), heatmap_attacked.max())

        self.metrics["auc_gt_pre"].append(
            sklearn.metrics.roc_auc_score(original_forgery_mask.flatten(), heatmap_pristine.flatten()))

        self.metrics["auc_gt_post"].append(
            sklearn.metrics.roc_auc_score(original_forgery_mask.flatten(), heatmap_attacked.flatten()))

        print()

    def get_samples_to_process(self):
        return [join(self.attacked_samples_folder, f) for f in listdir(self.attacked_samples_folder) if
                isfile(join(self.attacked_samples_folder, f))]

    @property
    def attacked_samples_folder(self):
        return join(self.execution_data_root, "outputs", "attackedSamples")

    @property
    def attacked_samples_heatmaps_folder(self):
        return join(self.attacked_samples_folder, "heatmaps")

    @property
    def attacked_samples_target_forgery_masks_folder(self):
        return join(self.attacked_samples_folder, "targetForgeryMasks")

    @property
    def pristine_samples_folder(self):
        return join(self.execution_data_root, "outputs", "pristine")

    @property
    def pristine_samples_heatmaps_folder(self):
        return join(self.pristine_samples_folder, "heatmaps")

    def get_attacked_heatmap(self, sample_name):
        sample_name = basename(sample_name).split(".")[0]
        heatmap_path = join(self.attacked_samples_heatmaps_folder, sample_name + ".npy")
        return np.load(heatmap_path)

    def get_pristine_heatmap(self, sample_name):
        sample_name = basename(sample_name).split(".")[0]
        heatmap_path = join(self.pristine_samples_heatmaps_folder, sample_name + ".npy")
        return np.load(heatmap_path)

    def get_original_forgery_mask(self, sample_path):
        gt_mask, _ = self.dataset.get_mask_of_image(image_path=sample_path)
        img = cv2.imread(sample_path)
        original_forgery_mask = resize_mask(gt_mask, (img.shape[1], img.shape[0]))
        return original_forgery_mask

    def get_target_forgery_mask(self, sample_name):

        taregt_forgery_mask = cv2.imread(
            join(self.attacked_samples_target_forgery_masks_folder, sample_name)) / (3 * 255)

        if taregt_forgery_mask.shape[2] == 3:
            taregt_forgery_mask = np.rint(
                taregt_forgery_mask[:, :, 0] + taregt_forgery_mask[:, :, 1] + taregt_forgery_mask[:, :, 2])
        return taregt_forgery_mask