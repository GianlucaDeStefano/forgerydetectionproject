import traceback
from statistics import mean

import cv2
import matplotlib
import sklearn as sklearn
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef as mcc

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
from Utilities.Logger.Logger import Logger


class MetricGenerator(Logger):

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
        self.metrics["median-decoy"] = []
        self.metrics["visibility-decoy"] = []
        self.metrics["visibility-gt"] = []

        self.metrics["median-bg-attacked"] = []
        self.metrics["median-gt-attacked"] = []
        self.metrics["median-decoy-attacked"] = []
        self.metrics["visibility-decoy-attacked"] = []
        self.metrics["visibility-gt-attacked"] = []

        # EXPERIMENT 0: metrics computed on the pristine sample

        # F1 and MCC values of the original forgery mask before the attack using the original thresholds
        self.metrics["original_forgery_f1s_0"] = []
        self.metrics["original_forgery_mcc_0"] = []

        # F1 and MCC values of the original forgery mask before the attack using the original thresholds
        self.metrics["target_forgery_f1s_0"] = []
        self.metrics["target_forgery_mcc_0"] = []

        self.metrics["dr_gt_f1_0"] = []
        self.metrics["dr_decoy_f1_0"] = []
        self.metrics["dr_bg_f1_0"] = []

        self.metrics["dr_gt_mcc_0"] = []
        self.metrics["dr_decoy_mcc_0"] = []
        self.metrics["dr_bg_mcc_0"] = []

        # EXPERIMENT 1: metrics computed on the attacked sample using the same thresholds as in the experiment 0

        # F1 and MCC values of the original forgery mask after the attack using the original thresholds
        self.metrics["original_forgery_f1s_1"] = []
        self.metrics["original_forgery_mccs_1"] = []

        # F1 and MCC values of the target forgery mask after the attack using the original thresholds
        self.metrics["target_forgery_f1s_1"] = []
        self.metrics["target_forgery_mccs_1"] = []

        self.metrics["dr_gt_f1_1"] = []
        self.metrics["dr_decoy_f1_1"] = []
        self.metrics["dr_bg_f1_1"] = []

        self.metrics["dr_gt_mcc_1"] = []
        self.metrics["dr_decoy_mcc_1"] = []
        self.metrics["dr_bg_mcc_1"] = []

        # EXPERIMENT 2: metrics computed on the attacked sample computing the threshold using the original forgery mask
        # as target

        # F1 and MCC values of the original forgery mask after the attack using thresholds computed on the attacked
        # heatmap with as target the original forgery
        self.metrics["original_forgery_f1s_2"] = []
        self.metrics["original_forgery_mccs_2"] = []

        # F1 and MCC values of the target forgery mask after the attack using thresholds computed on the attacked
        # heatmap with as target the original forgery
        self.metrics["target_forgery_f1s_2"] = []
        self.metrics["target_forgery_mccs_2"] = []

        self.metrics["dr_gt_f1_2"] = []
        self.metrics["dr_decoy_f1_2"] = []
        self.metrics["dr_bg_f1_2"] = []

        self.metrics["dr_gt_mcc_2"] = []
        self.metrics["dr_decoy_mcc_2"] = []
        self.metrics["dr_bg_mcc_2"] = []

        # EXPERIMENT 3: metric computed on the attacked sample computing the thresholds using the target forgery mask
        # as target

        # F1 and MCC values of the original forgery mask after the attack using thresholds computed on the attacked
        # heatmap with as target the target forgery
        self.metrics["original_forgery_f1s_3"] = []
        self.metrics["original_forgery_mccs_3"] = []

        # F1 and MCC values of the target forgery mask after the attach using thresholds computed on the attacked
        # heatmap with as target the target forgery
        self.metrics["target_forgery_f1s_3"] = []
        self.metrics["target_forgery_mccs_3"] = []

        self.metrics["dr_gt_f1_3"] = []
        self.metrics["dr_decoy_f1_3"] = []
        self.metrics["dr_bg_f1_3"] = []

        self.metrics["dr_gt_mcc_3"] = []
        self.metrics["dr_decoy_mcc_3"] = []
        self.metrics["dr_bg_mcc_3"] = []

        # EXPERIMENT 4: metric computed on the attacked sample computing the thresholds using the OTSU metric

        # F1 and MCC values of the original forgery mask after the attack using thresholds computed using OTSU
        self.metrics["original_forgery_f1s_4"] = []
        self.metrics["original_forgery_mccs_4"] = []

        # F1 and MCC values of the target forgery mask after the attack using thresholds computed using OTSU
        self.metrics["target_forgery_f1s_4"] = []
        self.metrics["target_forgery_mccs_4"] = []

        self.metrics["dr_gt_4"] = []
        self.metrics["dr_decoy_4"] = []
        self.metrics["dr_bg_4"] = []

        # EXPERIMENT 5: metric computed on the attacked sample a threshold of 0.5

        # F1 and MCC values of the original forgery mask after the attack using a threshold of 0.5
        self.metrics["original_forgery_f1s_5"] = []
        self.metrics["original_forgery_mccs_5"] = []

        # F1 and MCC values of the target forgery mask after the attack using a threshold of 0.5
        self.metrics["target_forgery_f1s_5"] = []
        self.metrics["target_forgery_mccs_5"] = []

        self.metrics["dr_gt_5"] = []
        self.metrics["dr_decoy_5"] = []
        self.metrics["dr_bg_5"] = []

        # EXPERIMENT 6: metric computed on the attacked sample computing the thresholds using percentile represented by the
        # area of the decoy

        # F1 and MCC values of the original forgery mask after the attack using as threshold the percentile of the decoy
        self.metrics["original_forgery_f1s_6"] = []
        self.metrics["original_forgery_mccs_6"] = []

        # F1 and MCC values of the target forgery mask after the attack using as threshold the percentile of the decoy
        self.metrics["target_forgery_f1s_6"] = []
        self.metrics["target_forgery_mccs_6"] = []

        self.metrics["dr_gt_6"] = []
        self.metrics["dr_decoy_6"] = []
        self.metrics["dr_bg_6"] = []

        self.metrics["quantile_6"] = []
        # EXPERIMENT 7: metric computed on the attacked sample computing the thresholds using the value at the 0.8 percentile

        # F1 and MCC values of the original forgery mask after the attack using as threshold a percentile of 0.8
        self.metrics["original_forgery_f1s_7"] = []
        self.metrics["original_forgery_mccs_7"] = []

        # F1 and MCC values of the target forgery mask after the attack using as threshold a percentile of 0.8
        self.metrics["target_forgery_f1s_7"] = []
        self.metrics["target_forgery_mccs_7"] = []

        self.metrics["dr_gt_7"] = []
        self.metrics["dr_decoy_7"] = []
        self.metrics["dr_bg_7"] = []

        # AUC computed w.r.t the gt forgery on the pristine heatmap
        self.metrics["auc_gt_pre"] = []

        # AUC computed w.r.t the decoy forgery on the pristine heatmap
        self.metrics["auc_dec_pre"] = []

        # AUC computed w.r.t the gt forgery on the attacked heatmap
        self.metrics["auc_gt_post"] = []

        # AUC computed w.r.t the decoy forgery on the attacked heatmap
        self.metrics["auc_dec_post"] = []

    def process(self):

        for sample_path in self.get_samples_to_process():

            try:
                sample_name = basename(sample_path)

                original_forgery_mask = self.get_original_forgery_mask(sample_path)

                target_forgery_mask_name = sample_name
                if "Columbia" in self.execution_data_root:
                    target_forgery_mask_name = target_forgery_mask_name.split('.')[0] + '.tif'

                target_forgery_mask = self.get_target_forgery_mask(target_forgery_mask_name)

                original_heatmap = self.get_pristine_heatmap(sample_name)
                attacked_heatmap = self.get_attacked_heatmap(sample_name)

                visualize_histogram(original_heatmap)
                visualize_histogram(attacked_heatmap)

                # compute metric on the pristine heatmap
                median_bg, median_gt, median_decoy, visibility_decoy, visibility_gt = self.compute_metric_of_heatmap(
                    original_heatmap, original_forgery_mask,
                    target_forgery_mask)

                self.metrics["median-bg"].append(median_bg)
                self.metrics["median-gt"].append(median_gt)
                self.metrics["median-decoy"].append(median_decoy)
                self.metrics["visibility-decoy"].append(visibility_decoy)
                self.metrics["visibility-gt"].append(visibility_gt)

                # compute metric on the attacked heatmap
                median_bg_attacked, median_gt_attacked, median_decoy_attacked, visibility_decoy_attacked, visibility_gt_attacked = self.compute_metric_of_heatmap(
                    attacked_heatmap, original_forgery_mask,
                    target_forgery_mask)

                self.metrics["median-bg-attacked"].append(median_bg_attacked)
                self.metrics["median-gt-attacked"].append(median_gt_attacked)
                self.metrics["median-decoy-attacked"].append(median_decoy_attacked)
                self.metrics["visibility-decoy-attacked"].append(visibility_decoy_attacked)
                self.metrics["visibility-gt-attacked"].append(visibility_gt_attacked)

                self.compute_thresholding_metrics(original_heatmap, attacked_heatmap, original_forgery_mask,
                                                  target_forgery_mask)
            except Exception as e:
                print("Exception: e")
                print(traceback.format_exc())

        for key, values in self.metrics.items():
            self.logger_module.info(f"key: {key}: {mean(values):.4f}")

    def compute_metric_of_heatmap(self, heatmap, original_forgery_mask, target_forgery_mask):

        assert (heatmap.min() == 0 and heatmap.max() == 1)
        assert (original_forgery_mask.min() == 0 and original_forgery_mask.max() == 1)
        assert (target_forgery_mask.min() == 0 and target_forgery_mask.max() == 1)

        combined_mask = (original_forgery_mask + target_forgery_mask) > 0

        # compute properties on the rpistine heatmap
        median_bg = numpy.ma.median(np.ma.masked_where(combined_mask == 1, heatmap))
        median_gt = numpy.ma.median(np.ma.masked_where(original_forgery_mask == 0, heatmap))
        median_decoy = numpy.ma.median(np.ma.masked_where(target_forgery_mask == 0, heatmap))

        visibility_decoy = median_decoy - median_bg
        visibility_gt = median_gt - median_bg

        return median_bg, median_gt, median_decoy, visibility_decoy, visibility_gt

    def compute_detection_rates(self, gt_mask, decoy_mask, combined_mask, predicted_mask):

        dr_gt = np.ma.masked_where(gt_mask == 0, predicted_mask).sum() / gt_mask.sum()

        dr_decoy = np.ma.masked_where(decoy_mask == 0, predicted_mask).sum() / decoy_mask.sum()

        dr_bg = np.ma.masked_where(combined_mask == 1, predicted_mask).sum() / (decoy_mask == 0).sum()

        return dr_gt, dr_decoy, dr_bg

    def compute_thresholding_metrics(self, heatmap_pristine, heatmap_attacked, original_forgery_mask,
                                     target_forgery_mask):
        combined_mask = (original_forgery_mask + target_forgery_mask) > 0

        # EXPERIMENT 0:
        # compute the base performance of the detector on the pristine sample saving the threshold for later use
        pristine_f1_original_mask, pristine_f1_target_forgery_mask, original_f1_threshold, mask_f1_0 = compute_score(
            heatmap_pristine, original_forgery_mask, f1_score, target_forgery_mask)
        pristine_mcc_original_mask, pristine_mcc_target_forgery_mask, original_mcc_threshold, mask_mcc_0 = compute_score(
            heatmap_pristine, original_forgery_mask, mcc, target_forgery_mask)

        # save results w.r.t original forgery
        self.metrics["original_forgery_f1s_0"].append(pristine_f1_original_mask)
        self.metrics["original_forgery_mcc_0"].append(pristine_mcc_original_mask)

        # save results w.r.t target forgery
        self.metrics["target_forgery_f1s_0"].append(pristine_f1_target_forgery_mask)
        self.metrics["target_forgery_mcc_0"].append(pristine_mcc_target_forgery_mask)

        dr_gt_f1_0, dr_decoy_f1_0, dr_bg_f1_0 = self.compute_detection_rates(original_forgery_mask, target_forgery_mask,
                                                                             combined_mask, mask_f1_0)

        self.metrics["dr_gt_f1_0"].append(dr_gt_f1_0)
        self.metrics["dr_decoy_f1_0"].append(dr_decoy_f1_0)
        self.metrics["dr_bg_f1_0"].append(dr_bg_f1_0)

        dr_gt_mcc_0, dr_decoy_mcc_0, dr_bg_mcc_0 = self.compute_detection_rates(original_forgery_mask,
                                                                                target_forgery_mask,
                                                                                combined_mask, mask_mcc_0)
        self.metrics["dr_gt_mcc_0"].append(dr_gt_mcc_0)
        self.metrics["dr_decoy_mcc_0"].append(dr_decoy_mcc_0)
        self.metrics["dr_bg_mcc_0"].append(dr_bg_mcc_0)

        # EXPERIMENT 1
        attacked_f1_original_mask_original_t, attacked_f1_target_mask_original_t, _, mask_f1_1 = compute_score(
            heatmap_attacked, original_forgery_mask, f1_score, target_forgery_mask, original_f1_threshold,
            test_flipped=False)
        attacked_mcc_original_mask_original_t, attacked_mcc_target_mask_original_t, _, mask_mcc_1 = compute_score(
            heatmap_attacked, original_forgery_mask, mcc, target_forgery_mask, original_f1_threshold,
            test_flipped=False)

        self.metrics["original_forgery_f1s_1"].append(attacked_f1_original_mask_original_t)
        self.metrics["original_forgery_mccs_1"].append(attacked_mcc_original_mask_original_t)

        self.metrics["target_forgery_f1s_1"].append(attacked_f1_target_mask_original_t)
        self.metrics["target_forgery_mccs_1"].append(attacked_mcc_target_mask_original_t)

        dr_gt_f1_1, dr_decoy_f1_1, dr_bg_f1_1 = self.compute_detection_rates(original_forgery_mask, target_forgery_mask,
                                                                             combined_mask, mask_f1_1)

        self.metrics["dr_gt_f1_1"].append(dr_gt_f1_1)
        self.metrics["dr_decoy_f1_1"].append(dr_decoy_f1_1)
        self.metrics["dr_bg_f1_1"].append(dr_bg_f1_1)

        dr_gt_mcc_1, dr_decoy_mcc_1, dr_bg_mcc_1 = self.compute_detection_rates(original_forgery_mask,
                                                                                target_forgery_mask,
                                                                                combined_mask, mask_mcc_1)
        self.metrics["dr_gt_mcc_1"].append(dr_gt_mcc_1)
        self.metrics["dr_decoy_mcc_1"].append(dr_decoy_mcc_1)
        self.metrics["dr_bg_mcc_1"].append(dr_bg_mcc_1)

        # EXPERIMENT 2
        attacked_f1_original_mask_original_recomputed_t, attacked_f1_target_mask_original_recomputed_t, _, mask_f1_2 = compute_score(
            heatmap_attacked, original_forgery_mask, f1_score, target_forgery_mask, test_flipped=False)
        attacked_mcc_original_mask_original_recomputed_t, attacked_mcc_target_mask_original_recomputed_t, _, mask_mcc_2 = compute_score(
            heatmap_attacked, original_forgery_mask, mcc, target_forgery_mask, test_flipped=False)

        self.metrics["original_forgery_f1s_2"].append(attacked_f1_original_mask_original_recomputed_t)
        self.metrics["original_forgery_mccs_2"].append(attacked_mcc_original_mask_original_recomputed_t)

        self.metrics["target_forgery_f1s_2"].append(attacked_f1_target_mask_original_recomputed_t)
        self.metrics["target_forgery_mccs_2"].append(attacked_mcc_target_mask_original_recomputed_t)

        dr_gt_f1_2, dr_decoy_f1_2, dr_bg_f1_2 = self.compute_detection_rates(original_forgery_mask, target_forgery_mask,
                                                                             combined_mask, mask_f1_2)

        self.metrics["dr_gt_f1_2"].append(dr_gt_f1_2)
        self.metrics["dr_decoy_f1_2"].append(dr_decoy_f1_2)
        self.metrics["dr_bg_f1_2"].append(dr_bg_f1_2)

        dr_gt_mcc_2, dr_decoy_mcc_2, dr_bg_mcc_2 = self.compute_detection_rates(original_forgery_mask,
                                                                                target_forgery_mask,
                                                                                combined_mask, mask_mcc_2)

        self.metrics["dr_gt_mcc_2"].append(dr_gt_mcc_2)
        self.metrics["dr_decoy_mcc_2"].append(dr_decoy_mcc_2)
        self.metrics["dr_bg_mcc_2"].append(dr_bg_mcc_2)

        # EXPERIMENT 3
        attacked_f1_target_mask_target_recomputed_t, attacked_f1_original_mask_target_recomputed_t, _, mask_f1_3 = compute_score(
            heatmap_attacked, target_forgery_mask, f1_score, original_forgery_mask, test_flipped=False)
        attacked_mcc_target_mask_target_recomputed_t, attacked_mcc_original_mask_target_recomputed_t, _, mask_mcc_3 = compute_score(
            heatmap_attacked, target_forgery_mask, mcc, original_forgery_mask, test_flipped=False)

        self.metrics["original_forgery_f1s_3"].append(attacked_f1_original_mask_target_recomputed_t)
        self.metrics["original_forgery_mccs_3"].append(attacked_mcc_original_mask_target_recomputed_t)

        self.metrics["target_forgery_f1s_3"].append(attacked_f1_target_mask_target_recomputed_t)
        self.metrics["target_forgery_mccs_3"].append(attacked_mcc_target_mask_target_recomputed_t)

        dr_gt_f1_3, dr_decoy_f1_3, dr_bg_f1_3 = self.compute_detection_rates(original_forgery_mask, target_forgery_mask,
                                                                             combined_mask, mask_f1_3)

        self.metrics["dr_gt_f1_3"].append(dr_gt_f1_3)
        self.metrics["dr_decoy_f1_3"].append(dr_decoy_f1_3)
        self.metrics["dr_bg_f1_3"].append(dr_bg_f1_3)

        dr_gt_mcc_3, dr_decoy_mcc_3, dr_bg_mcc_3 = self.compute_detection_rates(original_forgery_mask,
                                                                                target_forgery_mask,
                                                                                combined_mask, mask_mcc_3)
        self.metrics["dr_gt_mcc_3"].append(dr_gt_mcc_3)
        self.metrics["dr_decoy_mcc_3"].append(dr_decoy_mcc_3)
        self.metrics["dr_bg_mcc_3"].append(dr_bg_mcc_3)

        # EXPERIMENT 4:

        otsu_threshold, otsu_mask = cv2.threshold(np.array(np.rint(heatmap_attacked * 255), dtype=np.uint8), 0, 255,
                                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        otsu_mask = np.rint(otsu_mask / 255)
        # F1 and MCC values of the original forgery mask after the attack using thresholds computed using OTSU
        self.metrics["original_forgery_f1s_4"].append(f1_score(original_forgery_mask.flatten(), otsu_mask.flatten()))
        self.metrics["original_forgery_mccs_4"].append(mcc(original_forgery_mask.flatten(), otsu_mask.flatten()))

        # F1 and MCC values of the target forgery mask after the attack using thresholds computed using OTSU
        self.metrics["target_forgery_f1s_4"].append(f1_score(target_forgery_mask.flatten(), otsu_mask.flatten()))
        self.metrics["target_forgery_mccs_4"].append(mcc(target_forgery_mask.flatten(), otsu_mask.flatten()))

        dr_gt_mcc_4, dr_decoy_mcc_4, dr_bg_mcc_4 = self.compute_detection_rates(original_forgery_mask,
                                                                                target_forgery_mask,
                                                                                combined_mask, otsu_mask)

        self.metrics["dr_gt_4"].append(dr_gt_mcc_4)
        self.metrics["dr_decoy_4"].append(dr_decoy_mcc_4)
        self.metrics["dr_bg_4"].append(dr_bg_mcc_4)

        # EXPERIMENT 5:
        # Use a threshold of 0.5 to segment the attacked heatmap

        mask_5 = np.where(heatmap_attacked > 0.5, 1, 0)

        # F1 and MCC values of the original forgery mask after the attack using thresholds computed using OTSU
        self.metrics["original_forgery_f1s_5"].append(f1_score(original_forgery_mask.flatten(), mask_5.flatten()))
        self.metrics["original_forgery_mccs_5"].append(mcc(original_forgery_mask.flatten(), mask_5.flatten()))

        # F1 and MCC values of the target forgery mask after the attack using thresholds computed using OTSU
        self.metrics["target_forgery_f1s_5"].append(f1_score(target_forgery_mask.flatten(), mask_5.flatten()))
        self.metrics["target_forgery_mccs_5"].append(mcc(target_forgery_mask.flatten(), mask_5.flatten()))

        dr_gt_mcc_5, dr_decoy_mcc_5, dr_bg_mcc_5 = self.compute_detection_rates(original_forgery_mask,
                                                                                target_forgery_mask,
                                                                                combined_mask, mask_5)

        self.metrics["dr_gt_5"].append(dr_gt_mcc_5)
        self.metrics["dr_decoy_5"].append(dr_decoy_mcc_5)
        self.metrics["dr_bg_5"].append(dr_bg_mcc_5)

        # EXPERIMENT 6
        # Use as threshold the value of the quantile corresponding to the area of the decoy mask
        q = 1 - (target_forgery_mask.sum() / target_forgery_mask.size)
        heatmap_attacked = np.asarray(heatmap_attacked)
        t = np.quantile(np.array(heatmap_attacked), q)
        mask_6 = np.where(heatmap_attacked > t, 1, 0)

        # F1 and MCC values of the original forgery mask after the attack using thresholds computed using OTSU
        self.metrics["original_forgery_f1s_6"].append(f1_score(original_forgery_mask.flatten(), mask_6.flatten()))
        self.metrics["original_forgery_mccs_6"].append(mcc(original_forgery_mask.flatten(), mask_6.flatten()))

        # F1 and MCC values of the target forgery mask after the attack using thresholds computed using OTSU
        self.metrics["target_forgery_f1s_6"].append(f1_score(target_forgery_mask.flatten(), mask_6.flatten()))
        self.metrics["target_forgery_mccs_6"].append(mcc(target_forgery_mask.flatten(), mask_6.flatten()))

        dr_gt_mcc_6, dr_decoy_mcc_6, dr_bg_mcc_6 = self.compute_detection_rates(original_forgery_mask,
                                                                                target_forgery_mask,
                                                                                combined_mask, mask_6)

        self.metrics["dr_gt_6"].append(dr_gt_mcc_6)
        self.metrics["dr_decoy_6"].append(dr_decoy_mcc_6)
        self.metrics["dr_bg_6"].append(dr_bg_mcc_6)

        self.metrics["quantile_6"].append(q)

        # EXPERIMENT 7
        # Use as threshold the value of the quantile 0.8
        percentile = np.quantile(np.array(heatmap_attacked), 0.8)
        mask_7 = np.where(heatmap_attacked > percentile, 1, 0)
        # F1 and MCC values of the original forgery mask after the attack using thresholds computed using OTSU
        self.metrics["original_forgery_f1s_7"].append(f1_score(original_forgery_mask.flatten(), mask_7.flatten()))
        self.metrics["original_forgery_mccs_7"].append(mcc(original_forgery_mask.flatten(), mask_7.flatten()))

        # F1 and MCC values of the target forgery mask after the attack using thresholds computed using OTSU
        self.metrics["target_forgery_f1s_7"].append(f1_score(target_forgery_mask.flatten(), mask_7.flatten()))
        self.metrics["target_forgery_mccs_7"].append(mcc(target_forgery_mask.flatten(), mask_7.flatten()))

        dr_gt_mcc_7, dr_decoy_mcc_7, dr_bg_mcc_7 = self.compute_detection_rates(original_forgery_mask,
                                                                                target_forgery_mask,
                                                                                combined_mask, mask_7)

        self.metrics["dr_gt_7"].append(dr_gt_mcc_7)
        self.metrics["dr_decoy_7"].append(dr_decoy_mcc_7)
        self.metrics["dr_bg_7"].append(dr_bg_mcc_7)

        print(original_forgery_mask.shape, original_forgery_mask.min(), original_forgery_mask.max())
        print(target_forgery_mask.shape, target_forgery_mask.min(), target_forgery_mask.max())
        print(heatmap_pristine.shape, heatmap_pristine.min(), heatmap_pristine.max())
        print(heatmap_attacked.shape, heatmap_attacked.min(), heatmap_attacked.max())

        self.metrics["auc_gt_pre"].append(
            sklearn.metrics.roc_auc_score(original_forgery_mask.flatten(), heatmap_pristine.flatten()))
        self.metrics["auc_dec_pre"].append(
            sklearn.metrics.roc_auc_score(target_forgery_mask.flatten(), heatmap_pristine.flatten()))

        self.metrics["auc_gt_post"].append(
            sklearn.metrics.roc_auc_score(original_forgery_mask.flatten(), heatmap_attacked.flatten()))
        self.metrics["auc_dec_post"].append(
            sklearn.metrics.roc_auc_score(target_forgery_mask.flatten(), heatmap_attacked.flatten()))

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


def visualize_heatmap(heatmap,path=""):
    fig, axs = plt.subplots(1, 1, figsize=(5, 5))
    plt.axis('off')
    axs.imshow(heatmap, clim=[0, 1], cmap='jet')

    if path:
        plt.savefig(path,dpi=600,bbox_inches='tight')

    plt.show()


def visualize_mask(heatmap):
    fig, axs = plt.subplots(1, 1, figsize=(5, 5))

    axs.imshow(heatmap, clim=[0, 1], cmap='gray')

    plt.show()




def visualize_histogram(heatmap):
    im = np.rint(heatmap * 255).astype(np.uint8)
    plt.hist(im.ravel(), 256, (0, 256))
    plt.show()


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
    mask = None
    if threshold is not None:
        mask = np.where(heatmap > threshold, 1, 0)
        first_score = metric(mask.flatten(), target_mask.flatten())
        if second_mask is not None:
            second_score = metric(mask.flatten(), second_mask.flatten())
    else:

        mask, first_score, threshold = find_optimal_mask(heatmap, target_mask, metric,
                                                         test_flipped=test_flipped)
        if second_mask is not None:
            second_score = metric(mask.flatten(), second_mask.flatten())

    return float(first_score), float(second_score), threshold, mask
