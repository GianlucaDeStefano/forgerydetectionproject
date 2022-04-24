import os
from pathlib import Path
import statistics as st

import numpy as np
from cv2 import PSNR

from Attacks import BaseAttack
from Datasets import Dataset
from Detectors.Noiseprint.noiseprintEngine import find_best_theshold
from Detectors.Noiseprint.utility.utility import jpeg_quality_of_file
from Utilities.Experiments.Attacks.BaseExperiment import BaseExperiment
from sklearn.metrics import f1_score, matthews_corrcoef

from Utilities.Image.Picture import Picture
from Utilities.Visualizers.NoiseprintVisualizer import NoiseprintVisualizer


class NoiseprintExperiment(BaseExperiment):

    def __init__(self, attack: BaseAttack, dataset: Dataset, possible_forgery_masks: list, debug_root: str,
                 test_authentic: bool):
        super(NoiseprintExperiment, self).__init__(attack, dataset, possible_forgery_masks, debug_root, test_authentic)
        self.visualizer = NoiseprintVisualizer()

        # F1 and MCC values of the original forgery mask after the attach using the original thresholds
        self.original_mask_original_f1s = []
        self.original_mask_original_mccs = []

        # F1 and MCC values of the target forgery mask after the attach using the original thresholds
        self.target_forgery_original_f1s = []
        self.target_forgery_original_mccs = []

        # F1 and MCC values of the original forgery mask after the attack using thresholds computed on the attacked heatmap
        # with as target the original forgery
        self.original_mask_original_attacked_f1s = []
        self.original_mask_original_attacked_mccs = []

        # F1 and MCC values of the target forgery mask after the attack using thresholds computed on the attacked heatmap
        # with as target the original forgery
        self.target_forgery_original_attacked_f1s = []
        self.target_forgery_original_attacked_mccs = []

        # F1 and MCC values of the original forgery mask after the attack using thresholds computed on the attacked heatmap
        # with as target the target forgery
        self.original_mask_target_attacked_f1s = []
        self.original_mask_target_attacked_mccs = []

        # F1 and MCC values of the target forgery mask after the attach using thresholds computed on the attacked heatmap
        # with as target the target forgery
        self.target_forgery_target_attacked_f1s = []
        self.target_forgery_target_attacked_mccs = []

        self.mcc_target_forgery = []

    def compute_scores(self, original_image, attacked_image, original_mask, target_mask):

        psrn = PSNR(original_image, attacked_image)

        self.PSNRs.append(psrn)

        quality_factor = 101
        try:
            quality_factor = jpeg_quality_of_file(original_image.path)
        except:
            pass
        self.visualizer.load_quality(quality_factor)

        # Compute the heatmap of the image after the attack
        final_heatmap, _ = self.visualizer._engine.detect(attacked_image.one_channel().to_float())

        initial_heatmap, _ = self.visualizer._engine.detect(original_image.one_channel().to_float())

        if not self.test_authentic:
            # EXPERIMENTS 1 & 2
            # Compute the thresholds that generate the mask with the best f1&MCC on the pristine image w.r.t the original forgery
            best_starting_f1_threshold_original = find_best_theshold(initial_heatmap, original_mask, f1_score)
            best_starting_mcc_threshold_original = find_best_theshold(initial_heatmap, original_mask, matthews_corrcoef)

            # segment the attacked heatmap using the thresholds computed on the pristine heatmap
            mask_original_f1 = self.visualizer._engine.get_mask(final_heatmap,
                                                                threshold=best_starting_f1_threshold_original)
            mask_original_mcc = self.visualizer._engine.get_mask(final_heatmap,
                                                                 threshold=best_starting_mcc_threshold_original)

            # compute the F1 and MCC scores w.r.t the original forgery mask
            self.original_mask_original_f1s.append(f1_score(original_mask.flatten(), mask_original_f1.flatten()))
            self.original_mask_original_mccs.append(
                matthews_corrcoef(original_mask.flatten(), mask_original_mcc.flatten()))

            # compute the F1 and MCC scores w.r.t the target forgery mask:
            self.target_forgery_original_f1s.append(f1_score(target_mask.flatten(), mask_original_f1.flatten()))
            self.target_forgery_original_mccs.append(
                matthews_corrcoef(target_mask.flatten(), mask_original_mcc.flatten()))

            self.logger_module.info("#1  F1:{:.2} MCC:{:.2}".format(st.mean(self.original_mask_original_f1s),
                                                                    st.mean(self.original_mask_original_mccs)))

            self.logger_module.info("#2  F1:{:.2} MCC:{:.2}".format(st.mean(self.target_forgery_original_f1s),
                                                                    st.mean(self.target_forgery_original_mccs)))

            # EXPERIMENTS 3 & 4
            # Compute the thresholds that generate the mask with the best f1&MCC on the attacked image w.r.t the original forgery
            best_attacked_f1_threshold_original = find_best_theshold(final_heatmap, original_mask, f1_score)
            best_attacked_mcc_threshold_original = find_best_theshold(final_heatmap, original_mask, matthews_corrcoef)

            # segment the attacked heatmap using the thresholds computed on the attacked heatmap
            mask_original_attacked_f1 = self.visualizer._engine.get_mask(final_heatmap,
                                                                         threshold=best_attacked_f1_threshold_original)
            mask_original_attacked_mcc = self.visualizer._engine.get_mask(final_heatmap,
                                                                          threshold=best_attacked_mcc_threshold_original)

            # compute the F1 and MCC scores w.r.t the original forgery mask
            self.original_mask_original_attacked_f1s.append(
                f1_score(original_mask.flatten(), mask_original_attacked_f1.flatten()))
            self.original_mask_original_mccs.append(
                matthews_corrcoef(original_mask.flatten(), mask_original_attacked_mcc.flatten()))

            # compute the F1 and MCC scores w.r.t the target forgery mask
            self.target_forgery_original_attacked_f1s.append(
                f1_score(target_mask.flatten(), mask_original_attacked_f1.flatten()))
            self.target_forgery_original_attacked_mccs.append(
                matthews_corrcoef(target_mask.flatten(), mask_original_attacked_mcc.flatten()))

            self.logger_module.info("#3  F1:{:.2} MCC:{:.2}".format(st.mean(self.original_mask_original_attacked_f1s),
                                                                    st.mean(self.original_mask_original_mccs)))

            self.logger_module.info("#4  F1:{:.2} MCC:{:.2}".format(st.mean(self.target_forgery_original_attacked_f1s),
                                                                    st.mean(
                                                                        self.target_forgery_original_attacked_mccs)))

        # EXPERIMENTS 5 & 6
        # Compute the thresholds that generate the mask with the best f1&MCC on the attacked image w.r.t the original forgery
        best_attacked_f1_threshold_attacked = find_best_theshold(final_heatmap, target_mask, f1_score)
        best_attacked_mcc_threshold_attacked = find_best_theshold(final_heatmap, target_mask, matthews_corrcoef)

        # segment the attacked heatmap using the thresholds computed on the attacked heatmap
        mask_target_attacked_f1 = self.visualizer._engine.get_mask(final_heatmap,
                                                                   threshold=best_attacked_f1_threshold_attacked)
        mask_target_attacked_mcc = self.visualizer._engine.get_mask(final_heatmap,
                                                                    threshold=best_attacked_mcc_threshold_attacked)
        # compute the F1 and MCC scores w.r.t the original forgery mask
        self.original_mask_target_attacked_f1s.append(
            f1_score(original_mask.flatten(), mask_target_attacked_f1.flatten()))
        self.original_mask_target_attacked_mccs.append(
            matthews_corrcoef(original_mask.flatten(), mask_target_attacked_mcc.flatten()))

        # compute the F1 and MCC scores w.r.t the original forgery mask
        self.target_forgery_target_attacked_f1s.append(
            f1_score(target_mask.flatten(), mask_target_attacked_f1.flatten()))
        self.target_forgery_target_attacked_mccs.append(
            matthews_corrcoef(target_mask.flatten(), mask_target_attacked_mcc.flatten()))

        self.logger_module.info("#5 F1:{:.2} MCC:{:.2}".format(st.mean(self.original_mask_target_attacked_f1s),
                                                               st.mean(self.original_mask_target_attacked_mccs)))

        self.logger_module.info("#6  F1:{:.2} MCC:{:.2}".format(st.mean(self.target_forgery_target_attacked_f1s),
                                                                st.mean(self.target_forgery_target_attacked_mccs)))

        output_path = os.path.join(self.debug_foler, Path(original_image.path).stem)

        self.visualizer.save_prediction_pipeline(original_image, original_mask, initial_heatmap, target_mask, final_heatmap,
                                                 output_path)
