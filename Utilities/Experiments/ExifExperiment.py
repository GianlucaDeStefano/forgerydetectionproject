import os
from pathlib import Path

import numpy as np
from cv2 import PSNR

from Attacks import BaseAttack
from Datasets import Dataset
from Utilities.Experiments.BaseExperiment import BaseExperiment
from sklearn.metrics import f1_score, matthews_corrcoef
import statistics as st


class ExifExperiment(BaseExperiment):

    def __init__(self, attack: BaseAttack, dataset: Dataset, possible_forgery_masks: dict, debug_root: str,test_authentic:bool):
        super(ExifExperiment, self).__init__(attack, dataset, possible_forgery_masks, debug_root,test_authentic=test_authentic)
        self.visualizer = None

        self.f1_original_forgery = []
        self.f1_target_forgery = []

        self.mcc_original_forgery = []
        self.mcc_target_forgery = []

        self.logger_module.info("Starting the experiment")

    def compute_scores(self, original_image, attacked_image, original_mask, target_mask):
        psrn = PSNR(original_image, attacked_image)

        self.PSNRs.append(psrn)

        self.visualizer = self.attack.detector

        initial_heatmap, initial_mask = self.visualizer._engine.detect(original_image.to_float())

        final_heatmap, final_mask_original = self.visualizer._engine.detect(attacked_image.to_float(), original_mask)

        initial_mask = np.rint(initial_mask)
        final_mask_original = np.rint(final_mask_original)

        output_path = os.path.join(self.debug_foler, Path(original_image.path).stem)

        self.visualizer.complete_pipeline(original_image, original_mask, initial_heatmap, target_mask, final_heatmap,
                                          final_mask_original, output_path)

        if not self.test_authentic:
            self.f1_original_forgery.append(f1_score(original_mask.flatten(), final_mask_original.flatten()))
            self.mcc_original_forgery.append(matthews_corrcoef(original_mask.flatten(), final_mask_original.flatten()))

            self.logger_module.info("f1_original:{:.2} mcc_original:{:.2}".format(st.mean(self.f1_original_forgery),
                                                                              st.mean(self.mcc_original_forgery)))

        self.f1_target_forgery.append(f1_score(target_mask.flatten(), final_mask_original.flatten()))
        self.mcc_target_forgery.append(matthews_corrcoef(target_mask.flatten(), final_mask_original.flatten()))


        self.logger_module.info("f1_target:{} mcc_target:{}".format(st.mean(self.f1_target_forgery),
                                                                st.mean(self.mcc_target_forgery)))
