import os

import numpy as np

from Datasets.Dataset import Dataset
from Detectors.Noiseprint.noiseprintEngine import find_best_theshold
from Detectors.Noiseprint.utility.utility import jpeg_quality_of_file
from Utilities.Experiments.Transferability.TransferabilityExperiment import TransferabilityExperiment
from sklearn.metrics import f1_score, matthews_corrcoef
import statistics as st

from Utilities.Visualizers.ExifVisualizer import ExifVisualizer


class ExifTransferabilityExperiment(TransferabilityExperiment):

    def __init__(self, debug_root, data_root_folder, original_dataset: Dataset):
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

        super().__init__(debug_root, data_root_folder, original_dataset, ExifVisualizer())

        self.original_heatmap_original_forgery_f1_results = []
        self.original_heatmap_original_forgery_mcc_results = []

        self.original_forgery_f1_results = []
        self.original_forgery_mcc_results = []
        self.target_forgery_f1_results = []
        self.target_forgery_mcc_results = []

    def _compute_scores(self, sample_name, original_image, attacked_image, original_forgery_mask, target_forgery_mask):
        initial_heatmap, initial_mask = self.visualizer._engine.detect(original_image.to_float())

        initial_mask= np.rint(initial_mask)

        self.visualizer.base_results(original_image, original_forgery_mask, initial_heatmap, initial_mask,
                                     os.path.join(self.original_results_dir, sample_name))

        self.original_heatmap_original_forgery_f1_results.append(f1_score(initial_mask.flatten(), original_forgery_mask.flatten()))
        self.original_heatmap_original_forgery_mcc_results.append(matthews_corrcoef(initial_mask.flatten(), original_forgery_mask.flatten()))


        self.logger_module.info("Base results (no attack) F1:{:.2} MCC:{:.2}".format(st.mean(self.original_heatmap_original_forgery_f1_results),
                                                                            st.mean(self.original_heatmap_original_forgery_mcc_results)))

        # Compute the heatmap of the image after the attack
        heatmap, mask = self.visualizer._engine.detect(attacked_image.to_float())

        mask = np.rint(mask)

        self.original_forgery_f1_results.append(f1_score(mask.flatten(), original_forgery_mask.flatten()))
        self.original_forgery_mcc_results.append(matthews_corrcoef(mask.flatten(), original_forgery_mask.flatten()))

        self.logger_module.info("original forgery F1:{:.2} MCC:{:.2}".format(st.mean(self.original_forgery_f1_results),
                                                                             st.mean(
                                                                                 self.original_forgery_mcc_results)))

        self.target_forgery_f1_results.append(f1_score(mask.flatten(), target_forgery_mask.flatten()))
        self.target_forgery_mcc_results.append(matthews_corrcoef(mask.flatten(), target_forgery_mask.flatten()))

        self.logger_module.info("target forgery F1:{:.2} MCC:{:.2}".format(st.mean(self.target_forgery_f1_results),
                                                                           st.mean(self.target_forgery_mcc_results)))

        self.visualizer.complete_pipeline(attacked_image, original_forgery_mask, initial_heatmap, target_forgery_mask,
                                          heatmap,mask,
                                          os.path.join(self.attacked_results_dir, sample_name))
