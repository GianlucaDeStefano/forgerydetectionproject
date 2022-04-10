import os.path

from Datasets.Dataset import Dataset
from Detectors.Noiseprint.noiseprintEngine import find_best_theshold
from Detectors.Noiseprint.utility.utility import jpeg_quality_of_file
from Utilities.Experiments.Transferability.TransferabilityExperiment import TransferabilityExperiment
from sklearn.metrics import f1_score, matthews_corrcoef
import statistics as st

from Utilities.Visualizers.NoiseprintVisualizer import NoiseprintVisualizer


class NoiseprintTransferabilityExperiment(TransferabilityExperiment):

    def __init__(self, debug_root, data_root_folder, original_dataset: Dataset):
        """
        :param debug_root: str
            debug root folder where to save the logs
        :param data_root_folder: str
            folder containing the data to process
        :param original_dataset: Dataset
            original dataset of the samples to test (used to retrieve the original mask)
        """

        super().__init__(debug_root, data_root_folder, original_dataset, NoiseprintVisualizer())

        self.original_heatmap_original_forgery_f1_results = []
        self.original_heatmap_original_forgery_mcc_results = []

        self.original_forgery_f1_results = []
        self.original_forgery_mcc_results = []
        self.target_forgery_f1_results = []
        self.target_forgery_mcc_results = []

        self.original_forgery_target_threshold_f1_results = []
        self.original_forgery_target_threshold_mcc_results = []

        self.target_forgery_original_threshold_f1_results = []
        self.target_forgery_original_threshold_mcc_results = []

    def _compute_scores(self, sample_name, original_image, attacked_image, original_forgery_mask, target_forgery_mask):

        # Use the best available quality factor
        quality_factor = 101
        try:
            quality_factor = jpeg_quality_of_file(attacked_image.path)
        except:
            pass
        self.visualizer.load_quality(quality_factor)

        original_heatmap, _ = self.visualizer._engine.detect(original_image.one_channel().to_float())

        # foreach metric compute the best threshold
        best_f1_original_heatmap_original_forgery = find_best_theshold(original_heatmap, original_forgery_mask, f1_score)
        best_mcc_original_heatmap_original_forgery = find_best_theshold(original_heatmap, original_forgery_mask, matthews_corrcoef)

        # segment the attacked heatmap using the thresholds computed on the pristine heatmap
        mask_original_heatmap_f1 = self.visualizer._engine.get_mask(original_heatmap, threshold=best_f1_original_heatmap_original_forgery)
        mask_original_heatmap_mcc = self.visualizer._engine.get_mask(original_heatmap, threshold=best_mcc_original_heatmap_original_forgery)

        self.original_heatmap_original_forgery_f1_results.append(f1_score(mask_original_heatmap_f1.flatten(), original_forgery_mask.flatten()))
        self.original_heatmap_original_forgery_mcc_results.append(matthews_corrcoef(mask_original_heatmap_mcc.flatten(), original_forgery_mask.flatten()))

        self.visualizer.base_results(original_image, original_forgery_mask, original_heatmap, mask_original_heatmap_f1,
                                    os.path.join(self.original_results_dir, sample_name))

        self.logger_module.info("Base results (no attack) F1:{:.2} MCC:{:.2}".format(st.mean(self.original_heatmap_original_forgery_f1_results),
                                                                            st.mean(self.original_heatmap_original_forgery_mcc_results)))

        # Compute the heatmap of the image after the attack
        heatmap, mask = self.visualizer._engine.detect(attacked_image.one_channel().to_float())

        # foreach metric compute the best threshold
        best_f1_original_forgery = find_best_theshold(heatmap, original_forgery_mask, f1_score)
        best_mcc_original_forgery = find_best_theshold(heatmap, original_forgery_mask, matthews_corrcoef)

        # segment the attacked heatmap using the thresholds computed on the pristine heatmap
        mask_original_f1 = self.visualizer._engine.get_mask(heatmap, threshold=best_f1_original_forgery)
        mask_original_mcc = self.visualizer._engine.get_mask(heatmap, threshold=best_mcc_original_forgery)

        # foreach metric compute the best threshold
        best_f1_target_forgery = find_best_theshold(heatmap, target_forgery_mask, f1_score)
        best_mcc_target_forgery = find_best_theshold(heatmap, target_forgery_mask, matthews_corrcoef)

        # segment the attacked heatmap using the thresholds computed on the pristine heatmap
        mask_target_f1 = self.visualizer._engine.get_mask(heatmap, threshold=best_f1_target_forgery)
        mask_target_mcc = self.visualizer._engine.get_mask(heatmap, threshold=best_mcc_target_forgery)

        self.original_forgery_f1_results.append(f1_score(mask_original_f1.flatten(), original_forgery_mask.flatten()))
        self.original_forgery_mcc_results.append(
            matthews_corrcoef(mask_original_mcc.flatten(), original_forgery_mask.flatten()))

        self.logger_module.info(
            "original forgery original threshold F1:{:.2} MCC:{:.2}".format(st.mean(self.original_forgery_f1_results),
                                                                            st.mean(self.original_forgery_mcc_results)))

        self.target_forgery_original_threshold_f1_results.append(
            f1_score(mask_original_f1.flatten(), target_forgery_mask.flatten()))
        self.target_forgery_original_threshold_mcc_results.append(
            matthews_corrcoef(mask_original_mcc.flatten(), target_forgery_mask.flatten()))

        self.logger_module.info("target forgery original threshold F1:{:.2} MCC:{:.2}".format(
            st.mean(self.target_forgery_original_threshold_f1_results),
            st.mean(self.target_forgery_original_threshold_mcc_results)))

        self.target_forgery_f1_results.append(f1_score(mask_target_f1.flatten(), target_forgery_mask.flatten()))
        self.target_forgery_mcc_results.append(
            matthews_corrcoef(mask_target_mcc.flatten(), target_forgery_mask.flatten()))

        self.logger_module.info("target forgery target threshold F1:{:.2} MCC:{:.2}".format(st.mean(self.target_forgery_f1_results),
                                                                        st.mean(self.target_forgery_mcc_results)))

        self.original_forgery_target_threshold_f1_results.append(
            f1_score(mask_target_f1.flatten(), original_forgery_mask.flatten()))
        self.original_forgery_target_threshold_mcc_results.append(
            matthews_corrcoef(mask_target_mcc.flatten(), original_forgery_mask.flatten()))

        self.logger_module.info("original forgery target threshold F1:{:.2} MCC:{:.2}".format(
            st.mean(self.original_forgery_target_threshold_f1_results),
            st.mean(self.original_forgery_target_threshold_mcc_results)))

        self.visualizer.complete_pipeline(attacked_image, original_forgery_mask, original_heatmap, target_forgery_mask,
                                          heatmap,
                                          os.path.join(self.attacked_results_dir, sample_name))
