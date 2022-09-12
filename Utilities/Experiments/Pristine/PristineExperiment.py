import os.path
import traceback
from os.path import basename
from statistics import mean

import cv2
import numpy as np
from Utilities.Experiments.BaseExperiment import BaseExperiment
from Utilities.Image.Picture import Picture
from Utilities.Visualizers.BaseVisualizer import BaseVisualizer
from Utilities.Visualizers.NoiseprintVisualizer import NoiseprintVisualizer


class PristineExperiment(BaseExperiment):

    def __init__(self, visualizer: BaseVisualizer, debug_root, samples, dataset):
        super().__init__(visualizer, debug_root, samples, dataset)

    def process_sample(self, sample_path, gt_mask):

        try:
            self.visualizer.process_sample(sample_path)

            if self.visualizer.metadata["sample"].shape[:2] != gt_mask.shape[:2]:
                # if mask and image have different sizes reshape the  mask (it happens in the DSo dataset)
                gt_mask = np.rint(cv2.resize(np.array(gt_mask, dtype=np.float32), dsize=(
                self.visualizer.metadata["sample"].shape[1], self.visualizer.metadata["sample"].shape[0])))
                gt_mask = np.array(gt_mask, np.uint8)

            for metric_key, metric_func in self.metrics.items():
                optimal_mask,_ = self.visualizer._engine.generate_mask(gt_mask=gt_mask, metric=metric_func)
                self.pristine_scores[metric_key].append(metric_func(gt_mask.flatten(), optimal_mask.flatten()))

            self.visualizer.save_prediction_pipeline(os.path.join(self.debug_root, basename(sample_path)))
        except Exception as e:
            self.logger_module.warning("EXCEPTION")
            self.logger_module.warning(traceback.format_exc())
            self.logger_module.warning(e)

    def log_result(self):
        """
        Overridable method to customize the statistics that will be generated after each sample is processed
        @return:
        """
        self.logger_module.info("Sample's results: " + "".join(
            [f"{metric_key}:{scores[-1]:.3f}" for metric_key, scores in self.pristine_scores.items()]))
        self.logger_module.info(" Current means: " + "".join(
            [f"{metric_key}:{mean(scores):.3f}" for metric_key, scores in self.pristine_scores.items()]))

    def log_result_end(self):
        """
        Overridable method to customize the statistics that will be generated after the end of the experimnet
        @return: None
        """
        self.logger_module.info("FINAL STATISTICS:")
        for metric_key, scores in self.pristine_scores.items():
            self.logger_module.info(f" {metric_key}:{mean(scores):.5f}")