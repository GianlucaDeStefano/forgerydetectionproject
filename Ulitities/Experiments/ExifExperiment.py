import os
from pathlib import Path

import numpy as np
from cv2 import PSNR

from Attacks import BaseAttack
from Datasets import Dataset
from Detectors.Noiseprint.utility.utility import jpeg_quality_of_file
from Ulitities.Experiments.BaseExperiment import BaseExperiment
from sklearn.metrics import f1_score, matthews_corrcoef

from Ulitities.Visualizers.ExifVisualizer import ExifVisualizer
from Ulitities.Visualizers.NoiseprintVisualizer import NoiseprintVisualizer


class ExifExperiment(BaseExperiment):
    
    def __init__(self, attack: BaseAttack, dataset: Dataset, debug_root : str):
        super(ExifExperiment, self).__init__(attack,dataset,debug_root)
        self.visualizer = None
    
    def compute_scores(self,original_image, attacked_image, original_mask, target_mask):

        psrn = PSNR(original_image,attacked_image)

        self.PSNRs.append(psrn)

        self.visualizer = self.attack.detector

        initial_heatmap, initial_mask = self.visualizer._engine.detect(original_image.to_float())

        final_heatmap, final_mask_original = self.visualizer._engine.detect(attacked_image.to_float(), original_mask)

        initial_mask = np.rint(initial_mask)
        final_mask_original = np.rint(final_mask_original)

        output_path = os.path.join(self.debug_foler,Path(original_image.path).stem)

        self.visualizer.complete_pipeline(original_image,original_mask,initial_heatmap,target_mask,final_heatmap,output_path)

        self.f1_original_forgery.append(f1_score(original_mask.flatten(), final_mask_original.flatten()))
        self.mcc_original_forgery.append(matthews_corrcoef(original_mask.flatten(), final_mask_original.flatten()))

        self.f1_target_forgery.append(f1_score(target_mask.flatten(), final_mask_original.flatten()))
        self.mcc_target_forgery.append(matthews_corrcoef(target_mask.flatten(), final_mask_original.flatten()))

