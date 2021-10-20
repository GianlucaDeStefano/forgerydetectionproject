import os
from pathlib import Path

from cv2 import PSNR

from Attacks import BaseAttack
from Datasets import Dataset
from Detectors.Noiseprint.utility.utility import jpeg_quality_of_file
from Ulitities.Experiments.BaseExperiment import BaseExperiment
from sklearn.metrics import f1_score, matthews_corrcoef

from Ulitities.Visualizers.NoiseprintVisualizer import NoiseprintVisualizer


class NoiseprintExperiment(BaseExperiment):
    
    def __init__(self, attack: BaseAttack, dataset: Dataset, debug_root : str):
        super(NoiseprintExperiment, self).__init__(attack,dataset,debug_root)
        self.visualizer = NoiseprintVisualizer()
    
    def compute_scores(self,original_image, attacked_image, original_mask, target_mask):

        psrn = PSNR(original_image,attacked_image)

        self.PSNRs.append(psrn)

        quality_factor = 101
        try:
            quality_factor = jpeg_quality_of_file(original_image.path)
        except:
            pass
        self.visualizer.load_quality(quality_factor)

        initial_heatmap, _ = self.visualizer._engine.detect(original_image.one_channel().to_float())

        final_heatmap, final_mask_original = self.visualizer._engine.detect(attacked_image.one_channel().to_float(), original_mask)

        final_mask_target = self.visualizer._engine.get_mask(final_heatmap, target_mask)

        output_path = os.path.join(self.debug_foler,Path(original_image.path).stem)
        self.visualizer.complete_pipeline(original_image,original_mask,initial_heatmap,target_mask,final_heatmap,output_path)

        self.f1_original_forgery.append(f1_score(original_mask.flatten(), final_mask_original.flatten()))
        self.mcc_original_forgery.append(matthews_corrcoef(original_mask.flatten(), final_mask_original.flatten()))

        self.f1_target_forgery.append(f1_score(target_mask.flatten(), final_mask_target.flatten()))
        self.mcc_target_forgery.append(matthews_corrcoef(target_mask.flatten(), final_mask_target.flatten()))