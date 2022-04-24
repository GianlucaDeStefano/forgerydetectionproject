from abc import abstractmethod

import numpy as np
from matplotlib import pyplot as plt

from Detectors.Noiseprint.noiseprintEngine import NoiseprintEngine, normalize_noiseprint, find_best_theshold
from Detectors.Noiseprint.utility.utility import prepare_image_noiseprint
from Utilities.Image.Picture import Picture
from Utilities.Visualizers.BaseVisualizer import BaseVisualizer
import tensorflow as tf


class InvalidImageShape(Exception):
    def __init__(self, function_name, given_shape):
        super().__init__(
            "The function {} does not support the given image shape: {}".format(function_name, given_shape))


class NoiseprintVisualizer(BaseVisualizer):

    def __init__(self):
        super().__init__(NoiseprintEngine())

    def save_prediction_pipeline(self, path):
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        for ax in axs:
            ax.set_xticks([])
            ax.set_yticks([])

        assert (self.metadata["sample"] is not None)

        axs[0].imshow(np.array(np.rint(self.metadata["sample"]),dtype=np.uint8))

        if "noiseprint" not in self.metadata:
            self._engine.extract_features()

        axs[1].imshow(normalize_noiseprint(self.metadata["noiseprint"]), clim=[0, 1], cmap='gray')

        if "heatmap" not in self.metadata:
            self._engine.process_features()

        axs[2].imshow(self.metadata["heatmap"],
                      clim=[np.nanmin(self.metadata["heatmap"]), np.nanmax(self.metadata["heatmap"])], cmap='jet')

        plt.savefig(path, bbox_inches='tight')
        plt.show()
