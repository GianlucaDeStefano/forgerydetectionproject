from abc import abstractmethod
from multiprocessing import Pool

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from Detectors.Noiseprint.noiseprintEngine import NoiseprintEngine, normalize_noiseprint, find_best_theshold
from Detectors.Noiseprint.utility.utility import prepare_image_noiseprint
from Utilities.Image.Picture import Picture
from Utilities.Visualizers.BaseVisualizer import BaseVisualizer
import tensorflow as tf
from tqdm.contrib.concurrent import process_map


class InvalidImageShape(Exception):
    def __init__(self, function_name, given_shape):
        super().__init__(
            "The function {} does not support the given image shape: {}".format(function_name, given_shape))


class (BaseVisualizer):

    def __init__(self, quality_level=None):
        super().__init__(NoiseprintEngine(quality_level=quality_level))

    def save_prediction_pipeline(self, path, mask=None):
        cols = 3

        if "mask" in self.metadata or mask is not None:
            mask = mask if mask is not None else self.metadata["mask"]
            cols += 1

        fig, axs = plt.subplots(1, cols, figsize=(cols * 5, 5))

        for ax in axs:
            ax.set_xticks([])
            ax.set_yticks([])

        assert (self.metadata["sample"] is not None)

        axs[0].imshow(np.array(np.rint(self.metadata["sample"]), dtype=np.uint8))

        if "noiseprint" not in self.metadata:
            self._engine.extract_features()

        axs[1].imshow(normalize_noiseprint(self.metadata["noiseprint"]), clim=[0, 1], cmap='gray')

        if "heatmap" not in self.metadata:
            self._engine.generate_heatmap()

        axs[2].imshow(self.metadata["heatmap"],
                      clim=[np.nanmin(self.metadata["heatmap"]), np.nanmax(self.metadata["heatmap"])], cmap='jet')

        if mask is not None:
            axs[3].imshow(mask, clim=[0, 1], cmap="gray")

        plt.savefig(path, bbox_inches='tight')
        plt.close("all")

        del fig, axs
