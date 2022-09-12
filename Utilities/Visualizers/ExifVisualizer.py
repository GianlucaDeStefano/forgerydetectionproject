from abc import abstractmethod

import numpy as np
from matplotlib import pyplot as plt

from Utilities.Image.Picture import Picture
from Utilities.Visualizers.BaseVisualizer import BaseVisualizer
import tensorflow as tf
from Detectors.Exif.ExifEngine import ExifEngine

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 1
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)


class InvalidImageShape(Exception):
    def __init__(self, function_name, given_shape):
        super().__init__(
            "The function {} does not support the given image shape: {}".format(function_name, given_shape))


class ExifVisualizer(BaseVisualizer):

    def __init__(self, dense: bool = False):
        super().__init__(ExifEngine(dense))

    def save_prediction_pipeline(self, path, mask=None):

        cols = 2

        if mask is not None:
            cols += 1

        fig, axs = plt.subplots(1, cols, figsize=(cols * 5, 5))

        for ax in axs:
            ax.set_xticks([])
            ax.set_yticks([])

        axs[0].imshow(self.metadata["sample"])

        if "features" not in self.metadata:
            self._engine.extract_features()

        if "heatmap" not in self.metadata:
            self._engine.generate_heatmap()

        axs[1].imshow(self.metadata["heatmap"], cmap='jet', vmin=0.0, vmax=1.0)

        if mask is not None:
            axs[2].imshow(mask, clim=[0, 1], cmap="gray")

        plt.savefig(path, bbox_inches='tight')
        plt.close("all")

        del fig, axs