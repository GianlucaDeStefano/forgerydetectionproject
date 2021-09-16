from abc import abstractmethod

import numpy as np
from matplotlib import pyplot as plt

from Ulitities.Image.Picture import Picture
from Ulitities.Visualizers.BaseVisualizer import BaseVisualizer
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

    def __init__(self,):

        super().__init__(ExifEngine(),"EXIF-SC")



    def prediction_pipeline(self, image: Picture, path=None, original_picture=None,omask=None, note="",threshold=None):

        n_cols = 3

        if original_picture is not None:
            n_cols += 1

        heatmap, mask = self._engine.detect(image)

        fig, axs = plt.subplots(1, n_cols, figsize=(n_cols * 4, 5))

        axs[0].imshow(image)
        axs[0].set_title('Image')

        axs[1].imshow(heatmap, clim=[0, 1], cmap='jet')
        axs[1].set_title('Heatmap')

        axs[2].imshow(mask, clim=[0, 1], cmap='gray')
        axs[2].set_title('Mask')

        if original_picture is not None:
            noise = self.compute_difference(original_picture.one_channel(), image.one_channel())
            axs[3].imshow(noise, clim=[0, 1], cmap='gray')
            axs[3].set_title('Adversarial noise')

        if note:
            fig.text(0.9, 0.2, note, size=14, horizontalalignment='right', verticalalignment='top')

        # remove the x and y ticks
        for ax in axs:
            ax.set_xticks([])
            ax.set_yticks([])

        if path:
            plt.savefig(path, bbox_inches='tight')
            plt.close()
        else:
            return plt

    def predict(self, image: Picture, path=None):

        image_one_channel = image.one_channel().to_float()

        heatmap, mask = self._engine.detect(image_one_channel)

        plt.imshow(mask)

        if path:
            plt.savefig(path)
            plt.close()
        else:
            return plt
