from abc import abstractmethod

import numpy as np
from matplotlib import pyplot as plt

from Detectors.Noiseprint.noiseprintEngine import NoiseprintEngine, normalize_noiseprint, find_best_theshold
from Ulitities.Image.Picture import Picture
from Ulitities.Visualizers.BaseVisualizer import BaseVisualizer
import tensorflow as tf

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.5
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

class InvalidImageShape(Exception):
    def __init__(self, function_name, given_shape):
        super().__init__(
            "The function {} does not support the given image shape: {}".format(function_name, given_shape))


class NoiseprintVisualizer(BaseVisualizer):

    def __init__(self, qf: int = 101):
        super().__init__(NoiseprintEngine(),"Noiseprint")
        self.qf = qf
        self.load_quality(qf)

    def load_quality(self,qf :int):
        assert (50 < qf < 102)
        self.qf = qf
        self._engine = NoiseprintEngine()
        self._engine.load_quality(qf)

    def prediction_pipeline(self, image: Picture, path=None,original_picture = None,note="",mask=None):


        n_cols = 4

        image_one_channel = image.one_channel()

        if original_picture is not None:
            n_cols += 1

        noiseprint = self._engine.predict(image_one_channel)

        heatmap = self._engine.detect(image_one_channel)

        #this is the first computation, compute the best f1 threshold initially
        threshold = find_best_theshold(heatmap, mask)

        mask = np.array(heatmap > threshold, int).clip(0,1)

        fig, axs = plt.subplots(1, n_cols,figsize=(n_cols*5, 5))

        axs[0].imshow(image)
        axs[0].set_title('Image')

        axs[1].imshow(normalize_noiseprint(noiseprint), clim=[0, 1], cmap='gray')
        axs[1].set_title('Noiseprint')

        axs[2].imshow(heatmap, clim=[np.nanmin(heatmap), np.nanmax(heatmap)], cmap='jet')
        axs[2].set_title('Heatmap')

        axs[3].imshow(mask, clim=[0, 1], cmap='gray')
        axs[3].set_title('Mask')

        if original_picture is not None:
            noise = self.compute_difference(original_picture.one_channel(),image_one_channel)
            axs[4].imshow(noise, clim=[0, 1], cmap='gray')
            axs[4].set_title('Adversarial noise')

        if note:
            fig.text(0.9, 0.2, note,size=14, horizontalalignment='right', verticalalignment='top')

        # remove the x and y ticks
        for ax in axs:
            ax.set_xticks([])
            ax.set_yticks([])

        if path:
            plt.savefig(path,bbox_inches='tight')
            plt.close()
        else:
            return plt

    def predict(self, image: Picture, path = None):

        image_one_channel = image.one_channel().to_float()

        heatmap, mask = self._engine.detect(image_one_channel)

        plt.imshow(mask)

        if path:
            plt.savefig(path)
            plt.close()
        else:
            return plt


