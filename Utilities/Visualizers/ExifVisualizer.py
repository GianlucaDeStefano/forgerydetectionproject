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

    def __init__(self):
        super().__init__(ExifEngine(), "EXIF-SC")

    def prediction_pipeline(self, image: Picture, path=None, original_picture=None, omask=None, note="",
                            threshold=None):

        n_cols = 3
        normal_image = image
        if normal_image.max() > 1:
            normal_image = normal_image.to_float()

        if original_picture is not None:
            n_cols += 1

        heatmap, mask = self._engine.detect(image)

        fig, axs = plt.subplots(1, n_cols, figsize=(n_cols * 4, 5))

        axs[0].imshow(normal_image)
        axs[0].set_title('Image')

        axs[1].imshow(heatmap, vmin=0.0, vmax=1.0, cmap='jet')
        axs[1].set_title('Heatmap')

        axs[2].imshow(mask, clim=[0, 1], cmap='gray')
        axs[2].set_title('Mask')

        if original_picture is not None:

            if original_picture.max() > 1:
                original_picture = original_picture.to_float()

            noise = self.compute_difference(original_picture, normal_image)
            axs[3].imshow(noise, clim=[0, 1], cmap='gray')
            axs[3].set_title('Difference')

        if note:
            fig.text(0.9, 0.2, note, size=14, horizontalalignment='right', verticalalignment='top')

        # remove the x and y ticks
        for ax in axs:
            ax.set_xticks([])
            ax.set_yticks([])

        if path:
            plt.savefig(path, bbox_inches='tight')
            plt.close()

        return heatmap, mask

    def base_results(self, image, mask, base_result, result_mask, path):
        fig, axs = plt.subplots(1, 4, figsize=(20, 5))

        for ax in axs:
            ax.set_xticks([])
            ax.set_yticks([])

        axs[0].imshow(image)
        # axs[0].set_title('Image')

        axs[1].imshow(mask, clim=[0, 1], cmap='gray')
        # axs[1].set_title('Ground Truth')

        axs[2].imshow(base_result, vmin=0.0, vmax=1.0, cmap='jet')
        # axs[2].set_title('Predicted Heatmap')

        axs[3].imshow(result_mask, clim=[0, 1], cmap='gray')
        # axs[3].set_title('Predicted mask')

        plt.savefig(path, bbox_inches='tight')
        plt.close()

    def predict(self, image: Picture, path=None):

        heatmap, mask = self._engine.detect(image)

        plt.imshow(mask, cmap='gray')

        if path:
            plt.savefig(path)
            plt.close()

        return heatmap, mask

    def complete_pipeline(self, image, mask, base_result, target_mask, final_heatmap, final_mask, path):

        fig, axs = plt.subplots(1, 5, figsize=(25, 5))

        for ax in axs:
            ax.set_xticks([])
            ax.set_yticks([])

        axs[0].imshow(image)
        # axs[0].set_title('Image')

        axs[1].imshow(mask, clim=[0, 1], cmap='gray')
        # axs[1].set_title('Original Forgery')

        axs[2].imshow(base_result, vmin=0.0, vmax=1.0, cmap='jet')
        # axs[2].set_title('Original Heatmap')

        axs[3].imshow(target_mask, clim=[0, 1], cmap='gray')
        # axs[3].set_title('Target Forgery')

        axs[4].imshow(final_heatmap, vmin=0.0, vmax=1.0, cmap='jet')
        # axs[4].set_title('Final Heatmap')

        plt.savefig(path, bbox_inches='tight')
        plt.close()

    def save_heatmap(self, heatmap, path):
        fig = plt.imshow(heatmap, vmin=0.0, vmax=1.0, cmap='jet')
        plt.axis('off')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        plt.savefig(path, bbox_inches='tight', pad_inches=0)
