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

    def __init__(self, qf: int = 101):
        super().__init__(NoiseprintEngine(),"Noiseprint")
        self.qf = qf
        self.load_quality(qf)

    def load_quality(self,qf :int):
        assert (50 < qf < 102)
        self.qf = qf
        self._engine = NoiseprintEngine()
        self._engine.load_quality(qf)

    def prediction_pipeline(self, image: Picture, path=None,original_picture = None,note="",omask=None,debug=False,adversarial_noise=None):

        if image.max()> 1:
            image = image.to_float()

        image_one_channel = image
        if len(image_one_channel.shape) > 2 and image_one_channel.shape[2] == 3:
            image_one_channel = image.one_channel()

        if original_picture is not None:
            original_picture = prepare_image_noiseprint(original_picture)

        n_cols = 4

        if original_picture is not None:
            n_cols += 1
        else:
            debug = False

        if debug:
            fig, axs = plt.subplots(2, n_cols, figsize=(n_cols * 5, 5))

            axs0,axs1 = axs[0],axs[1]
        else:
            fig, axs0 = plt.subplots(1, n_cols, figsize=(n_cols * 5, 5))

        noiseprint = self.get_noiseprint(image_one_channel)

        heatmap = self._engine.detect(image_one_channel,)

        #this is the first computation, compute the best f1 threshold initially

        axs0[0].imshow(image)
        axs0[0].set_title('Image')

        axs0[1].imshow(normalize_noiseprint(noiseprint), clim=[0, 1], cmap='gray')
        axs0[1].set_title('Noiseprint')

        axs0[2].imshow(heatmap, clim=[np.nanmin(heatmap), np.nanmax(heatmap)], cmap='jet')
        axs0[2].set_title('Heatmap')

        mask = None

        if omask is not None:
            mask = np.rint(self.get_mask(heatmap,omask))

            axs0[3].imshow(mask, clim=[0, 1], cmap='gray')
            axs0[3].set_title('Mask')

        # remove the x and y ticks
        for ax in axs0:
            ax.set_xticks([])
            ax.set_yticks([])


        if original_picture is not None:
            noise = self.compute_difference(original_picture.one_channel(),image_one_channel)
            axs0[4].imshow(noise, clim=[0, 1], cmap='gray')
            axs0[4].set_title('Difference')

        if debug:
            original_noiseprint = self._engine.predict(original_picture.one_channel())
            axs1[0].imshow(original_picture)
            axs1[0].set_title('Original Image')

            axs1[1].imshow(normalize_noiseprint(original_noiseprint), clim=[0, 1], cmap='gray')
            axs1[1].set_title('Original Noiseprint')

            axs1[2].imshow(omask, clim=[0, 1], cmap='gray')
            axs1[2].set_title('Original Mask')

            noise = self.compute_difference(noiseprint, original_noiseprint,enhance_factor=100)
            axs1[3].imshow(noise, cmap='gray')
            axs1[3].set_title('Noiseprint differences')

            axs1[4].imshow(np.where(np.abs(adversarial_noise) > 0, 1, 0),clim=[0, 1], cmap='gray')
            axs1[4].set_title('Gradient')

            # remove the x and y ticks
            for ax in axs1:
                ax.set_xticks([])
                ax.set_yticks([])

        if note:
            fig.text(0.9, 0.2, note,size=14, horizontalalignment='right', verticalalignment='top')

        if path:
            plt.savefig(path,bbox_inches='tight')
            plt.close()

        return heatmap,mask

    @staticmethod
    def get_mask(heatmap,omask):
        threshold = find_best_theshold(heatmap, omask)
        mask = np.array(heatmap > threshold, int).clip(0,1)
        return mask

    def get_noiseprint(self,image):
        return self._engine.predict(image)

    def predict(self, image: Picture, path = None):

        image_one_channel = image.one_channel().to_float()

        heatmap, mask = self._engine.detect(image_one_channel)

        plt.imshow(mask)

        if path:
            plt.savefig(path,bbox_inches='tight')
            plt.close()


    def base_results(self,image,mask,base_result,result_mask,path):
        fig, axs = plt.subplots(1, 4, figsize=(20, 5))

        for ax in axs:
            ax.set_xticks([])
            ax.set_yticks([])

        axs[0].imshow(image)
        #axs[0].set_title('Image')

        axs[1].imshow(mask, clim=[0, 1], cmap='gray')
        #axs[1].set_title('Ground Truth')

        axs[2].imshow(base_result, clim=[np.nanmin(base_result), np.nanmax(base_result)], cmap='jet')
        #axs[2].set_title('Predicted heatmap')

        axs[3].imshow(result_mask, clim=[0, 1], cmap='gray')
        #axs[3].set_title('Predicted mask')

        plt.savefig(path,bbox_inches='tight')
        plt.close()


    def complete_pipeline(self,image,mask,base_result,target_mask,final_result,path):

        fig, axs = plt.subplots(1, 5, figsize=(25, 5))

        for ax in axs:
            ax.set_xticks([])
            ax.set_yticks([])


        axs[0].imshow(image)
        #axs[0].set_title('Image')

        axs[1].imshow(mask, clim=[0, 1], cmap='gray')
        #axs[1].set_title('Original Forgery')

        axs[2].imshow(base_result, clim=[np.nanmin(base_result), np.nanmax(base_result)], cmap='jet')
        #axs[2].set_title('Original Heatmap')

        axs[3].imshow(target_mask, clim=[0, 1], cmap='gray')
        #axs[3].set_title('Target Forgery')

        axs[4].imshow(final_result, clim=[np.nanmin(final_result), np.nanmax(final_result)], cmap='jet')
        #axs[4].set_title('Final Heatmap')

        plt.savefig(path,bbox_inches='tight')
        plt.close()


    def save_heatmap(self,heatmap,path):
        fig = plt.imshow(heatmap, clim=[np.nanmin(heatmap), np.nanmax(heatmap)], cmap='jet')
        plt.axis('off')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        plt.savefig(path, bbox_inches='tight', pad_inches=0)
