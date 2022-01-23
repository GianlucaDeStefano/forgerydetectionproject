import os
from abc import abstractmethod
import matplotlib
import numpy as np
from matplotlib import pyplot as plt

from Detectors import DetectorEngine
from Utilities.Image.Picture import Picture

matplotlib.use('Agg')


class BaseVisualizer:
    """
    A visualizer class is used to produce consistent visualizations of results across attacks and experiments
    using different Detectors
    """

    def __init__(self, engine: DetectorEngine, name):
        self._engine = engine
        self.name = name

    @abstractmethod
    def predict(self, image: Picture, path=None):
        """
        Function to predict the output map of an image and save it to the specified path
        """
        raise NotImplementedError

    @abstractmethod
    def prediction_pipeline(self, image: Picture, path=None, original_picture=None, note="", mask=None, debug=False,
                            adversarial_noise=None):
        """
        Function to print the output map of an image, together with every intermediate ste[, and save the final image
        it to the specified path
        """
        raise NotImplementedError

    def compute_difference(self, original_image, image, enhance_factor=100):
        return Picture(1 - np.abs(original_image - image) * enhance_factor).clip(0, 1).one_channel()

    def plot_graph(self, data, label_y, label_x="", path=None, min_range_value=1, initial_value=1):
        """
        Generate and save a cartesian graph displaying the given list of datapoints
        :param data: list of datapoints to display
        :param label: label to print on the x axis
        :param path: path to save the imae
        :param display: should the image be opened when created?
        :param min_range_value: minumum number of values to diaplay as index on the x axis
        :param initial_value: base value on the x axis
        :return:
        """
        plt.close()
        plt.plot(data)

        plt.ylabel(label_y)
        plt.xlabel(label_x)

        plt.xticks(range(min_range_value, len(data), max(initial_value, int(len(data) / 10))))
        if path:
            if not os.path.exists(os.path.split(path)[0]):
                os.makedirs(os.path.split(path)[0])
            plt.savefig(path)

        plt.close()

    @abstractmethod
    def complete_pipeline(self, image, mask, base_result, target_mask, final_heatmap, final_mask, path):
        raise NotImplementedError

    @abstractmethod
    def save_heatmap(self,heatmap,path):
        raise NotImplementedError