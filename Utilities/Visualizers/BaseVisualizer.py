import logging
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

    def __init__(self, engine: DetectorEngine):
        """
        @param engine: instance of the engine class to use
        """
        self._engine = engine

    def initialize(self, sample_path=None, sample=None, reset_instance=True, reset_metadata=True):
        """
        Initialize the visualizer to handle a new sample

        @param sample_path: str
            Path of the sample to analyze
        @param sample: numpy.array
            Preloaded sample to use (to be useful sample_path has to be None)
        @param reset_instance: Bool
            A flag indicating if this detector's internal classes should be completely scrapped and reloaded
        @param reset_metadata: Bool
            A flag indicating if this detector's metadata should be reinitialized before loading the new sample.
            (useful if sample_path = None and sample != None)
            Sometimes we need to maintain some hyperparameters from the metadata between detecto-runs
            (e.g: quality factor for Noiseprint)

        """
        self._engine.initialize(sample_path, sample, reset_instance, reset_metadata)

    def process_sample(self, image_path, reset_instance=True):
        self._engine.initialize(image_path, reset_instance=reset_instance, reset_metadata=True)
        return self._engine.process(image_path)

    @abstractmethod
    def predict(self, image: Picture, path=None):
        """
        Function to predict the output map of an image and save it to the specified path
        """
        raise NotImplementedError

    @abstractmethod
    def save_prediction_pipeline(self, path, mask=None):
        """
        Function to print the output map of an image, together with every intermediate ste[, and save the final image
        it to the specified path
        """
        raise NotImplementedError

    @abstractmethod
    def save_heatmap(self, path):
        """
        Save the predicted heatmap as an image
        @param path: path where to save the heatmap
        @return: None
        """
        raise NotImplementedError

    def get_mask(self):
        """
        Computes the mask of the forged area (if necessary) and then returns it
        @return: np.array
        """
        return self.metadata["mask"]

    @property
    def metadata(self):
        return self._engine.metadata

    def reset_metadata(self):
        del self._engine.metadata
        self._engine.metadata = dict()


def compute_difference(self, original_image, image, enhance_factor=100):
    return Picture(1 - np.abs(original_image - image) * enhance_factor).clip(0, 1).one_channel()
