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

    def initialize(self, sample_path=None, sample=None, reset_instance=False,reset_metadata = True):
        """
        Initialize the visualizer to handle a new sample

        @param sample_path: str
            Path of the sample to analyze
        @param sample: numpy.array
            Preloaded sample to use (to be useful sample_path has to be None)
        @param reset_instance: Bool
            A flag indicating if this detector's metadata should be reinitialized before loading the new sample.
            (useful if sample_path = None and sample != None)
            Sometimes we need to maintain some hyperparameters from the metadata between detecto-runs
            (e.g: quality factor for Noiseprint)
        @param reset_metadata: Bool

        """
        self._engine.initialize(sample_path, sample, reset_instance,reset_metadata)

    def process_sample(self, image_path):
        self.reset()
        self._engine.initialize(image_path)
        self._engine.process(image_path)

    def reset(self):
        """
        Reset the state of the visualizer to process a new sample
        """
        logging.log(logging.INFO, f"Resetting Detector : {self._engine.name}")
        self._engine.reset()
        logging.log(logging.INFO, f"Primal detector instance: {self._engine.name} successfully restored")

    @abstractmethod
    def predict(self, image: Picture, path=None):
        """
        Function to predict the output map of an image and save it to the specified path
        """
        raise NotImplementedError

    @abstractmethod
    def save_prediction_pipeline(self, path):
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

    @property
    def metadata(self):
        return self._engine.metadata

def compute_difference(self, original_image, image, enhance_factor=100):
    return Picture(1 - np.abs(original_image - image) * enhance_factor).clip(0, 1).one_channel()
