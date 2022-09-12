import logging
import time
from abc import abstractmethod
from multiprocessing import Pool

import numpy as np

from Utilities.Image.Picture import Picture
from Utilities.Logger.Logger import Logger


class DetectorEngine(Logger):
    always_reset_output = True

    def __init__(self, name):
        super().__init__()
        self.name = name
        self.metadata = None

    @abstractmethod
    def destroy(self):
        """
        Destroy the current instance of the detector engine freeing memory and resources
        """
        raise NotImplementedError

    @abstractmethod
    def reset_instance(self):
        """
        Destroy & Reload the detector's instance from memory
        @return:
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self, reset_instance, reset_metadata):
        """
        Reset the metadata of the detector to be used on a new sample

        @param reset_instance: Bool
            A flag indicating if this detector's model should be reimported
        @param reset_metadata:Bool
            A flag indicating if the metadata should be reinitialized before loading the new sample
        """

        if reset_instance:
            self.reset_instance()

        if reset_metadata:
            del self.metadata
            self.metadata = dict()
            print("metadata WIPED")

    def initialize(self, sample_path=None, sample=None, reset_instance=True, reset_metadata=True):
        """
        Initialize the detector to handle a new sample

        @param sample_path: str
            Path of the sample to analyze
        @param sample: numpy.array
            Preloaded sample to use (to be useful sample_path has to be None)
        @param reset_instance: Bool
            A flag indicating if this detector's model should be reimported
        @param reset_metadata:Bool
            A flag indicating if the metadata should be reinitialized before loading the new sample
        """

        self.reset(reset_instance, reset_metadata or sample_path)

        self.metadata["sample_path"] = str(sample_path)
        self.metadata["sample"] = sample

        if not reset_metadata and sample is not None:
            # In case the metadata have not been completely widped,
            # delete still the entries containing noiseprint, heatmap and mask to ensure a fair new run
            if "heatmap" in self.metadata: del self.metadata["heatmap"]
            if "mask" in self.metadata: del self.metadata["mask"]
            if "threshold" in self.metadata: del self.metadata["mask"]

    @abstractmethod
    def extract_features(self):
        """
        Function extracting the necessary features from the sample and saving them into the
        metadatas
        @return metadata: dict
            Dictionary containing all the metadata and results generated during the process
        """
        raise NotImplementedError

    def generate_heatmap(self):
        """
        Function processing the necessary features to create the heatmap of the forged area and save it into the
        metadata
        @return metadata: dict
            Dictionary containing all the metadata and results generated during the process
        """
        raise NotImplementedError

    def generate_mask(self, **kwargs):
        """
        Function processing the necessary features to create the heatmap of the forged area and save it into the
        metadata
        @return metadata: dict
            Dictionary containing all the metadata and results generated during the process
        """
        raise NotImplementedError

    def process(self, sample_path: Picture) -> dict:
        """
        Function populating and returning the output dictionary with the results created by the instanced
        detector
        @param sample_path: Picture
            Path of the sample to analyze
        @return metadata: dict
            Dictionary containing all the metadata and results generated during the process
        """

        if self.metadata and not self.always_reset_output:
            logging.warning("Output dictionary is not empty, its contents will be used during the processing")
        else:
            self.metadata = dict()
            self.initialize(sample_path)

        print("Extracting features..")
        start_time = time.time()

        # extract the features from the loaded sample
        self.extract_features()

        extraction_time = time.time()
        print(f"Completed in {extraction_time - start_time}s")
        print("Clustering...")

        # use the extracted features to compute the heatmap
        self.generate_heatmap()

        print(f"Completed in {time.time() - extraction_time}s")

        return self.metadata

    @staticmethod
    def load_sample(sample_path):
        """
        Method that specifies the correct procedure to load a sample from memory
        @param sample_path: sample to load
        @return: loaded sample
        """
        return NotImplementedError

    @staticmethod
    def transform_sample(pristine_sample):
        """
        Apply the detector-specific transformation necessary to transform a sample into
        the state that the classifeir wants
        @param pristine_sample:
        @return:
        """
        return pristine_sample


def find_optimal_mask(heatmap, gt_mask, metric, test_flipped=True):
    """
    Given the heatmap, returns the mask that satisfies the given metric
    @param heatmap: heatmap to segment
    @param gt_mask: ground truth to compare our result against
    @param metric: metric to maximize
    @param test_flipped: if true flip the heatmap and repeate the threshold selection process on it to verify if a better mask is
    obtainable in this way
    @return: np.array
    """

    assert (heatmap is not None)
    assert (1 >= heatmap.min() >= 0)
    assert (1 >= heatmap.max() >= 0)

    assert (1 >= gt_mask.min() >= 0)
    assert (1 >= gt_mask.max() >= 0)

    target_mask = np.array(gt_mask, dtype=np.uint8).flatten()
    best_mask = None
    best_score = - float('inf')
    best_threshold = None

    n_iterations = 25

    processes = []

    for t in range(0, n_iterations, 1):
        threshold = t / n_iterations
        processes.append((threshold, heatmap, target_mask, metric))

    # test also the flipped heatmap
    if test_flipped:
        print("TESTING FLIPPED MASK")
        for t in range(0, n_iterations, 1):
            threshold = t / n_iterations
            processes.append((threshold, 1 - heatmap, target_mask, metric))

    results = None
    with Pool(6) as pool:
        results = pool.starmap(test_threshold, processes)
        pool.close()

    for score, mask, threshold in results:
        if score > best_score:
            best_score = score
            best_mask = mask
            best_threshold = threshold

    assert (best_mask is not None)

    return best_mask, best_score, best_threshold


def test_threshold(threshold, heatmap, target_mask, metric):
    mask = segment_heatmap(heatmap, threshold)
    pred_mask = np.asarray(mask, dtype=np.uint8)
    score = metric(target_mask, pred_mask.flatten())
    return score, pred_mask, threshold


def segment_heatmap(heatmap, threshold):
    return np.where(heatmap > threshold, 1, 0)
