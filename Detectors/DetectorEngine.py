import logging
import time
from abc import abstractmethod

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
            self.metadata = dict()

    def initialize(self, sample_path=None, sample=None, reset_instance=False, reset_metadata=True):
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

        self.metadata["sample_path"] = sample_path
        self.metadata["sample"] = sample

    @abstractmethod
    def extract_features(self):
        """
        Function extracting the necessary features from the sample and saving them into the
        output dict object
        @return metadata: dict
            Dictionary containing all the metadata and results generated during the process
        """
        raise NotImplementedError

    def process_features(self):
        """
        Function extracting the necessary features from the sample and saving them into the
        output dict object
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

        # process features
        self.process_features()
        print(f"Completed in {time.time() - extraction_time}s")

        return self.metadata

    @staticmethod
    def transform_sample(pristine_sample):
        """
        Apply the detector-specific transformation necessary to transform a sample into
        the state that the classifeir wants
        @param pristine_sample:
        @return:
        """
        return pristine_sample
