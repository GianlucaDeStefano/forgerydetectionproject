import math
import random
from abc import ABC, abstractmethod

import numpy as np
from tensorflow.python.data import Dataset

from Geneartors.DataGenerator import DataGenerator


class Casia2Generator(DataGenerator, ABC):
    """
    Class that given a dataset, prepares and splits the data to serve to the model
    """

    def __init__(self, dataset: Dataset, x_data:list, batch_size, shuffle=True):
        super().__init__(dataset,batch_size,False,shuffle)
        self.x_samples = x_data
        self.indexes = []
        self.on_epoch_end()

    def _generate_x(self, sample_id):
        """
        Given a sample id, read the sample input from the
        :param sample_id: id of the sample we have to generte
        :return: input of the sample
        """

        # read the sample from the dataset
        sample = self.dataset[sample_id]

        sample_data = []

        for sample_key in self.x_samples:

            if sample_key.lower() == "rgb":
                sample_data.append(self.get_sample_rgb(sample))
            elif sample_key.lower() == "noiseprint":
                sample_data.append(self.get_sample_noiseprint(sample))
            elif sample_key.lower() == "srm":
                sample_data.append(self.get_sample_srm(sample))
            else:
                raise Exception("Invalid sample key")

        if len(sample_data) == 1:
            return sample_data[0]
        return sample_data

    def _generate_y(self, sample_id):
        """
        Given a sample id, read the sample input from the
        :param sample_id: id of the sample we have to generte
        :return: input of the sample
        """

        # read the sample from the dataset
        Y = self.dataset[sample_id]["tampered"]

        return Y


    def get_sample_rgb(self,sample):
        """
        Given a sample object returns the RGB image contained in the sample normalized and ready for training
        """
        # read the sample from the dataset
        image = np.array(sample["image"])

        # Normalize data
        image = (image / 255).astype('float32')

        return image

    def get_sample_noiseprint(self,sample):
        """
        Given a sample object returns the noiseprint image contained in the sample normalized and ready for training
        """

        # read the sample from the dataset
        image = np.array(sample["noiseprint"])

        return image

    def get_sample_srm(self, sample):
        """
        Given a sample object returns the SRM image contained in the sample normalized and ready for training
        """
        # read the sample from the dataset
        image = np.array(sample["SRM"])

        # Normalize data
        image = (image / 255).astype('float32')

        return image