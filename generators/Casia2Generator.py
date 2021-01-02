import math
import random
from abc import ABC

import numpy as np
from tensorflow.python.data import Dataset

from generators.DataGenerator import DataGenerator


class Casia2Generator(DataGenerator, ABC):
    """
    Class that given a dataset, prepares and splits the data to serve to the model
    """

    def __init__(self,dataset:Dataset, batch_size,
                 to_fit, shuffle=True):
        super().__init__(batch_size,to_fit,shuffle)
        self.dataset = dataset
        self.indexes = []
        self.on_epoch_end()


    def __len__(self):
        """
        Compute the length of the set using the elements in the given dataset divided by the number of elements
        in ach batch
        :return: integer
        """
        return math.ceil(len(self.dataset)/self.batch_size)

    def on_epoch_end(self):
        """
        After the execution of an epoch, set up the generator to execute another one.
        :return: None
        """

        #generate the indexes of the samples
        if not self.indexes:
            self.indexes = np.arange(len(self.dataset))

        #shuffle the samples orders
        if self.shuffle:
            random.shuffle(self.indexes)


    def _generate_indexes(self, batch_id):
        """
        Given the id of a batch, generate a list of samples indexes to use in that batch
        :param batch_id:
        :return:
        """
        return self.dataset[batch_id*self.batch_size,(batch_id+1)*self.batch_size]

    def _generate_x(self, list_IDs_temp):
        """
        Given a list of ids, return the respective input data ready to be used by the model
        :param list_IDs_temp: list of ids
        :return: list of sample data
        """

        #create batch object
        X = np.empty((self.batch_size, 3))

        #foreach sample in the batch
        for i, ID in enumerate(list_IDs_temp):

            #read the sample from the dataset
            X[i,] = self.dataset[ID]["image"]

            # Normalize data
            X = (X / 255).astype('float32')

        #return a 4 dimensional tensor
        return X[:, :, :, np.newaxis]

    def _generate_y(self, list_IDs_temp):
        """
        Given a list of ids, return the respective output data ready to be used by the model
        :param list_IDs_temp: list of ids
        :return: list of sample data
        """

        # create batch object
        Y = np.empty((self.batch_size, 3))

        #foreach sample in the batch
        for i, ID in enumerate(list_IDs_temp):

            # read the sample from the dataset
            Y[i,] = self.dataset[ID]["mask"]

            # Normalize data
            X = (X / 255).astype('float32')

        return X[:, :, :, np.newaxis]