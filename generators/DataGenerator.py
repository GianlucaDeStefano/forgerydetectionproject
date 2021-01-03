import math
from abc import ABC, abstractmethod
import numpy as np

import tensorflow as tf


class DataGenerator(ABC, tf.compat.v2.keras.utils.Sequence):
    """
    Class that given a dataset, prepares and splits the data to serve to the model
    """

    def __init__(self, batch_size, shuffle=True):
        """
        :param batch_size: how many elements will every batch contain?
        :param shuffle: should the elements be shuffled at each epoch?
        """
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.batch_index = 0

    def __next__(self):
        """
            Used to iterate over the batches until the end of the epoch
        :return: batch of seld.batch_size samples
        """

        # Get one batch of data
        data = self.__getitem__(self.batch_index)
        # Batch index
        self.batch_index += 1

        # If we have processed the entire dataset then
        if self.batch_index >= self.__len__():
            self.on_epoch_end()
            self.batch_index = 0

        return data

    def __getitem__(self, index):
        """
        Given a batch index, returns the set of elements inthat batch
        :param index:
        :return:
        """

        # Find list of IDs
        list_IDs_temp = self._generate_indexes(index)

        X = np.array(self._generate_x(list_IDs_temp))

        Y = np.array(self._generate_y(list_IDs_temp))

        return X.astype('float32') ,Y.astype('float32')

    @abstractmethod
    def __len__(self):
        """
        Compute the amount of batches in each epoch
        :return: the amount of batches in each epoch
        """
        # Return the number of batches of the dataset
        return NotImplementedError

    @abstractmethod
    def on_epoch_end(self):
        """
        After the execution of an epoch, set up the generator to execute another one
        :return: None
        """
        raise NotImplementedError

    @abstractmethod
    def _generate_indexes(self, batch_id):
        """
        Given the id of a batch, generate a list of samples indexes to use in that batch
        :param batch_id:
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def _generate_x(self, list_IDs_temp):
        """
        Given a list of ids, return the respective input data ready to be used by the model
        :param list_IDs_temp: list of ids
        :return: list of sample data
        """
        raise NotImplementedError

    @abstractmethod
    def _generate_y(self, list_IDs_temp):
        """
        Given a list of ids, return the respective output data ready to be used by the model
        :param list_IDs_temp: list of ids
        :return: list of sample data
        """
        raise NotImplementedError
