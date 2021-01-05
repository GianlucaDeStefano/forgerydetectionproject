import math
import random
from abc import ABC

import numpy as np
from tensorflow.python.data import Dataset

from Geneartors.DataGenerator import DataGenerator


class Casia2Generator(DataGenerator, ABC):
    """
    Class that given a dataset, prepares and splits the data to serve to the model
    """

    def __init__(self, dataset: Dataset, batch_size, shuffle=True):
        super().__init__(batch_size, shuffle)
        self.dataset = list(dataset)
        self.indexes = []
        self.on_epoch_end()

    def __len__(self):
        """
        Compute the length of the set using the elements in the given dataset divided by the number of elements
        in ach batch
        :return: integer
        """
        return math.ceil(len(self.dataset) / self.batch_size)

    def on_epoch_end(self):
        """
        After the execution of an epoch, set up the generator to execute another one.
        :return: None
        """

        # generate the indexes of the samples
        self.indexes = np.arange(len(self.dataset))

        # shuffle the samples orders
        if self.shuffle:
            random.shuffle(self.indexes)

    def _generate_indexes(self, batch_id):
        """
        Given the id of a batch, generate a list of samples indexes to use in that batch
        :param batch_id:
        :return:
        """
        return self.indexes[batch_id * self.batch_size:(batch_id + 1) * self.batch_size]

    def _generate_x(self, sample_id):
        """
        Given a sample id, read the sample input from the
        :param sample_id: id of the sample we have to generte
        :return: input of the sample
        """

        # read the sample from the dataset
        X = np.array(self.dataset[sample_id][0])

        # Normalize data
        X = (X / 255).astype('float32')

        return X

    def _generate_y(self, sample_id):
        """
        Given a sample id, read the sample input from the
        :param sample_id: id of the sample we have to generte
        :return: input of the sample
        """

        # read the sample from the dataset
        Y = np.array(self.dataset[sample_id][1])

        # Normalize data
        Y = (Y / 255).astype('float32')

        return Y

    def _apply_transformations(self, x, y):
        """
        Given a set of inputs and outpust representing a batch, apply some sort of data augmentation.
        For now we just implement flipping, since it' doesn't require to alter the image in any way

        :param x: batch of inputs elements
        :param y: batch of output elements
        :return: X,Y batches
        """

        # be sure the 2 batches are of equal size
        assert (len(x) == len(y))

        # foreach element in the batch
        for i in range(len(x)):

            # if random is > 0.5 flip horizzontally the input and the corresponding output
            if random.random() > 0.5:
                x[i] = np.fliplr(x[i])
                y[i] = np.fliplr(y[i])

            # if random is > 0.5 flip vertically the input and the corresponding output
            if random.random() > 0.5:
                x[i] = np.flipud(x[i])
                y[i] = np.flipud(y[i])

        return x,y