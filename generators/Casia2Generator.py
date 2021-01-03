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

    def __init__(self,dataset:Dataset, batch_size,shuffle=True):
        super().__init__(batch_size,shuffle)
        self.dataset = list(dataset)
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
        return self.indexes[batch_id*self.batch_size:(batch_id+1)*self.batch_size]

    def _generate_x(self, list_IDs_temp):
        """
        Given a list of ids, return the respective input data ready to be used by the model
        :param list_IDs_temp: list of ids
        :return: list of sample data
        """

        #create batch object
        batch = []

        #each batch should have a well defined dimension
        #this means that each element inside the np array should
        #have the same dimansion, therefore, we pad each image in the batch
        #to the dimension of the biggest one

        #compute the target dimension of each image
        samples = [self.dataset[i] for i in list_IDs_temp]
        max_width = max(sample[0].shape[0] for sample in samples)
        max_heigth = max(sample[0].shape[1] for sample in samples)
        target_shape = (max_width,max_heigth,3)

        #foreach sample in the batch
        for i, ID in enumerate(list_IDs_temp):

            #read the sample from the dataset
            X= np.array(self.dataset[ID][0])

            #pad each image
            result = np.zeros(target_shape)
            result[:X.shape[0], :X.shape[1],:X.shape[2]] = X

            # Normalize data
            batch.append((result / 255).astype('float32'))

        return batch

    def _generate_y(self, list_IDs_temp):
        """
        Given a list of ids, return the respective output data ready to be used by the model
        :param list_IDs_temp: list of ids
        :return: list of sample data
        """

        #create batch object
        batch = []

        #each batch should have a well defined dimension
        #this means that each element inside the np array should
        #have the same dimansion, therefore, we pad each image in the batch
        #to the dimension of the biggest one

        #compute the target dimension of each image
        samples = [self.dataset[i] for i in list_IDs_temp]
        max_width = max(sample[1].shape[0] for sample in samples)
        max_heigth = max(sample[1].shape[1] for sample in samples)
        target_shape = (max_width,max_heigth,1)

        #foreach sample in the batch
        for i, ID in enumerate(list_IDs_temp):

            #read the sample from the dataset
            Y= np.array(self.dataset[ID][1])

            #pad each image
            result = np.zeros(target_shape)
            result[:Y.shape[0], :Y.shape[1],:Y.shape[2]] = Y

            # Normalize data
            batch.append((result / 255).astype('float32'))

        return batch