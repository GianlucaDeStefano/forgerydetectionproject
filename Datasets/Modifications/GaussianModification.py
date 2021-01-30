from abc import abstractmethod
import numpy as np


class GaussianModification:


    @abstractmethod
    def apply(self, sample):
        """
        Function to apply this modification to an image
        :param sample as numpy array: sample to which this modification we be applied
        """

        row, col, ch = sample.shape
        mean = 0
        var = 0.1
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = sample + gauss
        return noisy
