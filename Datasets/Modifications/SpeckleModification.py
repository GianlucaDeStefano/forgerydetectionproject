from abc import abstractmethod
import numpy as np


class SpeckleModification:


    @abstractmethod
    def apply(self, sample):
        """
        Function to apply this modification to an image
        :param sample as numpy array: sample to which this modification we be applied
        """

        row, col, ch = sample.shape
        gauss = np.random.randn(row, col, ch)
        gauss = gauss.reshape(row, col, ch)
        noisy = sample + sample * gauss
        return noisy
