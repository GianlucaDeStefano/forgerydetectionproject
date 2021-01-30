from abc import abstractmethod
import numpy as np
from scipy.ndimage.filters import gaussian_filter
class BlurModification:

    def __init__(self,sigma):
        self.sigma = sigma

    @abstractmethod
    def apply(self,sample):
        """
        Function to apply this modification to an image
        :param sample as numpy array: sample to which this modification we be applied
        """

        assert(type(self.sigma) == int)
        return gaussian_filter(sample, sigma=self.sigma)