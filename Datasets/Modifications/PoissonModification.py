from abc import abstractmethod
import numpy as np


class PoissonModification:


    @abstractmethod
    def apply(self, sample):
        """
        Function to apply this modification to an image
        :param sample as numpy array: sample to which this modification we be applied
        """

        vals = len(np.unique(sample))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(sample * vals) / float(vals)
        return noisy
