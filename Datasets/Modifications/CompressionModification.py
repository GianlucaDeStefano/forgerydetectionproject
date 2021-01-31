from abc import abstractmethod
import numpy as np
import cv2
import io
import imageio

class CompressionModification:

    def __init__(self, quality):
        self.quality = quality

    @abstractmethod
    def apply(self, sample):
        """
        Function to apply this modification to an image
        :param sample as numpy array: sample to which this modification we be applied
        """

        buf = io.BytesIO()
        imageio.imwrite(buf, sample, format='jpg', quality=self.quality)
        s = buf.getbuffer()
        return imageio.imread(s, format='jpg')

