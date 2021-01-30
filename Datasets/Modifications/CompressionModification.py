from abc import abstractmethod
import numpy as np
import cv2


class CompressionModification:

    def __init__(self, value):
        self.value = value

    @abstractmethod
    def apply(self, sample):
        """
        Function to apply this modification to an image
        :param sample as numpy array: sample to which this modification we be applied
        """

        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.value]
        result, encimg = cv2.imencode('.jpg', sample, encode_param)
        return encimg
