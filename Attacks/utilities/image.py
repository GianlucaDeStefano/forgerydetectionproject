import numpy as np
from math import log10, sqrt

def three_2_one_channel(image: np.array) -> np.array:
    """
    Function to convert from 3 to one channel, by doing a weighted sum
    :param image: 3 channel image in the form of a numpy array (N x M x 3)
    :return: numpy array (N X M)
    """
    assert (image.shape[2] == 3)

    return 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]


def one_2_three_channel(image: np.array) -> np.array:
    """
    Given a 1 channel image, divide it into 3 channels by performin the inverse operations of
    three_2_one_channel
    :param image: 1 channel image (N x M x 1)
    :return: numpy array (N x M x 3)
    """

    image_3c = np.zeros((image.shape[0], image.shape[1], 3))
    image_3c[:, :, 0] = image[:, :] * 0.299
    image_3c[:, :, 1] = image[:, :] * 0.587
    image_3c[:, :, 2] = image[:, :] * 0.114

    assert (image_3c.shape[2] == 3)

    return image_3c

def normalize(noise:np.array):
    return (noise - np.min(noise))/np.ptp(noise)

def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

def normalize_noiseprint_no_margins(noiseprint):
        v_min = np.min(noiseprint)
        v_max = np.max(noiseprint)
        return ((noiseprint - v_min) / (v_max - v_min)).clip(0, 1)
