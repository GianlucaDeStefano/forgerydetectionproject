import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

from noiseprint2 import normalize_noiseprint


def get_shape_of_image(path: str):
    """
    Return the shape of the image given the path
    :param path: path of the image
    :return: tuple containing its shape
    """
    im = Image.open(path)
    return im.size


def drop_borders(img: np.array, borders_size: tuple = (1, 1, 1, 1)):
    """
    Given a numpy array, remove the borders along x and y for the specified number of pixels
    :param img: numpy array containing the image
    :param borders_size: tumple containing the size of the borders (top,right,bottom,left)
    :return: np.array containing the image without borders
    """

    top_index = borders_size[0]
    right_index = img.shape[0] - borders_size[1]
    bottom_index = img.shape[1] - borders_size[2]
    left_index = borders_size[3]
    return img[top_index:bottom_index, right_index:left_index]


def plot_noiseprint(noiseprint: np.array, saveTo: str = None, toNormalize: bool = True, showMetrics: bool = True):
    """
    Function used to plot the noiseprint into a file
    :param saveTo:
    :param noiseprint: the noiseprint map  to plot
    :param toNormalize: should we normalize the map?
    :return: matplotlib object
    """
    if toNormalize:
        noiseprint = normalize_noiseprint(noiseprint)

    plt.figure()

    if not showMetrics:
        plt.axis('off')

    plt.imshow(noiseprint, clim=[0, 1], cmap='gray')

    plt.savefig(saveTo)
    return plt
