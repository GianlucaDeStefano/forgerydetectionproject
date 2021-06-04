import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

from noiseprint2 import normalize_noiseprint
from noiseprint2.utility.utilityRead import imread2f


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


def load_mask(img_path, threshold: float = 0.5):
    """
    Load the groundtruth and be sure the mask is only composd by 0 or ones
    :param img_path: path  of the gt to load
    :param threshold: thereshold to define a positive or negative pixel
    :return:numpy array containing the mask
    """
    mask, mode = imread2f(img_path)
    mask = np.where(mask < threshold, 0, 1)
    return mask


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


def noise_to_3c(noise: np.array):
    noise_3c = np.zeros((noise.shape[0],noise.shape[1],3))
    noise_3c[:, :, 0] = noise / 0.299
    noise_3c[:, :, 1] = noise / 0.587
    noise_3c[:, :, 2] = noise / 0.114

    return noise_3c
