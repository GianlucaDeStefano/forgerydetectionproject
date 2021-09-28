import numpy as np
from matplotlib import pyplot as plt

from Detectors.Noiseprint.noiseprintEngine import normalize_noiseprint


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
    noise_3c = np.zeros((noise.shape[0], noise.shape[1], 3))
    noise_3c[:, :, 0] = noise / 0.299
    noise_3c[:, :, 1] = noise / 0.587
    noise_3c[:, :, 2] = noise / 0.114

    return noise_3c
