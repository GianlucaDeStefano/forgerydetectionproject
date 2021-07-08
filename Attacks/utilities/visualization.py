import numpy as np
from matplotlib import pyplot as plt


def visuallize_array_values(matrix, path):
    fig, ax = plt.subplots(figsize=(5, 5))

    ax.matshow(matrix, cmap=plt.cm.Blues)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            c = matrix[i, j]
            ax.text(j, i, "{:.2f}".format(c), va='center', ha='center', size=7)

    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.savefig(path, bbox_inches='tight', dpi=600)


def visualize_noiseprint_step(image, internal_representation, noise, heatmap, path, should_close=True):
    """
       Visualize the "full noiseprint's pipeline"
       :param image: numpy array containing the image
       :param internal_representation: noiseprint of the image
       :param noise: added adversarial noise
       :param heatmap: numpy array containing the heatmap
       :param path:
       :param should_close:
        :param noise_magnification_factor:
       :return:
       """
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))

    axs[0].imshow(image.clip(0, 1))
    axs[0].set_title('Image')

    axs[1].imshow(internal_representation, clim=[0, 1], cmap='gray')
    axs[1].set_title('Noiseprint')

    axs[2].imshow(noise)
    axs[2].set_title('Adversarial noise')

    axs[3].imshow(heatmap, clim=[np.nanmin(heatmap), np.nanmax(heatmap)], cmap='jet')
    axs[3].set_title('Heatmap')

    # remove the x and y ticks
    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.savefig(path, bbox_inches='tight', dpi=300)

    if should_close:
        plt.close(fig)

    return plt
