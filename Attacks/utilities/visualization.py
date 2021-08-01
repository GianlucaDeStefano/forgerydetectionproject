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




def visualize_noise(noise,path, should_close=True):
        fig, axs = plt.subplots(1, 1, figsize=(5, 5))

        axs.imshow(noise, clim=[0, 1], cmap='gray')
        axs.set_title('Adversarial noise')

        # remove the x and y ticks
        axs.set_xticks([])
        axs.set_yticks([])

        plt.savefig(path, bbox_inches='tight', dpi=300)

        if should_close:
            plt.close(fig)

        return plt
