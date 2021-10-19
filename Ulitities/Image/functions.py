import random

import numpy as np
from matplotlib import pyplot as plt

from Ulitities.Image.Picture import Picture


def visuallize_matrix_values(matrix, path):
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




def create_target_forgery_map(shape: tuple):
    assert (len(shape) == 2 and shape[0] > 300 and shape[1] > 300)

    possible_forgery_masks = [
        "./Data/custom/target_forgery_masks/1.png",
        "./Data/custom/target_forgery_masks/2.png",
        "./Data/custom/target_forgery_masks/2.png",
    ]

    forgery = Picture(
        np.where(np.all(Picture(path=str(random.choice(possible_forgery_masks))) == (255, 255, 255), axis=-1), 1, 0))

    forgeries_width = forgery.shape[0]
    forgeries_heigth = forgery.shape[1]

    target_mask = np.zeros(shape)

    x_start = random.randint(0, shape[0] - forgeries_width)
    y_start = random.randint(0, shape[1] - forgeries_heigth)

    target_mask[x_start:x_start + forgeries_width, y_start:y_start + forgeries_heigth] = forgery

    return Picture(target_mask)

