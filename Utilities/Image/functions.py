import glob
import os
import random

import cv2
import numpy as np
from matplotlib import pyplot as plt

from Utilities.Image.Picture import Picture


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


def create_random_nonoverlapping_mask(original_mask, possible_forgery_masks_folder="Data/custom/target_forgery_masks/"):
    mask_shape = original_mask.shape

    possible_forgery_masks = list(glob.glob(os.path.join(possible_forgery_masks_folder, "*.png")))

    mask = None
    i = 100

    while mask is None and i > 0:
        mask = create_target_forgery_map(mask_shape, possible_forgery_masks, 5)

        overlap = mask * original_mask

        if overlap.max() != 0:
            mask = None

        i -= 1

    assert mask is not None

    return mask

def create_target_forgery_map(image_shape: tuple, candidate_forgeries: list, forgery_factor=5):
    assert (candidate_forgeries != [])

    forgery = Picture(
        np.where(np.all(Picture(path=str(random.choice(candidate_forgeries))) == (255, 255, 255), axis=-1), 1, 0))

    forgery_target_side = min(image_shape[0], image_shape[1]) // forgery_factor

    scaled_forgery = cv2.resize(forgery, dsize=(forgery_target_side, forgery_target_side),
                                interpolation=cv2.INTER_NEAREST_EXACT)

    target_mask = np.zeros(tuple(image_shape[:2]))

    x_start = random.randint(0, image_shape[0] - forgery_target_side)
    y_start = random.randint(0, image_shape[1] - forgery_target_side)

    target_mask[x_start:x_start + forgery_target_side, y_start:y_start + forgery_target_side] = scaled_forgery

    return Picture(target_mask)
