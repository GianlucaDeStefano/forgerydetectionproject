from math import ceil

import numpy as np


def divide_in_patches(image: np.array, patch_shape: tuple, force_shape=False):
    """
    Function to divide an image into patches
    :param image: image to divide
    :param patch_shape: target shape of each patch
    :param force_shape: strictly produce only patches of the given shape
    :return: list of patches
    """

    assert (len(patch_shape) == 2)

    patches = []

    for column in range(ceil(image.shape[0] / patch_shape[0])):

        for row in range(ceil(image.shape[1] / patch_shape[1])):

            x_index = column * patch_shape[0]
            y_index = row * patch_shape[1]

            x_index_f = x_index + patch_shape[0]
            y_index_f = y_index + patch_shape[1]

            if x_index_f > image.shape[0]:
                if force_shape:
                    continue
                else:
                    x_index_f = image.shape[0]

            if y_index_f > image.shape[1]:
                if force_shape:
                    continue
                else:
                    y_index_f = image.shape[1]

            patch = image[x_index:x_index_f, y_index:y_index_f]

            if force_shape:
                assert (patch.shape == patch_shape)

            patches.append((x_index, y_index, patch))

    return patches


def get_authentic_patches(img: np.array, mask: np.array, patch_shape: tuple, force_shape=False):
    """
    Given an image and its relative mask ,return a list of batches containing elements that are not highlighted in
    the mask
    :param img: input image
    :param mask: input mask
    :param patch_shape: target shape of each patch
    :param force_shape: strictly produce only patches of the given shape
    :return: list of valid patches
    """
    assert (img.shape == mask.shape)

    # divide iimage and mask into patches
    img_patches = divide_in_patches(img, patch_shape, force_shape)
    mask_patches = divide_in_patches(mask, patch_shape, force_shape)
    assert (len(img_patches) == len(mask_patches))

    # disgard patches containing masked data
    authentic_patches = []
    for i in range(len(img_patches)):
        if mask_patches[i][2].sum() == 0:
            authentic_patches.append(img_patches[i])

    return authentic_patches


def get_forged_patches(img: np.array, mask: np.array, patch_shape: tuple, force_shape=False):
    """
    Given an image and its relative mask ,return a list of batches containing elements that are not highlighted in
    the mask
    :param img: input image
    :param mask: input mask
    :param patch_shape: target shape of each patch
    :param force_shape: strictly produce only patches of the given shape
    :return: list of valid patches
    """
    assert (img.shape == mask.shape)

    # divide iimage and mask into patches
    img_patches = divide_in_patches(img, patch_shape, force_shape)
    mask_patches = divide_in_patches(mask, patch_shape, force_shape)
    assert (len(img_patches) == len(mask_patches))

    # disgard patches not containing masked data
    authentic_patches = []
    for i in range(len(img_patches)):
        if mask_patches[i][2].sum() != 0:
            authentic_patches.append(img_patches[i])

    return authentic_patches


def scale_patch(patch, target_shape, x_index, y_index, mode="constant", constant=0):
    """
    Add a 0-padding around a patch's gradient to scale it to the desired shape while positioning the gradient in the
    correct position
    :param patch: patch to scale
    :param target_shape: shape of the final matrix
    :param x_index: starting position of the gradient in the matrix on the x axis
    :param y_index: starting position of the gradient in the matrix on the y axis
    :return:
    """
    left_padding = x_index
    right_padding = target_shape[0] - x_index - patch.shape[0]
    top_padding = y_index
    bottom_padding = target_shape[1] - y_index - patch.shape[1]

    gradient = np.pad(patch, ((left_padding, right_padding), (top_padding, bottom_padding)), mode=mode,
                      constant_values=constant)

    assert (gradient.shape == target_shape)

    return gradient
