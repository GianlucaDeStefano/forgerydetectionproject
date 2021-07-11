import os
import time
from math import ceil

import numpy as np
from tqdm import tqdm

from Attacks.utilities.visualization import visualize_noiseprint_step, visuallize_array_values, visualize_noise
from Datasets import find_dataset_of_image
from Detectors.Noiseprint.Noiseprint.noiseprint import NoiseprintEngine, normalize_noiseprint
from Detectors.Noiseprint.Noiseprint.noiseprint_blind import noiseprint_blind_post, genMappFloat
from Detectors.Noiseprint.Noiseprint.utility.utility import jpeg_quality_of_file
from Ulitities.Exceptions.arguments import InvalidArgumentException
from Ulitities.Image.Picture import Picture


def get_data_of_image(image_source):
    # select the first dataset having an image with the corresponding name
    dataset = find_dataset_of_image(DATASETS_ROOT, image_source)

    if not dataset:
        raise InvalidArgumentException("Impossible to find the dataset this image belongs to")

    dataset = dataset(DATASETS_ROOT)

    image_path = dataset.get_image(image_source)
    mask, mask_path = dataset.get_mask_of_image(image_path)

    # load the image as a 3 dimensional numpy array
    image = Picture(image_path)

    # load mask as a picture
    mask = Picture(mask, mask_path)

    return image, mask


def test_noiseprint(image, noise, qf, path):
    engine = NoiseprintEngine()
    engine.load_quality(qf)

    noiseprint = engine.predict(image.one_channel.to_float())

    mapp, valid, range0, range1, imgsize, other = noiseprint_blind_post(noiseprint, image.one_channel.to_float())
    attacked_heatmap = genMappFloat(mapp, valid, range0, range1, imgsize)

    noise = Picture(1 - np.abs(noise * 100))

    visualize_noiseprint_step(image.to_float(), normalize_noiseprint(noiseprint),
                              noise.clip(0, 1), attacked_heatmap, path)


def add_noise_blind(image: Picture, noise: np.ndarray, margin=8):
    """
    Given an image and a noise matrix, this method just applies the noise on the image.
    IF the noise is too small for the image, it is repeated multiple times to fit.
    IF it is bigger it is clamped
    :param image: Image on which to apply the noise
    :param noise: noise matrix containing the noise to apply
    :return: image with the noise applied on it
    """
    image = np.array(image, np.float)

    if image.shape[0] == noise.shape[0] and image.shape[1] == noise.shape[1]:
        image[margin:-margin, margin:-margin] = image[margin:-margin, margin:-margin] + noise[margin:-margin,
                                                                                        margin:-margin]
    else:
        noise_no_margins = noise[margin:-margin, margin:-margin]
        for x0 in range(margin, image.shape[0] - margin, noise_no_margins.shape[0]):
            for y0 in range(margin, image.shape[1] - margin, noise_no_margins.shape[1]):
                x1 = min(x0 + noise_no_margins.shape[0], image.shape[0] - margin)
                y1 = min(y0 + noise_no_margins.shape[1], image.shape[1] - margin)

                image[x0:x1, y0:y1] -= noise_no_margins[0:x1 - x0, 0:y1 - y0]

    return np.rint(image)


def add_targeted_noise(image: Picture, mask: Picture, noise_patch_authentic: np.array, noise_patch_forged: np.array,
                       margin=8):
    """
    This function extract the noise applied on the authentinc and forged areas of a picture, averaging it into 2 patches
    of the defined size, then re-applies the noise to a second image, appling the authentic patch on the authentic section,
    the forged patch on the others.
    Given an image,its mask a noise matrix, the pask of the noise and a patch_size

    :param image: Image on which to apply the noise
    :param mask: boolean mask of the imahe defining if each pixel is authentic or not: 0 authentic, 1 not authentic.
    :param noise_patch_authentic: patch of noise to apply to the authentic area of the image
    :param noise_patch_forged: patch of noise to apply to the forged area of the images
    :return: image with the noise applied
    """

    assert (authentic_noise_patch.shape == forged_noise_patch.shape)
    image = np.array(image,np.float)
    patch_shape = authentic_noise_patch.shape
    print("patch shape:",patch_shape)

    for x0 in range(margin,image.shape[0],patch_shape[0]):
        for y0 in range(margin,image.shape[1],patch_shape[0]):
            x1 = min(x0 + patch_shape[0], image.shape[0] - margin)
            y1 = min(y0 + patch_shape[1], image.shape[1] - margin)

            if mask[x0:x1,y0:y1].sum() == 0:
                image[x0:x1,y0:y1] -= noise_patch_authentic[0:x1-x0,0:y1-y0]
            else:
                image[x0:x1, y0:y1] -= noise_patch_forged[0:x1-x0,0:y1-y0]

    return np.rint(image)


if __name__ == "__main__":
    DEBUG_ROOT = os.path.abspath("./Data/Debug/")
    DATASETS_ROOT = os.path.abspath("./Data/Datasets/")

    noise_source = "./Data/Debug/1625861167.2283487/best-noise.npy"
    noise_source_image = "canong3_canonxt_sub_13.tif"

    image_source = "canonxt_kodakdcs330_sub_01.tif"

    patch_size = (8, 8)

    # CREATE debug FOLDER
    times = time.time()
    debug_folder = os.path.join(DEBUG_ROOT, str(times))
    os.makedirs(debug_folder)

    # load image and mask of the image to attack
    image, mask = get_data_of_image(image_source)

    # load image and mask of the image for which the noise has been generated
    image_noise, mask_noise = get_data_of_image(noise_source_image)

    # load matrix of noise
    noise = Picture(np.load(noise_source))[16:-16, 16:-16]

    visualize_noise(1 - np.abs(noise*100)/256,os.path.join(debug_folder, "noise"))

    # generate authentic patch noise
    authentic_noise_patch = np.zeros(patch_size)

    authentic_patches = noise.get_authentic_patches(mask_noise[16:-16, 16:-16], patch_size, (0, 0, 0, 0), force_shape=True,
                                                    zero_padding=False)
    for authentic_patch in tqdm(authentic_patches):
        authentic_noise_patch += authentic_patch / len(authentic_patches)

    # generate forged noise patch
    forged_noise_patch = np.zeros(patch_size)

    forged_patches = noise.get_forged_patches(mask_noise[16:-16, 16:-16], patch_size, (0, 0, 0, 0), force_shape=True,
                                              zero_padding=False)
    for forged_patch in tqdm(forged_patches):
        forged_noise_patch += forged_patch / len(forged_patches)

    authentic_noise_patch = authentic_noise_patch / np.max(np.abs(authentic_noise_patch))
    forged_noise_patch = forged_noise_patch / np.max(np.abs(forged_noise_patch))

    visuallize_array_values(authentic_noise_patch, os.path.join(debug_folder, "authentic_noise_patch.png"))
    visuallize_array_values(forged_noise_patch, os.path.join(debug_folder, "forged_noise_patch.png"))

    authentic_noise_patch = Picture(authentic_noise_patch).three_channels
    forged_noise_patch = Picture(forged_noise_patch).three_channels

    qf = 0

    if not qf or qf < 51 or qf > 101:
        try:
            qf = jpeg_quality_of_file(image.path)
        except:
            qf = 101

    image_noise_blind = Picture(add_noise_blind(image, noise.three_channels).clip(0, 255))
    image_noise_blind.save(os.path.join(debug_folder, "blind-noise.png"))

    image_noise_blind = Picture(path=os.path.join(debug_folder, "blind-noise.png"))
    test_noiseprint(image_noise_blind, image_noise_blind.one_channel - image.one_channel, qf,
                    os.path.join(debug_folder, "blind-noise-attack"))

    alpha = 2

    image_targeted_noise = Picture(add_targeted_noise(image, mask, authentic_noise_patch*alpha,
                                                      forged_noise_patch*alpha).clip(0, 255))
    image_targeted_noise.save(os.path.join(debug_folder, "targeted-noise.png"))

    test_noiseprint(image, np.zeros(image.shape), qf,
                    os.path.join(debug_folder, "base"))

    image_targeted_noise = Picture(path=os.path.join(debug_folder, "targeted-noise.png"))
    test_noiseprint(image_targeted_noise, image_targeted_noise.one_channel - image.one_channel, qf,
                    os.path.join(debug_folder, "targeted-noise-attack"))
