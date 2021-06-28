import argparse
import os
import warnings

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from Datasets import supported_datasets
from Datasets.Dataset import mask_2_binary, ImageNotFoundException
from Detectors.Noiseprint.Noiseprint.utility.utilityRead import imread2f
from Ulitities.Exceptions.arguments import InvalidArgumentException
from Ulitities.Image.Picture import Picture

DEBUG_ROOT = "./Data/Debug/"

parser = argparse.ArgumentParser()
parser.add_argument("-i", '--image', required=True, help='Name of the input image, or its path')
parser.add_argument("-m", '--mask', default=None, help='Path to the binary mask of the image')
parser.add_argument("-d", '--dataset', default=None, choices=supported_datasets.keys(),
                    help='Dataset to which the image belongs')
args = parser.parse_args()

image_path = args.image
mask_path = args.mask

# if both the mask and the dataset are provided, say that it is useless and then use the mask directly
dataset = None
if args.mask:
    if args.dataset:
        warnings.warn("Both mask and dataset have been provided, the dataset parameter will be ignored")

    mask, mode = imread2f(args.mask)
    mask = mask_2_binary(mask)
else:

    if args.dataset:
        # use the specified dataset
        dataset = supported_datasets[args.dataset]
    else:
        # select the first dataset having an image with the corresponding name
        for key, candidate_dataset in supported_datasets.items():
            try:
                print("{},{}".format(image_path, key))
                if candidate_dataset().get_image(image_path):
                    dataset = candidate_dataset()
                    print("Dataset found: {}".format(key))
                    break
            except ImageNotFoundException as e:
                continue
        if not dataset:
            raise InvalidArgumentException("Impossible to find the dataset this image belongs to")

    image_path = dataset.get_image(args.image)
    mask, mask_path = dataset.get_mask_of_image(image_path)

IMAGE = Picture(image_path)

print("SHAPE:{}".format(IMAGE.shape))

PATCHES = IMAGE.one_channel().divide_in_patches((8, 8), (32, 32, 32, 32), force_shape=False)

reconstruction = np.zeros(IMAGE.one_channel().shape)

imgplot = plt.imshow(PATCHES[1]/255)
plt.show()

for patch in tqdm(PATCHES):
    reconstruction = patch.add_to_image(reconstruction)

imgplot = plt.imshow(reconstruction/255)
plt.show()
