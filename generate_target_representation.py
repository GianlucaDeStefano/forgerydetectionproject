import argparse
import warnings

from Attacks import supported_attacks
from Datasets import supported_datasets
from Datasets.Dataset import mask_2_binary, ImageNotFoundException
from Detectors.Noiseprint.Noiseprint.utility.utilityRead import imread2f
from Ulitities.Exceptions.arguments import InvalidArgumentException
from Ulitities.Image.Picture import Picture

"""
    Given an image and its mask this script can be used to generate its target representation 
"""

patch_size = (16, 16)

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

# load the image as a 3 dimensional numpy array
image = Picture(path=image_path)

# load mask as a picture
mask = Picture(mask)

# assert image and mask have compatible shapes
assert (image.shape[0] == mask.shape[0] and image.shape[1] == mask.shape[1])
assert (len(mask.shape) == 2 or mask.shape[2] == 1)

attacks = None
if not args.attackType:
    print("\nNo specific attack has been selected, select one:")
    i = 1
    for key, supported_attack in supported_attacks.items():
        print("  {}) {}".format(i, key))
        i = i + 1
    print("  {}) {}".format(i, "All attacks in sequence"))
    attack_number = int(input("Enter attack number:"))

    if attack_number == i:
        attacks = list(supported_attacks.values())
    else:
        attacks = list(supported_attacks.values())[attack_number - 1]

else:
    # read the attack to perform and instantiate its class
    attacks = supported_attacks[args.attackType]

if not isinstance(attacks, list):
    attacks = [attacks]

# execute each attack sequentially
for current_attack in attacks:
    attack = current_attack(image, mask, image_path, mask_path)
    attack.execute()
