import argparse
import os
import warnings

from Attacks import supported_attacks
from Datasets import supported_datasets, find_dataset_of_image
from Datasets.Dataset import mask_2_binary
from Detectors.Noiseprint.Noiseprint.utility.utilityRead import imread2f
from Ulitities.Exceptions.arguments import InvalidArgumentException
from Ulitities.Image.Picture import Picture

DEBUG_ROOT = os.path.abspath("./Data/Debug/")
DATASETS_ROOT = os.path.abspath("./Data/Datasets/")

def attack_image(image_path,mask_path = None,dataset=None,attack_type = None):

    # if both the mask and the dataset are provided, say that it is useless and then use the mask directly
    if mask_path:
        if dataset:
            warnings.warn("Both mask and dataset have been provided, the dataset parameter will be ignored")

        mask, mode = imread2f(mask_path)
        mask = mask_2_binary(mask)
    else:

        if dataset:
            # use the specified dataset
            dataset = supported_datasets[dataset](DATASETS_ROOT)
        else:
            # select the first dataset having an image with the corresponding name
            dataset = find_dataset_of_image(DATASETS_ROOT, image_path)(DATASETS_ROOT)
            if not dataset:
                raise InvalidArgumentException("Impossible to find the dataset this image belongs to")

        image_path = dataset.get_image(image_path)
        mask, mask_path = dataset.get_mask_of_image(image_path)

    # load the image as a 3 dimensional numpy array
    image = Picture(image_path)

    # load mask as a picture
    mask = Picture(mask, mask_path)

    # assert image and mask have compatible shapes
    assert (image.shape[0] == mask.shape[0] and image.shape[1] == mask.shape[1])
    assert (len(mask.shape) == 2 or mask.shape[2] == 1)

    attacks = None
    if not attack_type:
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
        attacks = supported_attacks[attack_type]

    if not isinstance(attacks, list):
        attacks = [attacks]

    # execute each attack sequentially
    for current_attack in attacks:
        attack = current_attack(image, mask)
        attack.execute()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", '--image', required=True, help='Name of the input image, or its path')
    parser.add_argument("-m", '--mask', default=None, help='Path to the binary mask of the image')
    parser.add_argument("-d", '--dataset', default=None, choices=supported_datasets.keys(),
                        help='Dataset to which the image belongs')
    parser.add_argument("-a", '--attackType', choices=supported_attacks.keys(), help='Attack to perform')
    args = parser.parse_args()

    image_path = args.image
    mask_path = args.mask
    dataset = args.dataset
    attack_type = args.attack_type

    attack_image(image_path,mask_path,dataset,attack_type)