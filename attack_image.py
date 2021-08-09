import argparse
import os
import warnings

from Attacks import supported_attacks
from Datasets import supported_datasets, find_dataset_of_image
from Datasets.Dataset import mask_2_binary
from Detectors.Noiseprint.Noiseprint.utility.utilityRead import imread2f
from Ulitities.Exceptions.arguments import InvalidArgumentException
from Ulitities.Image.Picture import Picture

DEBUG_ROOT = os.path.abspath("Data/Debug/")
DATASETS_ROOT = os.path.abspath("Data/Datasets/")


def attack_pipeline():

    i = 1
    for key, supported_attack in supported_attacks.items():
        print("  {}) {}".format(i, key))
        i = i + 1
    print("  {}) {}".format(i, "All attacks in sequence"))
    attack_number = int(input("Enter attack number:"))

    if attack_number == i:
        attacks = supported_attacks.values()
    else:
        attacks = list(supported_attacks.values())[attack_number - 1]

    if not isinstance(attacks, list):
        attacks = [attacks]

    # execute each attack sequentially
    for current_attack in attacks:

        kwarg = current_attack.read_arguments(DATASETS_ROOT)

        attack = current_attack(**kwarg)
        attack.execute()


if __name__ == "__main__":
    attack_pipeline()
