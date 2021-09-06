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


def attack_pipeline(attack_number):

    if attack_number is None:
        i = 0
        for key, supported_attack in supported_attacks.items():
            print("  {}) {}".format(i, key))
            i = i + 1
        print("  {}) {}".format(i, "All attacks in sequence"))
        attack_number = int(input("Enter attack number:"))

    if attack_number == len(supported_attacks.items()):
        attacks = supported_attacks.values()
    else:
        attacks = list(supported_attacks.values())[attack_number]

    if not isinstance(attacks, list):
        attacks = [attacks]

    # execute each attack sequentially
    for current_attack in attacks:
        kwarg = current_attack.read_arguments(DATASETS_ROOT)

        attack = current_attack(**kwarg)
        attack.execute()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", '--method', default=None, type=int,help='Id of the attack to perform')
    args = parser.parse_known_args()[0]

    attack_pipeline(args.method)
