import argparse
import os

from Attacks import families_of_attacks
from Utilities.Confs.Configs import Configs

configs = Configs("config.yaml","Attacks")


def attack_pipeline(category_number, attack_number):

    configs = Configs("config.yaml")

    if category_number is None:

        c = 0
        for key, supported_attack in families_of_attacks.items():
            print("  {}) {}".format(c, key))
            c = c + 1
        category_number = int(input("Enter category number:"))

    supported_attacks = list(families_of_attacks.values())[category_number]

    if attack_number is None:
        i = 0
        for key, supported_attack in supported_attacks.items():
            print("  {}) {}".format(i, key))
            i = i + 1
        attack_number = int(input("Enter attack number:"))

    attack_class = list(supported_attacks.values())[attack_number]

    attack_arguments,setup_arguments = attack_class.read_arguments(configs["global"]["datasets"]["root"])
    attack_arguments["debug_root"] = configs.debug_root

    # create an instance of the attack
    attack = attack_class(**attack_arguments)

    # load iteration dependent data (images, masks...)
    attack.setup(**setup_arguments)

    # execute the attack
    attack.execute()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", '--type', default=None, type=int,help='Id of the category of the attack to perform')
    parser.add_argument("-m", '--method', default=None, type=int,help='Id of the attack to perform')
    args = parser.parse_known_args()[0]

    attack_pipeline(args.type,args.method)
