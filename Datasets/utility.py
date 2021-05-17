import os
from os import listdir
from pathlib import Path


def get_authentic_images(root: Path):
    """
    Return a list of paths to all the authentic images
    :param root: root folder of the dataset
    :return: list of Paths
    """
    paths = []

    for root, dirs, files in os.walk(root, topdown=False):
        for name in dirs:
            if name.startswith("Au"):
                paths += listdir(os.path.join(root, name))


def get_spliced_images(root: Path):
    """
    Return a list of paths to all the authentic images
    :param root: root folder of the dataset
    :return: list of Paths
    """
    paths = []

    for root, dirs, files in os.walk(root, topdown=False):
        for name in dirs:
            if name.startswith("Sp"):
                paths += listdir(os.path.join(root, name))

