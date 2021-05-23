import glob
import os
from os import listdir
from pathlib import Path


def get_authentic_images(root: str = os.path.dirname(__file__) + "/Data/",target_shape=None):
    """
    Return a list of paths to all the authentic images
    :param root: root folder of the dataset
    :return: list of Paths
    """
    paths = []

    for root, dirs, files in os.walk(root, topdown=False):
        for name in dirs:
            if name.startswith("Au"):
                paths += Path(os.path.join(root, name)).glob("*.bmp")
    return paths


def get_forgered_images(root: str= os.path.dirname(__file__) + "/Data/",target_shape=None):
    """
    Return a list of paths to all the authentic images
    :param root: root folder of the dataset
    :return: list of Paths
    """
    paths = []

    for root, dirs, files in os.walk(root, topdown=False):
        for name in dirs:
            if name.startswith("Sp"):
                paths += Path(os.path.join(root, name)).glob("*.bmp")

    return paths
