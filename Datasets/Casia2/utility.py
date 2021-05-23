import glob
import os
from os import listdir
from pathlib import Path
from PIL import Image

def get_authentic_images(root: str = os.path.dirname(__file__) + "/Data/",target_shape:tuple = None):
    """
    Return a list of paths to all the authentic images
    :param target_shape: The shape the desired images must have
    :param root: root folder of the dataset
    :return: list of Paths
    """
    paths = []

    for filename in os.listdir(os.path.join(root,"Au")):

        if not filename.endswith("jpg"):
            continue

        if target_shape:

            im = Image.open(os.path.join(root,"Au",filename))

            if im.size != target_shape:
                continue

        paths.append(os.path.join(root,"Au",filename))

    return paths


def get_forgered_images(root: str= os.path.dirname(__file__) + "/Data/",target_shape:tuple=None):
    """
    Return a list of paths to all the authentic images
    :param target_shape: The shape the desired images must have
    :param root: root folder of the dataset
    :return: list of Paths
    """
    paths = []

    for filename in os.listdir(os.path.join(root,"Tp")):

        if not filename.endswith("jpg"):
            continue

        if target_shape:

            im = Image.open(os.path.join(root,"Tp",filename))

            if im.size != target_shape:
                continue

        paths.append(os.path.join(root,"Tp",filename))
    return paths


def get_mask_of_image(image_path:str,root: str= os.path.dirname(__file__) + "/Data/"):
    """
    Given the path of an image belonging to the Casia 2 dataset return the path to the mas of said image
    :param image_path: path to the image in question
    :param root:  root folder of the dataset
    :return: path to the mask
    """

    filename = os.path.basename(image_path)
    filename = os.path.splitext(filename)[0]+ "_gt.png"
    return os.path.join(root, "Gt", filename)