import argparse
import os
from pathlib import Path

from Datasets import find_dataset_of_image, get_image_and_mask
from Detectors.Noiseprint.utility.utility import jpeg_quality_of_file, jpeg_quality_of_img
from Ulitities.Exceptions.arguments import InvalidArgumentException
from Ulitities.Image.Picture import Picture
from Ulitities.Visualizers.ExifVisualizer import ExifVisualizer
from Ulitities.Visualizers.NoiseprintVisualizer import NoiseprintVisualizer
from Ulitities.io.folders import create_debug_folder


def test_image_noiseprint(image, mask, save_path, qf,note="",original_picture = None):
    """
    Test the given image using noiseprint
    :param image:
    :param save_path:
    :return:
    """
    visualizer = NoiseprintVisualizer(qf)

    visualizer.prediction_pipeline(image.to_float(), save_path, omask=mask,note=note,original_picture=original_picture)


def test_image_exif(image, save_path,note=""):
    """
    Test the given image using Exif-sc
    :param image:
    :param save_path:
    :return:
    """
    visualizer = ExifVisualizer()

    visualizer.prediction_pipeline(image.to_float(), save_path,note=note)

DEBUG_ROOT = os.path.abspath("Data/Debug/")
DATASETS_ROOT = os.path.abspath("Data/Datasets/")

if __name__ == "__main__":

    debug_folder = create_debug_folder(DEBUG_ROOT)

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image',type=str,required=True, help='Name of the input image, or its path')
    parser.add_argument('-m', '--mask',type=str,default=None, help='PAth to the mask of the image')
    args = parser.parse_args()
    image_path = args.image
    mask_path = args.mask

    if Path(image_path).exists() and not mask_path:
        mask_path = str(input("Input the path to the mask of the image"))

    image, mask = get_image_and_mask(DATASETS_ROOT, image_path, mask_path)

    test_image_noiseprint(image,mask,os.path.join(debug_folder,"noiseprint"),100)
    test_image_exif(image,os.path.join(debug_folder,"exif"))


