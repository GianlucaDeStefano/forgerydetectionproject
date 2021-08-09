import argparse
import os
from pathlib import Path

from Datasets import find_dataset_of_image
from Detectors.Noiseprint.utility.utility import jpeg_quality_of_file, jpeg_quality_of_img
from Ulitities.Exceptions.arguments import InvalidArgumentException
from Ulitities.Image.Alterations import available_alterations, choose_alteration
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

    visualizer.prediction_pipeline(image.to_float(), save_path, mask=mask,note=note,original_picture=original_picture)


def test_image_exif(image_path, save_path,note=""):
    """
    Test the given image using Exif-sc
    :param image:
    :param save_path:
    :return:
    """
    visualizer = ExifVisualizer()

    image = Picture(image_path)

    visualizer.prediction_pipeline(image.to_float(), save_path,note=note)



if __name__ == "__main__":

    DATASETS_ROOT = os.path.abspath("Data/Datasets/")

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--images', nargs='+', help='Name/s of the input image/s, or their path', default=[])

    args = parser.parse_args()
    images = args.images

    if images  == []:
        raise Exception("No images have been given")

    debug_folder = create_debug_folder()

    detectors = int(input("""With which detectors should the images be tested? \n
    0) All
    1) Noiseprint
    2) Exif-sc"""))

    assert (0 <= detectors < 3)

    alterations = []
    while True:
        alteration = choose_alteration()
        if alteration is None:
            break
        alterations.append(alteration)
        print("Selected \"{}\"".format(alteration.name))

    for image_path in images:

        image_name = os.path.splitext(os.path.basename(image_path))[0]
        image_debug_folder = debug_folder

        mask = None
        if len(images) > 1:
            image_debug_folder = os.path.join(debug_folder, image_name)
            os.makedirs(image_debug_folder)

        try:
            # find reference to image
            if not Path(image_path).exists():
                # if the path given is not a direct path, search the image in the datasets
                dataset = find_dataset_of_image(DATASETS_ROOT, image_path)
                if not dataset:
                    raise InvalidArgumentException("Impossible to find the dataset this image belongs to")

                dataset = dataset(DATASETS_ROOT)

                image_path = dataset.get_image(image_path)
                mask, _ = dataset.get_mask_of_image(image_path)
        except:
            print("Image: {} not found".format(image_path))
            continue

        image = Picture(image_path)

        for alteration in alterations:
            image = alteration.apply(image)

        if detectors == 0 or detectors == 1:
            # Test Noiseprint
            try:
                qf = jpeg_quality_of_img(image)
            except:
                qf = 101
            test_image_noiseprint(image,mask, os.path.join(image_debug_folder, "noiseprint result"), qf)

        if detectors == 0 or detectors == 1:
            # Test exif_sc
            test_image_exif(image, os.path.join(image_debug_folder, "exif-sc result"))
