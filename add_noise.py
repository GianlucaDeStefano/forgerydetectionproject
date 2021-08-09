import argparse
import os
import time
from pathlib import Path

import numpy as np
from cv2 import PSNR

from Datasets import find_dataset_of_image
from Detectors.Noiseprint.utility.utility import jpeg_quality_of_file
from Ulitities.Exceptions.arguments import InvalidArgumentException
from Ulitities.Image.Picture import Picture
from Ulitities.Visualizers.ExifVisualizer import ExifVisualizer
from Ulitities.Visualizers.NoiseprintVisualizer import NoiseprintVisualizer
from Ulitities.io.folders import create_debug_folder


def add_noise_blind(image, noise: np.ndarray, margin=8):
    """
    Given an image and a noise matrix, this method just applies the noise on the image.
    IF the noise is too small for the image, it is repeated multiple times to fit.
    IF it is bigger it is clamped
    :param image: Image on which to apply the noise
    :param noise: noise matrix containing the noise to apply
    :return: image with the noise applied on it
    """

    image_no_margin = np.array(image, np.float)
    noise_no_margin = np.array(noise, np.float)
    if margin > 0:
        noise_no_margin = noise_no_margin[margin:-margin, margin:-margin]

    if image_no_margin.shape[0] == noise_no_margin.shape[0] and image_no_margin.shape[1] == noise_no_margin.shape[1]:
        image_no_margin = image_no_margin - noise_no_margin
    else:
        for x0 in range(margin, image_no_margin.shape[0]-margin, noise_no_margin.shape[0]):
            for y0 in range(margin, image_no_margin.shape[1]-margin, noise_no_margin.shape[1]):
                x1 = min(x0 + noise_no_margin.shape[0], image_no_margin.shape[0]-margin)
                y1 = min(y0 + noise_no_margin.shape[1], image_no_margin.shape[1]-margin)

                image_no_margin[x0:x1, y0:y1] -= noise_no_margin[0:x1 - x0, 0:y1 - y0]

    return Picture(np.array(np.rint(image_no_margin), np.int))


if __name__ == "__main__":
    DEBUG_ROOT = os.path.abspath("Data/Debug/")
    DATASETS_ROOT = os.path.abspath("Data/Datasets/")

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--images', nargs='+', help='Name/s of the input image/s, or their path', default=[])
    parser.add_argument("-n", '--noise', required=True, help='Path to the noise to apply')
    parser.add_argument("-m", '--margin', default=0, type=int, help='Margin to emove from the noise sides')
    args = parser.parse_args()

    visualizers = [NoiseprintVisualizer,ExifVisualizer]

    images = args.images
    noise_source = args.noise

    margin = args.margin

    start_time = time.time()

    # get the maximum strength to use
    attack_strength_cap = 2

    for visualizer_class in visualizers:
        # create debug folder
        print(visualizer_class)
        # create debug folder
        debug_folder = os.path.join(create_debug_folder(),visualizer_class().name)
        os.makedirs(debug_folder)

        for image_path in images:

            image_name = os.path.splitext(os.path.basename(image_path))[0]
            image_debug_folder = debug_folder

            mask =None
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
                    mask,_ = dataset.get_mask_of_image(image_path)
            except:
                print("Image: {} not found".format(image_path))
            # load image from memory
            image = Picture(image_path)

            # create log file
            with open(os.path.join(image_debug_folder, "logs.txt"), "w") as file:

                file.write("Source file: {}\n".format(image_path))
                file.write("Noise file: {}\n".format(noise_source))

                # find the quality factor to decide which noiseprint model to use
                try:
                    qf = jpeg_quality_of_file(image_path)
                except:
                    qf = 101

                # print load nosie
                noise = np.array(np.load(noise_source))

                # load matrix of noise
                noise = Picture(noise).three_channels()

                psnr_values = []

                visualizer = visualizer_class()

                for alpha in range(0, attack_strength_cap, 1):

                    # apply noise
                    attacked_image = add_noise_blind(image, noise*alpha, margin)

                    psnr = PSNR(image, attacked_image)
                    if alpha > 0:
                        psnr_values.append(psnr)
                    else:
                        psnr = 00

                    text = "alpha:{}, psnr:{:.2f}".format(alpha, psnr)
                    file.write(text+"\n")

                    # print comparison
                    visualizer.prediction_pipeline(attacked_image.to_float(), os.path.join(image_debug_folder, "{}".format(alpha)),image.to_float(),text,mask)

                visualizer.plot_graph(psnr_values, "PSNR", "alpha", path=os.path.join(image_debug_folder, "psnr"))

                file.close()
