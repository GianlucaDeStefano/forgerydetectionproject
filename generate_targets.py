import argparse

from matplotlib import pyplot as plt
from tqdm import tqdm as tqdm
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from Ulitities.Images import get_shape_of_image, plot_noiseprint
from Datasets.datasets import supported_datasets

# Disable warnings from tensorflow due to lots of depracated functions
from noiseprint2 import NoiseprintEngine
from noiseprint2.utility.utilityRead import imread2f

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np

# parse input parameters
# if true it will use the authentic path of forgered images to
# generate the target

parser = argparse.ArgumentParser()
parser.add_argument('-o','--output', default="./Targets", help='Output path')
args = parser.parse_args()

use_forged = True

save_images = True
params = {"target_shape": (640, 480)}

dataset_name = "casia2"

#get functions to load data of the desired dataset
path_authentic_images = supported_datasets[dataset_name]["authetic"](**params)
path_forged_images = supported_datasets[dataset_name]["forged"](**params)
findMask = supported_datasets[dataset_name]["mask"]


# if no data has been loaded raise an exception
if path_authentic_images == []:
    raise Exception("No authentic images have been found, have you downloaded the datasets?")

# Supposing all images are of the same shape
# define a variable to hold the average mask
img_shape = get_shape_of_image(path_authentic_images[0])

#create the directory in which to save tarets and corresponding images
os.makedirs(os.path.join(args.output,"Images"))

print("Genearting target representations")

for QF in tqdm(range(51, 102)):
    engine = NoiseprintEngine()
    engine.load_quality(QF)

    avg_noiseprint = np.zeros((params["target_shape"][1], params["target_shape"][0]))

    # let's generate the average deep feature representation produced by noiseprintw when no tampering is present
    for img_path in path_authentic_images:
        # compute the noiseprintg of the image
        img, mode = imread2f(img_path)

        noiseprint = engine.predict(img)

        # generate the noiseprint
        avg_noiseprint += noiseprint / len(path_authentic_images)

    if use_forged:

        for img_path in path_forged_images:
            # compute the noiseprintg of the image
            img, mode = imread2f(img_path)

            mask, mode = imread2f(findMask(img_path))

            assert (img.shape == mask.shape)

            # generate the noiseprint
            noiseprint = engine.predict(img)

            #average the authentic noiseprint
            avg_noiseprint += ((1 - mask) * noiseprint) / len(path_authentic_images)

    plot_noiseprint(avg_noiseprint,os.path.join(args.output,"Images/{}.png".format(QF)))

    # save the average noiseprint
    np.save(os.path.join(args.output,str(QF)), save_images)
