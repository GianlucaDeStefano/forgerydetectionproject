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
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset',
                    help='Specify the dataset to use to generate the average noiseprint')
args = parser.parse_args()

# if true it will use the authentic path of forgered images to
# generate the target
use_forged = True

save_images = True

params = {"target_shape": (640, 480)}
# get a list of paths to all the authentic images
path_authentic_images = None
path_forged_images = None
findMask = None

foundDatasets = False

for key in supported_datasets:
    if key == args.dataset:
        path_authentic_images = supported_datasets[key]["authetic"](**params)
        path_forged_images = supported_datasets[key]["forged"](**params)

        if "mask" in supported_datasets[key]:
            findMask = supported_datasets[key]["mask"]

        foundDatasets = True
        break

# if no dataset with the given name is found raise an exeption
if not foundDatasets:
    raise Exception("No dataset has been found, please choose a dataset from the list: \n{}".format(
        " \n".join(supported_datasets.keys())))

# if no data has been loaded raise an exception
if path_authentic_images == []:
    raise Exception("No authentic images have been found, have you downloaded the datasets?")

# Supposing all images are of the same shape
# define a variable to hold the average mask
img_shape = get_shape_of_image(path_authentic_images[0])

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

            noiseprint = engine.predict(img)

            # generate the noiseprint
            avg_noiseprint += ((1 - mask) * noiseprint) / len(path_authentic_images)

            plt.figure()

            plt.imshow((1 - mask) * img, clim=[0, 1], cmap='gray')

            plt.show()

    plot_noiseprint(avg_noiseprint,"./Targets/Images/{}.png".format(QF))

    # save the average noiseprint
    np.save('./Targets/{}'.format(QF), save_images)
