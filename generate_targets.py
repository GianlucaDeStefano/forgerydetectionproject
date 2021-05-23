import argparse
from tqdm import tqdm as tqdm
import os
from Ulitities.Images import get_shape_of_image, plot_noiseprint
from Datasets.datasets import supported_datasets

# Disable warnings from tensorflow due to lots of depracated functions
from noiseprint2 import NoiseprintEngine
from noiseprint2.utility.utilityRead import imread2f

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np

# parse input parameters
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', default="columbia",
                    help='Specify the dataset to use to generate the average noiseprint')
args = parser.parse_args()

QF = args.qualityFactor
params = {"target_shape": (640, 480)}
# get a list of paths to all the authentic images
path_authentic_images = None
foundDatasets = False

for key in supported_datasets:
    if key == args.dataset:
        path_authentic_images = supported_datasets[key]["authetic"](**params)
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

for QF in tqdm(range(51,102)):
    engine = NoiseprintEngine()
    engine.load_quality(QF)

    avg_noiseprint = np.zeros((params["target_shape"][1], params["target_shape"][0]))

    # let's generate the average deep feature representation produced by noiseprintw when no tampering is present
    for img_path in tqdm(path_authentic_images):
        # compute the noiseprintg of the image
        img, mode = imread2f(img_path)

        noiseprint = engine.predict(img)

        # generate the noiseprint
        avg_noiseprint += noiseprint / len(path_authentic_images)

    # visualize the noiseprintw map
    plot_noiseprint(avg_noiseprint, "./avg_noiseprint", toNormalize=True, showMetrics=False)

    # save the average noiseprint
    np.save('/Targets/{}'.format(QF), avg_noiseprint)

