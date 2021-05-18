import argparse

from matplotlib import pyplot as plt
from tqdm import tqdm as tqdm
import os
#Disable warnings from tensorflow due to lots of depracated functions (old version of tf)
from Ulitities.Images import get_shape_of_image
from noiseprint.utilities import genMappUint8
from noiseprint.utilities import normalize_noiseprint

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from Datasets.Columbia.utility import get_authentic_images as get_authentic_images_columbia
import numpy as np
from PIL import Image
# Path to the root folder of the columbia dataset
from noiseprint.noiseprint import genNoiseprint

#shape (width x height) of the images we are going to use to produce the average noiseprintw map
from noiseprint.utility.utilityRead import imread2f, jpeg_qtableinv

#define a dict holding as key the name of the supported dataset and as
#value the function used to gather its authentic data
supported_datasets = {
    "columbia":get_authentic_images_columbia
}

#parse input parameters
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-o','--output', default="avgNoiseprint", help='Specify the name to give at the output .npy file')
parser.add_argument('-d','--dataset', default="columbia", help='Specify the dataset to use to generate the average noiseprint')
parser.add_argument('-q','--qualityFactor',default=101,type=int,choices=range(51,102), help='Specify the quality factor of the model to use')
args = parser.parse_args()

QF = args.qualityFactor

#get a list of paths to all the authentic images
path_authentic_images = None
foundDatasets = False

for key in supported_datasets:
    if key == args.dataset:
        path_authentic_images = supported_datasets[key]()
        foundDatasets = True
        break

#if no dataset with the given name is found raise an exeption
if not foundDatasets:
    raise Exception("No dataset has been found, please choose a dataset from the list: \n{}".format(" \n".join(supported_datasets.keys())))

#if no data has been loaded raise an exception
if path_authentic_images == []:
    raise Exception("No authentic images have been found, have you downloaded the datasets?")

#Supposing all images are of the same shape
#define a variable to hold the average mask
img_shape = get_shape_of_image(path_authentic_images[0])
avg_noiseprint = np.zeros(img_shape)

#let's generate the average deep feature representation produced by noiseprintw when no tampering is present
for img_path in tqdm(path_authentic_images):

    #compute the noiseprintg of the image
    img, mode = imread2f(img_path)

    #generate the noiseprint
    avg_noiseprint += genNoiseprint(img, QF)/len(path_authentic_images)


#visualize the noiseprintw map
normalize_noiseprint(avg_noiseprint)
vmin = np.min(avg_noiseprint[34:-34, 34:-34])
vmax = np.max(avg_noiseprint[34:-34, 34:-34])

plt.figure()
plt.imshow(avg_noiseprint, clim=[vmin,vmax], cmap='gray')
plt.savefig("./avg_noiseprint")
plt.show()

#save the average noiseprintw
np.save('{}.npy'.format(args.output), avg_noiseprint)