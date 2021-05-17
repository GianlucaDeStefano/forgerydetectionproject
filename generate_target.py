from matplotlib import pyplot as plt
from tqdm import tqdm as tqdm
import os
#Disable warnings from tensorflow due to lots of depracated functions (old version of tf)
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

img_shape = (128,128)

#get a list of paths to all the authentic images
path_authentic_images = get_authentic_images_columbia()

print(len(path_authentic_images))

#define a variable to hold the average mask
avg_noiseprint = np.zeros(img_shape)

QF = 51
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
np.save('average_authentic_noiseprint.npy', avg_noiseprint)