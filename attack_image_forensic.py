import argparse
import os
import time

import numpy as np
from matplotlib import pyplot as plt

from LOTS.forensic_attack import attack_noiseprint_model
from noiseprint2 import normalize_noiseprint, jpeg_quality_of_file
from noiseprint2.utility.utilityRead import imread2f
from noiseprint2.utility.visualization import image_noiseprint_heatmap_visualization_2

parser = argparse.ArgumentParser()
parser.add_argument('-i','--inputImage', required=True, help='Input image')
parser.add_argument('-g','--groundTruth', required=False,default=None, help='Input image')
parser.add_argument('-q','--qualityFactor',default=None,type=int,choices=range(51,102), help='Specify the quality factor of the model to use')
parser.add_argument('-s','--steps', default=50,type=int, help='Cap to the LOTS steps per image')
parser.add_argument('-d','--debug', action="store_true", help='Activate debug mode, with extensive outputs and logging')
args = parser.parse_args()


# load the groundtruth as a 2d array
img, mode = imread2f(args.inputImage,channel=3)

#if no quality factor has been specified, get the one that fits best the image
quality = args.qualityFactor
if quality is None:
    try:
        quality = int(jpeg_quality_of_file(args.inputImage))
    except AttributeError:
        quality = 101

print("Using quality level:{}".format(quality))

# create the debug folder
start_time = time.time()
debug_folder = os.path.join("./Data/Debug/", str(start_time))
os.makedirs(debug_folder)

# add adversarial perturbation generated using the LOTS method
res = attack_noiseprint_model(args.inputImage,args.groundTruth, quality, args.steps,debug_folder)

if not res:
    print("The attack has not been succesful")
else:
    attacked_image, original_noiseprint, attacked_noiseprint, \
    original_heatmap, attacked_heatmap = res

    image_noiseprint_heatmap_visualization_2(img,attacked_image,original_noiseprint,attacked_noiseprint,original_heatmap
                                             ,attacked_heatmap,os.path.join(debug_folder,"final comparison"))



