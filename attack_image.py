import argparse

import numpy as np
from matplotlib import pyplot as plt

from LOTS.attack import attack_noiseprint_model
from noiseprint2 import normalize_noiseprint, jpeg_quality_of_file
from noiseprint2.utility.utilityRead import imread2f

parser = argparse.ArgumentParser()
parser.add_argument('-i','--inputImage', required=True, help='Input image')
parser.add_argument('-g','--groundTruth', required=True, help='Input image')
parser.add_argument('-o','--output', required=True, help='Output path')
parser.add_argument('-t','--target', default=None, help='Load target representation as a .npy file')
parser.add_argument('-q','--qualityFactor',default=None,type=int,choices=range(51,102), help='Specify the quality factor of the model to use')
parser.add_argument('-s','--steps', default=100,type=int, help='Cap to the LOTS steps per image')
parser.add_argument('-d','--debug', action="store_true", help='Activate debug mode, with extensive outputs and logging')
args = parser.parse_args()

# load the image to attack as a 2d array
img, mode = imread2f(args.inputImage)

# load the groundtruth as a 2d array
gt, mode = imread2f(args.groundTruth)

#if no quality factor has been specified, get the one that fits best the image
quality = args.qualityFactor
if quality is None:
    try:
        quality = int(jpeg_quality_of_file(args.inputImage))
    except AttributeError:
        quality = 101

print("Using quality level:{}".format(quality))

#load target representation
target_path = args.target
#check if a custom target representation has been given
if not target_path:
    #if not get the default representation based on the quality level
    target_path = "./Targets/{}.npy".format(quality)

#load the target representation
target_representation = np.load(target_path)

# add adversarial perturbation generated using the LOTS method
res = attack_noiseprint_model(img,gt, target_representation, quality, args.steps,args.debug)

if not res:
    print("The attack has not been succesful")
else:
    attacked_image, original_noiseprint, attacked_noiseprint, \
    original_heatmap, attacked_heatmap = res

    fig, axs = plt.subplots(2, 3)
    axs[0, 0].imshow(img)
    axs[0, 0].set_title('Original image')
    axs[1, 0].imshow(attacked_image)
    axs[1, 0].set_title('Attacked image')
    axs[0, 1].imshow(normalize_noiseprint(original_noiseprint))
    axs[0, 1].set_title('Original noiseprint')
    axs[1, 1].imshow(normalize_noiseprint(attacked_noiseprint))
    axs[1, 1].set_title('Attacked noiseprint')
    axs[0, 2].imshow(original_heatmap)
    axs[0, 2].set_title('Original heatmap')
    axs[1, 2].imshow(attacked_heatmap)
    axs[1, 2].set_title('Attacked heatmap')

    plt.savefig(args.output)
    plt.show()

