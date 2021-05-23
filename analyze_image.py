import argparse

from noiseprint2 import jpeg_quality_of_file
from noiseprint2.utility.visualization import full_image_visualization

parser = argparse.ArgumentParser()
parser.add_argument('-i','--inputImage', required=True, help='Input image')
parser.add_argument('-g','--truthImage', required=True, help='Truth image')
parser.add_argument('-q','--qualityFactor',default=None,type=int,choices=range(51,102), help='Specify the quality factor of the model to use')
args = parser.parse_args()

# load the image
img_path = args.inputImage
quality = args.qualityFactor

if quality is None:
    try:
        quality = jpeg_quality_of_file(img_path)
    except AttributeError:
        quality = 101

print("Quality level set to:{}".format(quality))

full_image_visualization(args.inputImage,args.truthImage,quality)