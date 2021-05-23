import argparse

from noiseprint2 import jpeg_quality_of_file
from noiseprint2.utility.visualization import full_image_visualization

parser = argparse.ArgumentParser()
parser.add_argument('-i','--inputImage', required=True, help='Input image')
parser.add_argument('-g','--truthImage', required=True, help='Truth image')
parser.add_argument('-q','--qualityFactor',default=None,type=int,choices=range(51,102), help='Specify the quality '
                                                                                             'factor of the model to '
                                                                                             'use, i it is not '
                                                                                             'specified, it will be '
                                                                                             'computed from the file')
args = parser.parse_args()

# load the image
img_path = args.inputImage
quality = args.qualityFactor

#if not quality factor has been specified
if quality is None:
    try:
        #get it from the file
        quality = jpeg_quality_of_file(img_path)
    except AttributeError:
        #if it is not possible, use the default one = 101
        quality = 101

print("Quality level set to:{}".format(quality))
full_image_visualization(args.inputImage,args.truthImage,quality)