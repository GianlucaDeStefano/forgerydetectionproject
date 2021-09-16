import os
from os.path import basename

import cv2
import numpy as np
from PIL import Image
from sklearn.metrics import matthews_corrcoef, f1_score
from tqdm import tqdm

from Datasets import find_dataset_of_image
from Detectors.Exif.ExifEngine import ExifEngine
from Ulitities.Exceptions.arguments import InvalidArgumentException
from Ulitities.Image.Picture import Picture
from Ulitities.Visualizers.ExifVisualizer import ExifVisualizer
from Ulitities.io.folders import create_debug_folder
import tensorflow as tf

DEBUG_ROOT = os.path.abspath("Data/Debug/")
DATASETS_ROOT = os.path.abspath("Data/Datasets/")

#image_name = "canong3_canonxt_sub_13.tif"  # spliced
image_name = "canong3_08_sub_08.tif"  # authentic

root = str(create_debug_folder(DEBUG_ROOT))

engine = ExifVisualizer()

dataset = find_dataset_of_image(DATASETS_ROOT, image_name)
if not dataset:
    raise InvalidArgumentException("Impossible to find the dataset this image belongs to")

dataset = dataset(DATASETS_ROOT)
image_path = dataset.get_image(image_name)
mask, _ = dataset.get_mask_of_image(image_name)

image = Picture(image_path)

engine.prediction_pipeline(image, path=os.path.join(root, "result"))
