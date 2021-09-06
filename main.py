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
from Ulitities.io.folders import create_debug_folder
import tensorflow as tf

DEBUG_ROOT = os.path.abspath("Data/Debug/")
DATASETS_ROOT = os.path.abspath("Data/Datasets/")

images = [
    "canong3_canonxt_sub_13.tif",  # canong3
    "canonxt_kodakdcs330_sub_01.tif",  # canonxt
    "nikond70_kodakdcs330_sub_22.tif",  # nikond70
    "splicing-70.png",
    #"pristine/r1be2a3d5t.TIF",  # NikonD90
]

root = create_debug_folder(DEBUG_ROOT)

log_file = open(os.path.join(root, "log noise.txt"), "w+")

engine = ExifEngine()

root_folder = os.path.join(root, "noise addition")

log_file.write("\n \n {} \n".format("#### WHITE NOISE RESULTS ####"))
log_file.flush()

# gaussian addition noise
for standard_deviation in range(0, 10, 1):

    break
    root_folder_level = os.path.join(root_folder, str(standard_deviation*0.5))
    os.makedirs(root_folder_level)

    cumulative_f1 = 0
    cumulative_mcc = 0

    for image_name in tqdm(images):

        dataset = find_dataset_of_image(DATASETS_ROOT, image_name)
        if not dataset:
            raise InvalidArgumentException("Impossible to find the dataset this image belongs to")

        dataset = dataset(DATASETS_ROOT)

        image_path = dataset.get_image(image_name)
        mask, _ = dataset.get_mask_of_image(image_name)

        image = Picture(image_path)

        image = (image + np.random.normal(0, standard_deviation*0.5, size=image.shape)).clip(0,255)

        heatmap, predicted_mask = engine.detect(image.to_float())

        predicted_mask = np.rint(predicted_mask)

        cv2.imwrite(os.path.join(root_folder_level, '{}.png'.format(basename(image_name))), predicted_mask * 255)

        cumulative_mcc += matthews_corrcoef(mask.flatten(),predicted_mask.flatten())

        cumulative_f1 += f1_score(mask.flatten(),predicted_mask.flatten())

    msg = "standard_deviation:{}, f1:{:.2f} mcc:{:.2f}".format(standard_deviation*0.5,cumulative_f1/len(images),cumulative_mcc/len(images))
    print(msg)
    log_file.write(msg + "\n")
    log_file.flush()


log_file.write("\n \n {} \n".format("#### JPEG RESULTS ####"))
log_file.flush()

root_folder = os.path.join(root, "jpeg compression")

for quality in range(5,25, 5):

    root_folder_level = os.path.join(root_folder, str(quality))
    os.makedirs(root_folder_level)

    cumulative_f1 = 0
    cumulative_mcc = 0

    for image_name in tqdm(images):

        dataset = find_dataset_of_image(DATASETS_ROOT, image_name)
        if not dataset:
            raise InvalidArgumentException("Impossible to find the dataset this image belongs to")

        dataset = dataset(DATASETS_ROOT)

        image_path = dataset.get_image(image_name)
        mask, _ = dataset.get_mask_of_image(image_name)

        qf = 100 - quality

        if quality < 0:
            path = image_path
        else:
            path = os.path.join(root_folder, 'compressed_image.jpg')
            img = Image.fromarray(np.array(Picture(image_path), np.uint8))
            img.save(path, quality=qf)
            print("jpeg quality: {}".format(qf))

        image = Picture(path=path)
        heatmap, predicted_mask = engine.detect(image.to_float())

        predicted_mask = np.rint(predicted_mask)

        cv2.imwrite(os.path.join(root_folder_level, '{}.png'.format(basename(image_name))), predicted_mask * 255)

        cumulative_mcc += matthews_corrcoef(mask.flatten(),predicted_mask.flatten())

        cumulative_f1 += f1_score(mask.flatten(),predicted_mask.flatten())

    msg = "quality:{}, f1:{:.2f} mcc:{:.2f}".format(quality,cumulative_f1/len(images),cumulative_mcc/len(images))
    print(msg)
    log_file.write(msg + "\n")
    log_file.flush()

log_file.close()