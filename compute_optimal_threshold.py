import os
import random

import numpy as np
from tqdm import tqdm

from Datasets.ColumbiaUncompressed.ColumbiaUncompressedDataset import ColumbiaUncompressedDataset
from Detectors.Noiseprint.utility import jpeg_quality_of_file
from Ulitities.Image.Picture import Picture
from Ulitities.Visualizers.NoiseprintVisualizer import NoiseprintVisualizer
from sklearn.metrics import f1_score

DATASETS_ROOT = os.path.abspath("./Data/Datasets/")

min_threshold = 100
max_threshold = 1000
step = 1
n_images = 50

dataset = ColumbiaUncompressedDataset(DATASETS_ROOT)

visualizer = NoiseprintVisualizer(101)

images = dataset.get_forged_images()
random.shuffle(images)

average_f1 = {}

for threshold in range(min_threshold,max_threshold,step):
    average_f1[threshold] = 0

for image_path in tqdm(images[:n_images]):

    try:
        qf = jpeg_quality_of_file(image_path)
    except:
        qf = 101

    if visualizer.qf != qf:
        visualizer.load_quality(qf)

    image = Picture(str(image_path))

    heatmap = visualizer.compute(image)

    mask,_ = dataset.get_mask_of_image(image_path)

    for threshold in range(min_threshold,max_threshold,step):
        pred_mask = np.array(heatmap > threshold,int)
        average_f1[threshold] += f1_score(mask.flatten(),pred_mask.flatten())/n_images


with open("thresholds.txt", "w") as file:

    best_threshold_index = max(average_f1, key=average_f1.get)

    file.write("Best threshold: {}, F1:{}\n".format(best_threshold_index, average_f1[best_threshold_index]))
    for threshold,f1 in average_f1.items():
        file.write("{},{}\n".format(threshold,f1))
