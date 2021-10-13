import os
import random
import statistics
from os.path import basename
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, matthews_corrcoef
from tqdm import tqdm

from Attacks.Exif.Mimicking.ExifMimickingIntelligentAttack import ExifIntelligentAttack
from Datasets.DSO.DsoDataset import DsoDatasetDataset
from Ulitities.Image.Picture import Picture
from Ulitities.Visualizers.ExifVisualizer import ExifVisualizer
from Ulitities.io.folders import create_debug_folder

DEBUG_ROOT = os.path.abspath("Data/Debug/")
DATASETS_ROOT = os.path.abspath("Data/Datasets/")

root = create_debug_folder(DEBUG_ROOT)
attacks_root = os.path.join(root, "attacks")
results_root = os.path.join(root, "results")
os.mkdir(results_root)
os.mkdir(attacks_root)

possible_forgery_masks = [
    "./Data/custom/DSO_target_forgery_masks/1.png",
    "./Data/custom/DSO_target_forgery_masks/2.png",
    "./Data/custom/DSO_target_forgery_masks/3.png",
    "./Data/custom/DSO_target_forgery_masks/4.png",
    "./Data/custom/DSO_target_forgery_masks/5.png"
]

dataset = DsoDatasetDataset(DATASETS_ROOT)

target_shape = (1536, 2048)

paths_images = dataset.get_authentic_images(target_shape) + dataset.get_forged_images(target_shape)

attack_class = ExifIntelligentAttack
visualizer = ExifVisualizer()

original_f1_scores = []
original_mcc_scores = []
target_f1_scores = []
target_mcc_scores = []

for image_path in tqdm(paths_images):
    image = Picture(path=image_path)

    mask, mask_path = dataset.get_mask_of_image(image_path)
    target_forgery_mask = Picture(
        np.where(np.all(Picture(path=str(random.choice(possible_forgery_masks))) == (255, 255, 255), axis=-1), 1, 0))

    print(Picture(path=str(random.choice(possible_forgery_masks))).shape)
    image_results_root = os.path.join(results_root, Path(image_path).stem)
    os.mkdir(image_results_root)

    try:
        attack = attack_class(image, Picture(mask, mask_path), target_forgery_mask, 30, detector=visualizer, alpha=10,
                              verbosity=0,
                              debug_root=attacks_root, plot_interval=-1)

        attacked_image = attack.execute()
    except:
        continue

    attacked_image.save(os.path.join(image_results_root, "attacked_image.png"))

    heatmap, predicte_mask = visualizer.prediction_pipeline(attacked_image, os.path.join(image_results_root, "result"))

    target_forgery_mask.save(os.path.join(image_results_root, "target_mask.png"))

    predicte_mask = np.rint(predicte_mask)

    original_f1_scores += [f1_score(mask.flatten(), predicte_mask.flatten())]
    original_mcc_scores += [matthews_corrcoef(mask.flatten(), predicte_mask.flatten())]

    target_f1_scores += [f1_score(target_forgery_mask.flatten(), predicte_mask.flatten())]
    target_mcc_scores += [matthews_corrcoef(target_forgery_mask.flatten(), predicte_mask.flatten())]

    print("original forgery = avg_f1:{:.2f} avg_mcc:{:.2f}".format(statistics.mean(original_f1_scores),
                                                                   statistics.mean(original_mcc_scores)))
    print("target forgery = avg_f1:{:.2f} avg_mcc:{:.2f}".format(statistics.mean(target_f1_scores),
                                                                 statistics.mean(target_mcc_scores)))

print("Final original forgery = avg_f1:{:.2f} avg_mcc:{:.2f}".format(statistics.mean(original_f1_scores),
                                                                     statistics.mean(original_mcc_scores)))
print("Final target forgery = avg_f1:{:.2f} avg_mcc:{:.2f}".format(statistics.mean(target_f1_scores),
                                                                   statistics.mean(target_mcc_scores)))
