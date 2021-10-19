import os
import random
import statistics
from os.path import basename
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, matthews_corrcoef
from tqdm import tqdm
from Attacks.Noiseprint.Mimiking.NoiseprintMimickingIntelligentGlobal import NoiseprintGlobalIntelligentMimickingAttack
from Datasets.DSO.DsoDataset import DsoDatasetDataset
from Ulitities.Image.Picture import Picture
from Ulitities.Visualizers.NoiseprintVisualizer import NoiseprintVisualizer
from Ulitities.io.folders import create_debug_folder

DEBUG_ROOT = os.path.abspath("Data/Debug/")
DATASETS_ROOT = os.path.abspath("Data/Datasets/")

root = create_debug_folder(DEBUG_ROOT)
attacks_root = os.path.join(root, "attacks")
results_root = os.path.join(root, "results")
os.mkdir(results_root)
os.mkdir(attacks_root)


dataset = DsoDatasetDataset(DATASETS_ROOT)

target_shape = (1536, 2048)

paths_images = dataset.get_forged_images(target_shape)

attack_class = NoiseprintGlobalIntelligentMimickingAttack
visualizer = NoiseprintVisualizer()

conservative_original_f1_scores = []
conservative_original_mcc_scores = []
conservative_target_f1_scores = []
conservative_target_mcc_scores = []


optimal_original_f1_scores = []
optimal_original_mcc_scores = []
optimal_target_f1_scores = []
optimal_target_mcc_scores = []

for image_path in tqdm(paths_images):
    image = Picture(path=image_path)

    mask, mask_path = dataset.get_mask_of_image(image_path)

    target_forgery_mask = None

    for i in range(0,15):
        candidate_mask = create_target_forgery_map(mask.shape)

        overlap = (candidate_mask == 1) & (create_target_forgery_map == 1)

        if overlap.sum() == 0:
            target_forgery_mask = Picture(candidate_mask)
            break

    if target_forgery_mask is None:
        print("Unable to compute non overlapping mask, skipping {}".format(basename(image_path)))
        continue


    image_results_root = os.path.join(results_root, Path(image_path).stem)
    os.mkdir(image_results_root)

    Picture(mask).save(os.path.join(image_results_root, "original_mask.png"))
    target_forgery_mask.save(os.path.join(image_results_root,"target_forgery_mask.png"))

    try:
        attack = attack_class(image, Picture(mask, mask_path), target_forgery_mask, 30, alpha=10,
                              verbosity=0,
                              debug_root=attacks_root, plot_interval=-1)

        attacked_image = attack.execute()
    except Exception as e:
        print(e)
        print("Skipped: {}".format(basename(image_path)))
        continue

    attacked_image.save(os.path.join(image_results_root, "attacked_image.png"))

    heatmap, _ = visualizer.prediction_pipeline(attacked_image, os.path.join(image_results_root, "result"), omask=mask)

    predicted_mask = visualizer.get_mask(heatmap, mask)

    predicted_mask = np.rint(predicted_mask)

    conservative_original_f1_scores += [f1_score(mask.flatten(), predicted_mask.flatten())]
    conservative_original_mcc_scores += [matthews_corrcoef(mask.flatten(), predicted_mask.flatten())]

    conservative_target_f1_scores += [f1_score(target_forgery_mask.flatten(), predicted_mask.flatten())]
    conservative_target_mcc_scores += [matthews_corrcoef(target_forgery_mask.flatten(), predicted_mask.flatten())]

    print("conservative original forgery = avg_f1:{:.2f} avg_mcc:{:.2f}".format(statistics.mean(conservative_original_f1_scores),
                                                                   statistics.mean(conservative_original_mcc_scores)))
    print("conservative target forgery = avg_f1:{:.2f} avg_mcc:{:.2f}".format(statistics.mean(conservative_target_f1_scores),
                                                                 statistics.mean(conservative_target_mcc_scores)))

    predicted_mask = visualizer.get_mask(heatmap, target_forgery_mask)

    predicted_mask = np.rint(predicted_mask)

    optimal_original_f1_scores += [f1_score(mask.flatten(), predicted_mask.flatten())]
    optimal_original_mcc_scores += [matthews_corrcoef(mask.flatten(), predicted_mask.flatten())]

    optimal_target_f1_scores += [f1_score(target_forgery_mask.flatten(), predicted_mask.flatten())]
    optimal_target_mcc_scores += [matthews_corrcoef(target_forgery_mask.flatten(), predicted_mask.flatten())]

    print("optimal original forgery = avg_f1:{:.2f} avg_mcc:{:.2f}".format(statistics.mean(optimal_original_f1_scores),
                                                                   statistics.mean(optimal_original_mcc_scores)))
    print("optimal target forgery = avg_f1:{:.2f} avg_mcc:{:.2f}".format(statistics.mean(optimal_target_f1_scores),
                                                                 statistics.mean(optimal_target_mcc_scores)))


print("Final conservative original forgery = avg_f1:{:.2f} avg_mcc:{:.2f}".format(statistics.mean(conservative_original_f1_scores),
                                                                     statistics.mean(conservative_original_mcc_scores)))
print("Final conservative target forgery = avg_f1:{:.2f} avg_mcc:{:.2f}".format(statistics.mean(conservative_target_f1_scores),
                                                                   statistics.mean(conservative_target_mcc_scores)))

print("Final optimal original forgery = avg_f1:{:.2f} avg_mcc:{:.2f}".format(statistics.mean(optimal_original_f1_scores),
                                                                     statistics.mean(optimal_original_mcc_scores)))
print("Final optimal target forgery = avg_f1:{:.2f} avg_mcc:{:.2f}".format(statistics.mean(optimal_target_f1_scores),
                                                                   statistics.mean(optimal_target_mcc_scores)))