import os
from os.path import basename
from pathlib import Path

from Attacks.Exif.Mimicking.ExifMimickingIntelligentAttack import ExifIntelligentAttack
from Attacks.Noiseprint.Mimiking.NoiseprintMimickingIntelligent import NoiseprintIntelligentMimickingAttack
from Attacks.Noiseprint.Mimiking.NoiseprintMimickingIntelligentGlobal import NoiseprintGlobalIntelligentMimickingAttack
from Datasets.DSO.DsoDataset import DsoDatasetDataset
from Ulitities.Image.Picture import Picture
from Ulitities.Image.functions import create_target_forgery_map

DEBUG_ROOT = os.path.abspath("Data/Debug/")
DATASETS_ROOT = os.path.abspath("Data/Datasets/")
OUTPUT_ROOT  = os.path.abspath("Data/Tampered/DSO/Exif")

attack = ExifIntelligentAttack(50, 5,plot_interval=50,verbosity=0)

dataset = DsoDatasetDataset(DATASETS_ROOT)

paths_images = dataset.get_authentic_images()

for path_image in paths_images:

    if Path(os.path.join(OUTPUT_ROOT, Path(path_image).stem + ".png")).exists():
        continue

    image = Picture(path=path_image)

    mask, mask_path = dataset.get_mask_of_image(path_image)

    target_forgery_mask = None

    for i in range(0, 15):
        candidate_mask = create_target_forgery_map(mask.shape)

        overlap = (candidate_mask == 1) & (mask == 1)

        if overlap.sum() == 0:
            target_forgery_mask = Picture(candidate_mask)
            break

    if target_forgery_mask is None:
        continue

    attack.setup(image, Picture(mask), target_forgery_mask=target_forgery_mask)
    attacked_image = Picture(attack.execute())
    attacked_image.save(os.path.join(OUTPUT_ROOT, Path(path_image).stem + ".png"))

    final_heatmap, final_mask_original = attack.detector._engine.detect(attacked_image.to_float(), Picture(mask))

