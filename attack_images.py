import os

from Attacks import supported_attacks
from Datasets import find_dataset_of_image
from attack_image import attack_image

DEBUG_ROOT = os.path.abspath("./Data/Debug/")
DATASETS_ROOT = os.path.abspath("./Data/Datasets/")

# images to try to attack
images = [
    "canong3_canonxt_sub_13.tif",  # canong3
    "canonxt_kodakdcs330_sub_01.tif",  # canonxt
    "nikond70_kodakdcs330_sub_22.tif",  # nikond70
    "splicing-70.png",
    "DPP0122.TIF",  # Canon60D
    "r1be2a3d5t.TIF",  # NikonD90
    "r09696ba3t.TIF",  # Nikon D7000
    "DSC05635.TIF",  # Sony A57
]

attacks = ["Lots4Noiseprint.2","Lots4Noiseprint.3"]

for attack_type in attacks:
    for image in images:
        attack_image(image, attack_type=attack_type)
