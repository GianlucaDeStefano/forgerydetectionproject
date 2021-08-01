import os
import random
import time

import numpy as np

from Attacks import LotsNoiseprint1B, LotsNoiseprint2, LotsNoiseprint1
from Datasets.RIT.RitDataset import RitDataset
from Ulitities.Image.Picture import Picture

DEBUG_ROOT = os.path.abspath("Data/Debug/")
DATASETS_ROOT = os.path.abspath("Data/Datasets/")

# number of images to use to generate the average adversarial noise
N_IMAGES_TO_USE = 5

attack_to_use = LotsNoiseprint1B

times = time.time()
debug_folder = os.path.join(DEBUG_ROOT, str(times))
os.makedirs(os.path.join(debug_folder, "Individual attacks"))

if __name__ == "__main__":

    # shape of the adversarial noise
    target_shape = (1920, 1080)

    # use the RIT dataset since it contains images of constant size
    dataset = RitDataset(DATASETS_ROOT)

    # get the image to use
    images = dataset.get_forged_images(target_shape)

    # noise object
    noise = np.zeros((1080, 1920))

    patch_size = (16, 16)

    for image_path in images[:N_IMAGES_TO_USE]:
        mask, mask_path = dataset.get_mask_of_image(image_path)

        # load picture and mask
        image = Picture(image_path)
        mask = Picture(mask)

        # prepare the attack
        attack = attack_to_use(image, mask, steps=50, debug_root=os.path.join(debug_folder, "Individual attacks"),
                               plot_interval=0, verbose=False, patch_size=patch_size)

        # execute the attack
        attack.execute()

        # read the generated adversarial noise
        noise += attack.adversarial_noise / N_IMAGES_TO_USE

        np.save(os.path.join(debug_folder, "average_noise_{}_{}".format(attack.name,N_IMAGES_TO_USE)), noise)
