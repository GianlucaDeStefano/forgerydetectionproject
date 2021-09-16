import os
from os.path import basename

import numpy as np
from cv2 import PSNR
from Attacks.Noiseprint.Mimiking.NoiseprintMimickingAttack import NoiseprintMimickingAttack
from Datasets import get_image_and_mask
from Ulitities.Image.Picture import Picture
from Ulitities.Visualizers.NoiseprintVisualizer import NoiseprintVisualizer
from Ulitities.io.folders import create_debug_folder
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(gpus[0], [
            tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024 * 6)])
    except RuntimeError as e:
        print(e)

DEBUG_ROOT = os.path.abspath("Data/Debug/")
DATASETS_ROOT = os.path.abspath("Data/Datasets/")

# images to try to attack
images_for_geneating_noise = [
    "splicing-07.png",
    "normal-42.png",
    "splicing-93.png",
    "splicing-86.png",
]

source_image_path = "./Data/custom/splicing-70-artificial.png"

images_to_attack_bb = [
    "splicing-28.png",
    "splicing-29.png",
    "splicing-30.png"
]

if __name__ == "__main__":

    debug_folder = os.path.join(create_debug_folder(DEBUG_ROOT))
    mimicking_debug_folder = os.path.join(debug_folder,"mimiking attack")
    os.makedirs(mimicking_debug_folder)

    source_image = Picture(str(source_image_path))
    source_image_mask = Picture(np.where(np.all(source_image == (255, 255, 255), axis=-1), 0, 1))

    noise = np.zeros(source_image_mask.shape)

    for image_path in images_for_geneating_noise:

        image, mask = get_image_and_mask(DATASETS_ROOT, image_path)

        attack = NoiseprintMimickingAttack(image, mask, source_image, source_image_mask, 50, 5,plot_interval=0,debug_root=mimicking_debug_folder)

        attack.execute()

        noise += attack.noise / len(image_path)

        np.save(os.path.join(debug_folder,"noise"), noise)

    noiseprint_detector = NoiseprintVisualizer()

    transferred_debug_folder = os.path.join(debug_folder, "transferred noise")
    os.makedirs(transferred_debug_folder)

    for image_path in images_to_attack_bb:
        image, mask = get_image_and_mask(DATASETS_ROOT, image_path)

        attacked_image =Picture(np.array((image - Picture(noise).three_channels(1/3,1/3,1/3)).clip(0,255),np.int))

        psnr = PSNR(image, np.array(attacked_image, np.int))

        noiseprint_detector.prediction_pipeline(attacked_image,
                                                os.path.join(transferred_debug_folder, "{}".format(basename(image_path)))
                                                , image,note="PSNR:{:.2f}".format(psnr), omask=mask)
