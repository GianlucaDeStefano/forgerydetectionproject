import os
from os.path import basename
from Attacks.AdversarialAttacks.GaussianNoiseAddition import GaussianNoiseAdditionAttack
from Attacks.AdversarialAttacks.JpegCompression import JpegCompressionAttack
from Datasets import get_image_and_mask
from Ulitities.io.folders import create_debug_folder
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*6)])
  except RuntimeError as e:
    print(e)


DEBUG_ROOT = os.path.abspath("Data/Debug/")
DATASETS_ROOT = os.path.abspath("Data/Datasets/")

# images to try to attack
images = [
    "pristine/r1be2a3d5t.TIF"
]


if __name__ == "__main__":

    debug_folder = os.path.join(create_debug_folder(DEBUG_ROOT), "images")
    os.makedirs(debug_folder)

    for image_path in images:

        debug_folder_image = os.path.join(debug_folder, basename(image_path))
        os.makedirs(debug_folder_image)

        quality = 2

        image, mask = get_image_and_mask(DATASETS_ROOT, image_path)

        attack = GaussianNoiseAdditionAttack(image,mask,debug_root=debug_folder_image,mean=0,standard_deviation=quality)

        attack.execute()
