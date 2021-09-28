import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from Datasets import get_image_and_mask
from Detectors.Exif.lib.utils import ops
from Detectors.Exif.models.exif import exif_net
from Ulitities.Image.Picture import Picture

model_checkpoint = "Detectors/Exif/ckpt/exif_final/exif_final.ckpt"

DEBUG_ROOT = os.path.abspath("Data/Debug/")
DATASETS_ROOT = os.path.abspath("Data/Datasets/")

# images to try to attack
target_image_path = "canong3_08_sub_08.tif"

source_image_path = "canong3_08_sub_08.tif"

net_args = {'num_classes': 80 + 3,
            'is_training': False,
            'train_classifcation': True,
            'freeze_base': True,
            'im_size': 128,
            'batch_size': 64,
            'use_gpu': [0],
            'use_tf_threading': False,
            'learning_rate': 1e-4}

tf.compat.v1.disable_eager_execution()

target_image, target_image_mask = get_image_and_mask(DATASETS_ROOT, target_image_path)
source_image, source_image_mask = get_image_and_mask(DATASETS_ROOT, source_image_path)

net = exif_net.initialize(net_args)

patches = Picture(target_image).divide_in_patches((128, 128), force_shape=False)
batch_size = 1

for batch_idx in tqdm(range(0, (len(patches) + batch_size - 1) // batch_size, 1)):
    starting_idx = batch_size * batch_idx
    batch_patches = patches[starting_idx:min(starting_idx + batch_size, len(patches))]

    patch = np.array(batch_patches)
    tensor_patch = tf.convert_to_tensor(patch,dtype=tf.float32)

    feature_patch = net.extract_features_resnet50(tensor_patch, "test",reuse=True)
