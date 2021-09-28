import os
import pathlib

import numpy as np
import tensorflow as tf

from Detectors.DetectorEngine import DeterctorEngine
from Detectors.Exif import demo
from Ulitities.Image.Picture import Picture


class ExifEngine(DeterctorEngine):

    def __init__(self):
        super().__init__("ExifEngine")

        ckpt_path = os.path.join(pathlib.Path(__file__).parent, './ckpt/exif_final/exif_final.ckpt')
        self.model = demo.Demo(ckpt_path=ckpt_path, use_gpu=0, quality=3.0, num_per_dim=30)

    def detect(self, image: Picture):
        res = self.model.run(image, use_ncuts=True, blue_high=True)
        return res[0], res[1]

    def get_feature_representation(self, patches: list):
        """
        Compute the feature representation on the fiven patches
        :param patches: list of patches for which we have to compute the target reptresentation
        :return: numpy array in the shape #P x 4096 where #P is the number of patches
        """
        # transform the list of patches into a #P x 128 x 128 x 3 numpy array
        patch = np.array(patches)

        # transform the numpy array into a tensorflow tensor, casting it to the type required by exif
        tensor_patch = tf.convert_to_tensor(patch, dtype=tf.float32)

        # foreach patch, extract the corresponding 4096 exif dimensional vector
        patches_features = self.model.solver.net.extract_features_resnet50(tensor_patch, "test", reuse=True)

        return patches_features


