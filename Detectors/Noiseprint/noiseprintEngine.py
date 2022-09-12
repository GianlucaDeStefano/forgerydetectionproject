import logging
import multiprocessing
import os
import time
import traceback

import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.metrics import f1_score
from tensorflow.python.keras.layers import Conv2D, BatchNormalization, Activation
from tensorflow.python.keras.models import Model

# Bias layer necessary because noiseprint applies bias after batch-normalization.
from tqdm import tqdm

from Detectors.DetectorEngine import DetectorEngine, find_optimal_mask
from Detectors.Noiseprint.noiseprint_blind import noiseprint_blind_post, genMappFloat
from Detectors.Noiseprint.utility.utility import jpeg_quality_of_file
from Detectors.Noiseprint.utility.utilityRead import jpeg_qtableinv, imread2f, three_2_one_channel, \
    imread2int_pil, imconver_int_2_float
from Utilities.Image.Picture import Picture
import multiprocessing as mp


class NoiseprintEngine(DetectorEngine):
    _save_path = os.path.join(os.path.dirname(__file__), './weights/net_jpg%d/')
    slide = 512  # 3072
    large_limit = 262144  # 9437184
    overlap = 34

    force_reload_quality = True

    def __init__(self, setup_on_init=True, quality_level=None):

        super().__init__("Noiseprint")

        self._model = _full_conv_net()
        self._fixed_quality_level = quality_level
        self._loaded_quality = quality_level
        self._session = None
        self.i = 0
        if setup_on_init:
            self.reset()

    def destroy(self):
        """
        @return:
        """

        # check if a session object exists
        if self._session:

            # close tf section if open
            if not self._session._closed:
                self._session.close()

            # remove the session object
            self._session = None

    def reset_instance(self):
        """
        Destroy & Reload the detector's instance from memory
        @return:
        """
        # If a session is loaded and the reload_session flag is true close it
        self.destroy()

        print("\n \n NEW SESSION CREATED \n \n")
        self._session = setup_session()

    def reset(self, reset_instance=False, reset_metadata=True):
        """
        Reset Noiseprint to a pristine state
        @return: None
        """

        # call the constructor reset class
        super().reset(reset_instance, reset_metadata)

        # Reset the quality level
        if reset_metadata:
            self._loaded_quality = None

    @staticmethod
    def load_sample(sample_path):
        """
        Method that specifies the correct procedure to load a sample from memory
        @param sample_path: sample to load
        @return: loaded sample
        """
        return imread2int_pil(sample_path, channel=3, dtype=np.float32)

    @staticmethod
    def transform_sample(pristine_sample):
        """
        Apply the detector-specific transformations necessary to transform a sample into
        the state that the classifier wants.
        As an assumption metadata["sample"] is a numpy array of type np.uint8
        @param pristine_sample: sample to transform in the forma of a 3d matrix of integers in the range of [0,255]
        @return:
        """

        assert (len(pristine_sample.shape) == 3)
        assert (np.min(pristine_sample) >= 0 and 255 >= np.max(pristine_sample) > 1)

        one_channel_image = three_2_one_channel(pristine_sample)
        scaled_image = imconver_int_2_float(one_channel_image, np.float32)

        return scaled_image

    @property
    def transformed_sample(self):
        """
        Sample loaded and transformed, ready to be processed by the noiseprint models
        @return:
        """
        # Make sure the necessary data has been loaded
        assert ("sample" in self.metadata.keys())

        return self.transform_sample(self.metadata["sample"])

    def initialize(self, sample_path=None, sample=None, reset_instance=False, reset_metadata=True):
        """
        Initialize the detector to handle a new sample, inferring also the Noiseprint's quality factor to use.

        @param sample_path: str
            Path of the sample to analyze
        @param sample: numpy.array
            Preloaded sample to use (to be useful sample_path has to be None)
        @param reset_instance: Bool
            A flag indicating if this detector's model should be reimported
        @param reset_metadata:Bool
            A flag indicating if the metadata should be reinitialized before loading the new sample
        @param reset_metadata:Bool
            A flag indicating if the metadata should be reinitialized before loading the new sample

        """

        # Generic initialization of the detector engine
        super(NoiseprintEngine, self).initialize(sample_path, sample, reset_instance, reset_metadata)

        if not reset_metadata and sample is not None:
            # In case the metadata have not been completely widped,
            # delete still the entries containing noiseprint to ensure a fair new run
            if "noiseprint" in self.metadata: del self.metadata["noiseprint"]

        # Make sure the necessary data has been loaded
        assert (self.metadata["sample_path"] or self.metadata["sample"] is not None)

        # check if a sample instance has been given
        if self.metadata["sample"] is None:

            # no instance given, load it from the path
            assert (self.metadata["sample_path"] is not None)

            # read the necessary metadata
            sample_path = self.metadata["sample_path"]

            # Load the sample
            sample = self.load_sample(sample_path)
            self.metadata["sample"] = sample

            # Infer the quality level to use only if necessary
            if self._loaded_quality is None or self.force_reload_quality:

                if not self._fixed_quality_level:
                    try:
                        # Infer the quality level from the quantization table
                        self._loaded_quality = jpeg_qtableinv(sample_path)
                    except Exception as E:
                        self.logger_module.info(
                            f"Impossible to infer quality from file for the following error:\n {E} \n adopting default quality level 101")
                        self._loaded_quality = 101
                else:
                    self._loaded_quality = self._fixed_quality_level

                # Load the model with the desired quality level
                self.load_quality(self._loaded_quality)
                self.logger_module.info(f"LOADED quality: {self._loaded_quality}")

            # Save the adopted quality level in the metadata
            self.metadata["quality_level"] = self._loaded_quality

    def extract_features(self):
        """
        Extract from the image its noiseprint map and save int into the output dict object.
        @param image: Picture
            The sample to analyze
        @return metadata: dict
            Dictionary containing all the metadata and results generated during the process
        """

        # check if the features have already been extracted
        if "noiseprint" in self.metadata:
            return self.metadata

        # Make sure the necessary data has been loaded
        assert ("sample" in self.metadata.keys())

        # read the necessary metadata
        sample = self.transformed_sample

        # produce the noiseprint
        noiseprint = self.predict(sample)

        self.metadata["noiseprint"] = noiseprint

        return self.metadata

    def generate_heatmap(self):
        """
        Function extracting the necessary features from the sample and saving them into the
        output dict object
        @param image: Picture
            The sample to analyze
        @return metadata: dict
            Dictionary containing all the metadata and results generated during the process
        """

        # check if the heatmap has already been computed
        if "heatmap" in self.metadata:
            return self.metadata

        # Make sure the necessary data has been loaded
        assert (s in self.metadata for s in ["sample", "noiseprint"])

        # Read the necessary metadata
        sample = self.transform_sample(self.metadata["sample"])
        noiseprint = self.metadata["noiseprint"]

        # Produce the heatmap
        mapp, valid, range0, range1, imgsize, other = noiseprint_blind_post(noiseprint, sample)
        heatmap = genMappFloat(mapp, valid, range0, range1, imgsize)

        # Normalize the heatmap between the range [0,1]
        heatmap = (heatmap - heatmap.min())
        heatmap = heatmap / heatmap.max()

        # More than halp of the pixels are above the 0.5 threshold, flip them
        if np.mean(heatmap > 0.5) > 0.5:
            heatmap = 1 - heatmap

        # Save the heatmap into the metadata object
        self.metadata["heatmap"] = heatmap

        return self.metadata

    def generate_mask(self, threshold=None, gt_mask=None, metric=None):
        """
        Returns the mask of the forged area computing it from the heatmap
        @param threshold: threshold to use to segment the heatmap
        @param gt_mask: ground truth mask used to compute the optimal threshold (works only if metric != None)
        @param metric: metric function used to compute the optimal threshold (works only if gt_mask != None)
        @return: mask of the forged area
        """
        assert (self.metadata["heatmap"] is not None)
        mask = None
        if threshold is not None:
            mask = np.where(self.metadata["heatmap"] > threshold, 1, 0)
        elif metric is not None and gt_mask is not None:
            mask,threshold = find_optimal_mask(self.metadata["heatmap"], gt_mask, metric)
        else:
            raise ValueError

        mask = np.array(np.rint(mask), dtype=np.uint8)

        if np.mean(mask == 1) > 0.5:
            mask = 1 - mask

        self.metadata["mask"] = mask

        return mask,threshold

    def load_quality(self, quality):
        """
        Loads a quality level for the next noiseprint predictions.
        Quality level can be obtained by the file quantization table.
        :param quality: Quality level, int between 51 and 101 (included)
        """
        if quality < 51 or quality > 101:
            raise ValueError("Quality must be between 51 and 101 (included). Provided quality: %d" % quality)
        self.logger_module.info("Loading checkpoint quality %d" % quality)
        checkpoint = self._save_path % quality
        self._model.load_weights(checkpoint)
        self._loaded_quality = quality

    @tf.function(experimental_relax_shapes=True,
                 input_signature=[tf.TensorSpec(shape=(1, None, None, 1), dtype=tf.float32)])
    def _predict(self, img):
        return self._model(img)

    def _predict_small(self, img):
        return np.squeeze(self._predict(img[np.newaxis, :, :, np.newaxis]).numpy())

    def _predict_large(self, img):
        # prepare output array
        res = np.zeros((img.shape[0], img.shape[1]), np.float32)

        # iterate over x and y, strides = self.slide, window size = self.slide+2*self.overlap
        for x in range(0, img.shape[0], self.slide):
            x_start = x - self.overlap
            x_end = x + self.slide + self.overlap
            for y in range(0, img.shape[1], self.slide):
                y_start = y - self.overlap
                y_end = y + self.slide + self.overlap
                patch = img[max(x_start, 0): min(x_end, img.shape[0]), max(y_start, 0): min(y_end, img.shape[1])]
                patch_res = np.squeeze(self._predict_small(patch))

                # discard initial overlap if not the row or first column
                if x > 0:
                    patch_res = patch_res[self.overlap:, :]
                if y > 0:
                    patch_res = patch_res[:, self.overlap:]
                # discard data beyond image size
                patch_res = patch_res[:min(self.slide, patch.shape[0]), :min(self.slide, patch.shape[1])]
                # copy data to output buffer
                res[x: min(x + self.slide, res.shape[0]), y: min(y + self.slide, res.shape[1])] = patch_res
        return res

    def predict(self, img: np.array):
        """
        Run the noiseprint generation CNN over the input image
        :param img: input image, 2-D numpy array
        :return: output noisepritn, 2-D numpy array with the same size of the input image
        """
        if len(img.shape) != 2:
            raise ValueError("Input image must be 2-dimensional. Passed shape: %r" % img.shape)
        if self._loaded_quality is None:
            raise RuntimeError("The engine quality has not been specified, please call load_quality first")
        if img.shape[0] * img.shape[1] > self.large_limit:
            return self._predict_large(img)
        else:
            return self._predict_small(img)

    @property
    def model(self):
        return self._model


class BiasLayer(tf.keras.layers.Layer):
    """
    Simple bias layer
    """

    def build(self, input_shape):
        self.bias = self.add_weight('bias', shape=input_shape[-1], initializer="zeros")

    @tf.function
    def call(self, inputs, training=None):
        return inputs + self.bias


def _full_conv_net(num_levels=17, padding='SAME'):
    """FullConvNet model."""
    activation_fun = [tf.nn.relu, ] * (num_levels - 1) + [tf.identity, ]
    filters_num = [64, ] * (num_levels - 1) + [1, ]
    batch_norm = [False, ] + [True, ] * (num_levels - 2) + [False, ]

    inp = tf.keras.layers.Input([None, None, 1])
    model = inp

    for i in range(num_levels):
        model = Conv2D(filters_num[i], 3, padding=padding, use_bias=False)(model)
        if batch_norm[i]:
            model = BatchNormalization(epsilon=1e-5)(model)
        model = BiasLayer()(model)
        model = Activation(activation_fun[i])(model)

    return Model(inp, model)


def setup_session():
    """
    Set the session allow_growth option for GPU usage
    """
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 1
    session = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(session)
    return session


def gen_noiseprint(image, quality=None):
    """
    Generates the noiseprint of an image
    :param image: image data. Numpy 2-D array or path string of the image
    :param quality: Desired quality level for the noiseprint computation.
    If not specified the level is extracted from the file if image is a path string to a JPEG file, else 101.
    :return: The noiseprint of the input image
    """
    if isinstance(image, str):
        # if image is a path string, load the image and quality if not defined
        if quality is None:
            try:
                quality = jpeg_quality_of_file(image)
            except AttributeError:
                quality = 101
        image = np.asarray(Image.open(image).convert("YCbCr"))[..., 0].astype(np.float32) / 256.0
    else:
        if quality is None:
            quality = 101
    engine = NoiseprintEngine()
    engine.load_quality(quality)
    return engine.predict(image)


def normalize_noiseprint(noiseprint, margin=34):
    """
    Normalize the noiseprint between 0 and 1, in respect to the central area
    :param noiseprint: noiseprint data, 2-D numpy array
    :param margin: margin size defining the central area, default to the overlap size 34
    :return: the normalized noiseprint data, 2-D numpy array with the same size of the input noiseprint data
    """
    v_min = noiseprint.min()
    v_max = noiseprint.max()
    if margin > 0:
        v_min = np.min(noiseprint[margin:-margin, margin:-margin])
        v_max = np.max(noiseprint[margin:-margin, margin:-margin])

    return ((noiseprint - v_min) / (v_max - v_min)).clip(0, 1)


def noiseprint_blind_file(filename, model_name='net'):
    try:
        img, mode = imread2f(filename, channel=1)
    except:
        print('Error opening image')
        return -1, -1, -1e10, None, None, None, None, None, None

    try:
        QF = jpeg_qtableinv(filename)
    except:
        QF = 200

    mapp, valid, range0, range1, imgsize, other = noiseprint_blind(img, QF, model_name=model_name)
    return QF, mapp, valid, range0, range1, imgsize, other


def e_score(mask1: np.array, mask2: np.array):
    return np.array(np.equal(mask1, mask2)).sum() ** 2 / (mask1.shape[0] * mask2.shape[0])


def find_best_theshold(heatmap, mask, measure):
    manager = multiprocessing.Manager()

    gap = min(max(10, int(heatmap.max() / 50)), 100)
    min_t = max(int(heatmap.min()), -25600)
    max_t = min(int(heatmap.max()), 25600)
    max_index = -1

    start_time = time.time()
    while gap >= 1:

        return_dict = manager.dict()
        jobs = []

        for threshold in range(min_t, max_t, gap):
            p = multiprocessing.Process(target=test_threshold, args=(heatmap, mask, measure, threshold, return_dict))
            jobs.append(p)
            p.start()

        for proc in jobs:
            proc.join()

        max_index = max(max_index, max(return_dict, key=return_dict.get))

        min_t = max(max_index - gap * 4, 0)
        max_t = min(max_index + gap * 4, int(heatmap.max()))

        if gap > 1:
            gap = gap // 2
        else:
            break

    return max_index


def test_threshold(heatmap, mask, measure, threshold, return_dict):
    pred_mask = np.array(heatmap > threshold, int)
    try:
        return_dict[threshold] = measure(mask.flatten(), pred_mask.flatten())
    except:
        pass


def noiseprint_blind(img, QF):
    res = gen_noiseprint(img, QF)

    if isinstance(img, str):
        img, mode = imread2f(img)

    assert (img.shape == res.shape)
    return noiseprint_blind_post(res, img)
