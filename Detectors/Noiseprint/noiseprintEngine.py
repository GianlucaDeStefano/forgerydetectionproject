import logging
import os

import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.metrics import f1_score
from tensorflow.python.keras.layers import Conv2D, BatchNormalization, Activation
from tensorflow.python.keras.models import Model

# Bias layer necessary because noiseprint applies bias after batch-normalization.
from Detectors.DetectorEngine import DeterctorEngine
from Detectors.Noiseprint.noiseprint_blind import noiseprint_blind_post, genMappFloat
from Detectors.Noiseprint.utility.utility import jpeg_quality_of_file
from Detectors.Noiseprint.utility.utilityRead import jpeg_qtableinv, imread2f
from Ulitities.Image.Picture import Picture


class NoiseprintEngine(DeterctorEngine):
    _save_path = os.path.join(os.path.dirname(__file__), './weights/net_jpg%d/')
    slide = 512  # 3072
    large_limit = 262144  # 9437184
    overlap = 34
    setup_on_init = True

    def __init__(self):

        super().__init__("Noiseprint Engine")

        self._model = _full_conv_net()
        self._loaded_quality = None
        if self.setup_on_init:
            setup_session()

    def detect(self, image: Picture):

        # check that the image has only one channel
        assert (len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1))

        # produce the noiseprint
        noiseprint = self.predict(image)

        # generate heatmap
        mapp, valid, range0, range1, imgsize, other = noiseprint_blind_post(noiseprint, image)
        attacked_heatmap = genMappFloat(mapp, valid, range0, range1, imgsize)

        return attacked_heatmap

    def get_mask(self,image: Picture,gtmask):

        heatmap = self.detect(image)

        threshold = find_best_theshold(heatmap, gtmask)
        predicted_mask = np.array(heatmap > threshold, int).clip(0, 1)


        return predicted_mask

    def load_quality(self, quality):
        """
        Loads a quality level for the next noiseprint predictions.
        Quality level can be obtained by the file quantization table.
        :param quality: Quality level, int between 51 and 101 (included)
        """
        if quality < 51 or quality > 101:
            raise ValueError("Quality must be between 51 and 101 (included). Provided quality: %d" % quality)
        if quality == self._loaded_quality:
            return
        print("Loading checkpoint quality %d" % quality)
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

    def predict(self, img):
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
        print('QF=', QF)
    except:
        QF = 200

    mapp, valid, range0, range1, imgsize, other = noiseprint_blind(img, QF, model_name=model_name)
    return QF, mapp, valid, range0, range1, imgsize, other

def e_score(mask1:np.array,mask2:np.array):
    return np.array(np.equal(mask1,mask2)).sum()**2 / (mask1.shape[0]*mask2.shape[0])

def find_best_theshold(heatmap, mask):
    best_theshold = -1
    best_f1 = -1
    seen_peak = False
    for threshold in range(50, 2000, 5):
        pred_mask = np.array(heatmap > threshold, int)
        score = f1_score(mask.flatten(), pred_mask.flatten())

        if score > best_f1:
            best_f1 = score
            best_theshold = threshold
        else:
            break

    return best_theshold


def noiseprint_blind(img, QF):
    res = gen_noiseprint(img, QF)

    if isinstance(img, str):
        img, mode = imread2f(img)

    assert (img.shape == res.shape)
    return noiseprint_blind_post(res, img)
