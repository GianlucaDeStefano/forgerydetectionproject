from abc import ABC

import numpy as np
import tensorflow as tf
from tensorflow.python.ops.gen_math_ops import squared_difference

from Attacks.BaseWhiteBoxAttack import BaseWhiteBoxAttack
from Utilities.Image.Picture import Picture
from Utilities.Visualizers import ExifVisualizer


class BaseExifAttack(BaseWhiteBoxAttack, ABC):
    """
        This class is used to implement white box attacks on the Exif-Sc detector
    """

    def __init__(self, steps: int, alpha: float, detector: ExifVisualizer = None,
                 regularization_weight=0.05, plot_interval=5, patch_size=(128, 128), batch_size: int = 64,
                 debug_root: str = "./Data/Debug/", verbosity: int = 2):
        """
        :param steps: number of attack iterations to perform
        :param alpha: strength of the attack
        :param detector: instance of the detector class to use to process the results, usefull also to share weights
            between multiple instances of the attack
        :param regularization_weight: [0,1] importance of the regularization factor in the loss function
        :param plot_interval: how often (# steps) should the step-visualizations be generated?
        :param patch_size: Width and Height of the patches we are using to compute the Exif parameters
                            we assume that the the patch is always a square eg patch_size[0] == patch_size[1]
        :param batch_size: how many patches shall be processed in parallel
        :param debug_root: root folder inside which to create a folder to store the data produced by the pipeline
        :param verbosity: is this a test mode? In test mode visualizations and superfluous steps will be skipped in favour of a
            faster execution to test the code
        """

        super().__init__(detector, steps, alpha, 0.5,
                         regularization_weight,
                         plot_interval, True, debug_root, verbosity)

        assert (patch_size[0] == patch_size[1])
        self.batch_size = batch_size

        self.patch_size = patch_size

        self._engine = None
        self._sess = None

        self.moving_avg_gradient = None
        self.noise = None

        self.get_gradient = self._get_gradient_of_batch

        self.x = None
        self.y = None

        self.gradient_op = None
        self.loss_op = None

    def setup(self, target_image_path: Picture, target_image_mask: Picture, source_image_path: Picture = None,
              source_image_mask: Picture = None, target_forgery_mask: Picture = None):
        del self.detector, self._engine, self._sess

        self.detector = ExifVisualizer()
        self._engine = self.detector._engine.model.solver.net
        self._sess = self.detector._engine.model.solver.sess

        self.x = tf.compat.v1.placeholder(tf.float32, shape=[None, 128, 128, 3])
        self.y = tf.compat.v1.placeholder(tf.float32, shape=[None, 4096])

        super().setup(target_image_path, target_image_mask, source_image_path, source_image_mask, target_forgery_mask)

        # create variable to store the generated adversarial noise
        self.noise = np.zeros(target_image_path.shape)

        # create variable to store the momentum of the gradient
        self.moving_avg_gradient = np.zeros(target_image_path.shape)

    def _on_before_attack(self):
        """
        Populate the gradent_op and loss_op variables
        :return:
        """

        super()._on_before_attack()

        with self._sess.as_default():
            self.gradient_op, self.loss_op = self._get_gradient_of_batch()

    def _get_gradient_of_batch(self):
        """
        Given a list of patches ready to be processed and their target representation, compute the gradient w.r.t the
        specified loss
        :param x: list of patches to process (len(batch_patches) must be equal to
            self.batch_size)
        :param y: list of target representations (len(target_batch_patches) must
            be equal to self.batch_size)
        :param regularization_value: regularizataion value that will be applied to the loss function
        :return: return a list of gradients (one per patch) and the cumulative loss
        """

        # perform feed forward pass
        feature_representation = self._engine.extract_features_resnet50(self.x, "test", reuse=True)

        # compute the loss with respect to the target representation
        loss = tf.norm(feature_representation - self.y, 2)

        # construct the gradients object
        gradients = tf.gradients(loss, self.x)

        return gradients, loss

    def loss_function(self, y_pred, y_true):
        """
        Specify a loss function to drive the image we are attacking towards the target representation
        The default loss is the l2-norm
        :param y_pred: last output of the model
        :param y_true: target representation
        :return: loss value
        """
        return tf.reduce_sum(squared_difference(y_pred, y_true), [1, 2])

    def regularizer_function(self, perturbation=None):
        """
        Compute te regularization value to add to the loss function
        :param perturbation:perturbation for which to compute the regularization value
        :return: regularization value
        """

        # if no perturbation is given return 0
        if perturbation is None:
            return 0

        return tf.norm(perturbation, ord='euclidean', axis=[1, 2])
