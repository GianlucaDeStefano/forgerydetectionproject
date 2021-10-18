import argparse
import os
from abc import ABC, abstractmethod
import numpy as np
import tensorflow as tf
from cv2 import PSNR
from tensorflow.python.ops.gen_math_ops import squared_difference
from tensorflow.python.ops.linalg_ops import norm

from Attacks.BaseIterativeAttack import BaseIterativeAttack
from Detectors.DetectorEngine import DeterctorEngine
from Ulitities.Image.Picture import Picture


def normalize_gradient(gradient, margin=17):
    """
    Normalize the gradient cutting away the values on the borders
    :param margin: margin to use along the bordes
    :param gradient: gradient to normalize
    :return: normalized gradient
    """

    # set to 0 part of the gradient too near to the border
    if margin > 0:
        gradient[0:margin, :] = 0
        gradient[-margin:, :] = 0
        gradient[:, 0:margin] = 0
        gradient[:, -margin:] = 0

    # scale the final gradient using the computed infinity norm
    if np.max(np.abs(gradient)) > 0:
        gradient = gradient / np.max(np.abs(gradient))

    return gradient


class BaseWhiteBoxAttack(BaseIterativeAttack, ABC):
    """
    This class in used as a base to implement attacks whose aim is to reproduce some kind of "target representation"
    """

    name = "Base Mimicking Attack"

    def __init__(self, target_image: Picture, target_image_mask: Picture, source_image: Picture,
                 source_image_mask: Picture, detector: DeterctorEngine, steps: int, alpha: float, momentum_coeficient: float = 0.5,
                 regularization_weight=0.05, plot_interval=5, additive_attack=True,
                 root_debug: str = "./Data/Debug/", verbosity: int = 2):
        """
        :param target_image: original image on which we should perform the attack
        :param target_image_mask: original mask of the image on which we should perform the attack
        :param source_image: image from which we will compute the target representation
        :param source_image_mask: mask of the imae from which we will compute the target representation
        :param detector: name of the detector to be used to visualize the results
        :param steps: number of attack iterations to perform
        :param alpha: strength of the attack
        :param momentum_coeficient: [0,1] how relevant is the velocity ferived from past gradients for computing the
            current gradient? 0 -> not relevant, 1-> is the only thing that matters
        :param regularization_weight: [0,1] importance of the regularization factor in the loss function
        :param plot_interval: how often (# steps) should the step-visualizations be generated?
        :param additive_attack: show we feed the result of the iteration i as the input of the iteration 1+1?
        :param root_debug: root folder inside which to create a folder to store the data produced by the pipeline
        :param verbosity: is this a test mode? In test mode visualizations and superfluous steps will be skipped in favour of a
            faster execution to test the code
        """

        super(BaseWhiteBoxAttack, self).__init__(target_image, target_image_mask, detector, steps, plot_interval,
                                                 additive_attack,
                                                 root_debug, verbosity)

        # save the source image and its mask
        self.source_image = source_image
        self.source_image_mask = source_image_mask

        # save the regularization weight
        self.regularization_weight = regularization_weight

        # save the strength of the attack
        self.alpha = alpha

        # variable for storing the target representation of the attack
        self.target_representation = None

        # create list for tracking the loss during iterations
        self.loss_steps = [9999]

        # create list for tracking the PSNR during iterations
        self.psnr_steps = [999]

        # variable to store the detector engine
        self._engine = None

        # variable used to store the generated adversarial noise
        self.noise = None

        # create variable to store the momentum of the gradient
        self.moving_avg_gradient = None

        # variable to control the strength of the momentum
        assert (0 <= momentum_coeficient <= 1)
        self.momentum_coeficient = momentum_coeficient

    def _on_before_attack(self):
        """
        Write the input parameters into the logs, generate target representation
        :return: None
        """
        super(BaseWhiteBoxAttack, self)._on_before_attack()

        self.write_to_logs("Source image: {}".format(self.source_image.path))
        self.write_to_logs("Alpha: {}".format(self.alpha))
        self.write_to_logs("Momentum coefficient:{}".format(self.momentum_coeficient))
        self.write_to_logs("Regularization weight:{}".format(self.regularization_weight))

        # compute the target representation
        self.target_representation = self._compute_target_representation(self.source_image, self.source_image_mask)

        if self.source_image.path != self.target_image.path and not self.test:
            self.detector.prediction_pipeline(self.source_image,
                                              path=os.path.join(self.debug_folder, "Initial result source")
                                              , original_picture=self.source_image, omask=self.source_image_mask,
                                              note="Initial state")

    def _on_after_attack_step(self, attacked_image: Picture, *args, **kwargs):
        """
        At each step update the graph of the loss and of the PSNR
        :param attacked_image:
        :param args:
        :param kwargs:
        :return:
        """

        # compute the PSNR between the initial image
        psnr = PSNR(self.target_image, np.array(attacked_image, np.int))
        self.psnr_steps.append(psnr)

        super()._on_after_attack_step(attacked_image)

        self.detector.plot_graph(self.loss_steps[1:], "Loss", "Attack iteration",
                                 os.path.join(self.debug_folder, "loss"))
        self.detector.plot_graph(self.psnr_steps[1:], "PSNR", "Attack iteration",
                                 os.path.join(self.debug_folder, "psnr"))

        # write the loss and psnr into the log
        self.write_to_logs("Loss: {:.2f}".format(self.loss_steps[-1]))
        self.write_to_logs("Psnr: {:.2f}".format(psnr))

    @abstractmethod
    def _get_gradient_of_image(self, image: Picture, target: Picture, old_perturbation: Picture = None):
        """
        Function to compute and return the gradient of the image w.r.t  the given target representation.
        The old perturbation is used for computing the l2 regularization
        :param image: image on which to calculate the gradient
        :param target: target representation
        :param old_perturbation: old_perturbation that has already been applied to the image, it is used for computing the
            regularization.
        :return: image_gradient, cumulative_loss
        """
        raise NotImplementedError

    def step_note(self):
        return "Step:{} Loss:{:.2f} PSNR:{:.2f}".format(self.step_counter + 1, self.loss_steps[-1], self.psnr_steps[-1])

    @abstractmethod
    def _compute_target_representation(self, target_representation_source_image: Picture,
                                       target_representation_source_image_mask: Picture):
        """
        Function to compute the target representation our attack are going to try to mimic
        :param target_representation_source_image: image source of the target representation
        :param target_representation_source_image_mask: mask of the image source of the target representation
        :return: target representation, depending on the attack this may be a numpy array, a tuple of arrays, ...
        """
        raise NotImplementedError

    @property
    def attacked_image(self):
        """
        Compute the attacked image using the original image and the cumulative noise to reduce
        rounding artifacts caused by translating the noise from one to 3 channels and vie versa multiple times,
        still this operation here is done once so some rounding error is still present.
        Use attacked_image_monochannel to get the one channel version of the image withoud rounding errors
        :return:
        """
        return Picture((self.target_image - Picture(self.noise)).clip(0, 255))

    def loss(self, y_pred, y_true, perturbation=None):
        """
        Specify a loss function to drive the image we are attacking towards the target representation
        The default loss is the l2-norm
        :param y_pred: last output of the model
        :param y_true: target representation
        :param perturbation:perturbation for which to compute the regularization value
        :return: loss value
        """

        loss = tf.cast(self.loss_function(y_pred, y_true), tf.float64)
        perturbation = tf.cast(self.regularizer_function(perturbation) * self.regularization_weight, tf.float64)

        return loss + perturbation

    def loss_function(self, y_pred, y_true):
        """
        Specify a loss function to drive the image we are attacking towards the target representation
        The default loss is the l2-norm
        :param y_pred: last output of the model
        :param y_true: target representation
        :return: loss value
        """
        raise NotImplementedError

    def regularizer_function(self, perturbation=None):
        """
        Compute te regularization value to add to the loss function
        :param perturbation:perturbation for which to compute the regularization value
        :return: regularization value
        """

        # if no perturbation is given return 0
        if perturbation is None:
            return 0

        raise NotImplementedError

    @staticmethod
    def read_arguments(dataset_root) -> dict:
        """
        Read arguments from the command line or ask for them if they are not present, validate them raising
        an exception if they are invalid, it is called by the launcher script
        :param args: args dictionary containing the arguments passed while launching the program
        :return: kwargs to pass to the attack
        """
        kwarg = BaseIterativeAttack.read_arguments(dataset_root)
        parser = argparse.ArgumentParser()
        parser.add_argument("-a", '--alpha', default=5, type=float, help='Strength of the attack')
        args = parser.parse_known_args()[0]

        kwarg["alpha"] = float(args.alpha)

        return kwarg
