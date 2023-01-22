import argparse
import os
from abc import ABC, abstractmethod
import numpy as np
import tensorflow as tf
from cv2 import PSNR

from Attacks.BaseIterativeAttack import BaseIterativeAttack
from Detectors.DetectorEngine import DetectorEngine
from Utilities.Image.Picture import Picture
from Utilities.Image.functions import create_random_nonoverlapping_mask
from Utilities.Plots import plot_graph
from Utilities.Visualizers.BaseVisualizer import BaseVisualizer


def normalize_gradient(gradient, margin=17):
    """
    Normalize the gradient cutting away the values on the borders
    @param margin: margin to use along the bordes
    @param gradient: gradient to normalize
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

    def __init__(self, visualizer: BaseVisualizer, steps: int, alpha: float, momentum_coeficient: float = 0.5,
                 regularization_weight=0.05, plot_interval=5, additive_attack=True,
                 debug_root: str = "./Data/Debug/", verbosity: int = 2):
        """
        @param visualizer: instance of the visualizer class wrapping the functionalities of the targeted detector
        @param steps: number of attack iterations to perform
        @param alpha: strength of the attack
        @param momentum_coeficient: [0,1] how relevant is the velocity ferived from past gradients for computing the
            current gradient? 0 -> not relevant, 1-> is the only thing that matters
        @param regularization_weight: [0,1] importance of the regularization factor in the loss function
        @param plot_interval: how often (# steps) should the step-visualizations be generated?
        @param additive_attack: show we feed the result of the iteration i as the input of the iteration 1+1?
        @param debug_root: root folder inside which to create a folder to store the data produced by the pipeline
        @param verbosity: is this a test mode? In test mode visualizations and superfluous steps will be skipped in favour of a
            faster execution to test the code
        """
        super(BaseWhiteBoxAttack, self).__init__(visualizer, steps, plot_interval, additive_attack,
                                                 debug_root, verbosity)

        # save the source image and its mask
        self.source_image_path = None
        self.source_image = None
        self.source_image_mask = None

        # Create object to store the target forgery mask (That is the forgery we want to inprint in an image)
        self.target_forgery_mask = None

        # save the regularization weight
        self.regularization_weight = regularization_weight

        # save the strength of the attack
        self.alpha = alpha

        # variable for storing the target representation of the attack
        self.target_representation = None

        # create list for tracking the loss during iterations
        self.loss_steps = None

        # create list for tracking the PSNR during iterations
        self.psnr_steps = None

        # variable used to store the generated adversarial noise
        self.noise = None

        # create variable to store the momentum of the gradient
        self.moving_avg_gradient = None

        # variable to control the strength of the momentum
        assert (0 <= momentum_coeficient <= 1)
        self.momentum_coeficient = momentum_coeficient

    def setup(self, target_image_path: str, target_image_mask: Picture, source_image_path: Picture = None,
              source_image_mask: Picture = None, target_forgery_mask: Picture = None):
        """
        @param target_image_path: path fo the sample to process
        @param target_image_mask: np.array containing a binary mask where 0 -> pristine 1-> forged pixel
        @param source_image_path: image from which we will compute the target representation
        @param source_image_mask: mask of the image from which we will compute the target representation
        @param target_forgery_mask: mask highlighting the section of the image that should be identified as forged after the attack
        :return:
        """
        self.target_image_path = None
        self.source_image = None
        self.target_image = None
        self.target_forgery_mask = None
        self.source_image_mask = None
        self.source_image_path = None

        # if a ad hoc source image is given load it and print its initial results
        if source_image_path is not None and target_image_path != source_image_path:
            # Load the source image in the visualizer
            print('####SETTING UP THE ATTACK###')
            self.visualizer.initialize(source_image_path)

            # Read the loaded data
            self.source_image = self.visualizer.metadata["sample"]

            # Print the detector's results on the loaded sample
            self.visualizer.save_prediction_pipeline(os.path.join(self.debug_folder, "initial result source.png"))

        # reload the target sample in the visualizer
        self.visualizer.initialize(target_image_path)

        super().setup(target_image_path, target_image_mask)


        # If no source image is given use the target image as source
        if self.source_image is None:
            self.source_image = self.visualizer.metadata["sample"]

        # save the source image and its mask
        self.source_image_mask = source_image_mask
        if self.source_image_mask is None:
            self.source_image_mask = self.target_image_mask

        # If no target forgery maks is given and if it is not false, generate one randomly
        self.target_forgery_mask = target_forgery_mask
        if target_forgery_mask is None:
            self.logger_module.warn("No target forgery mask has been given, it will be generated randomly")

            self.target_forgery_mask = create_random_nonoverlapping_mask(self.target_image_mask)

            Picture(self.target_forgery_mask*255).save(os.path.join(self.debug_folder, "computed target forgery "
                                                                                       "mask.png"))
        # create list for tracking the loss during iterations
        self.loss_steps = [9999]

        # create list for tracking the PSNR during iterations
        self.psnr_steps = [999]

    def _on_before_attack(self):
        """
        Write the input parameters into the logs, generate target representation
        :return: None
        """

        super(BaseWhiteBoxAttack, self)._on_before_attack()
        self.logger_module.info("Source image: {}".format(self.source_image_path))
        self.logger_module.info("Alpha: {}".format(self.alpha))
        self.logger_module.info("Momentum coefficient:{}".format(self.momentum_coeficient))
        self.logger_module.info("Regularization weight:{}".format(self.regularization_weight))

        # compute the target representation
        self.target_representation = self._compute_target_representation(Picture(value=self.source_image),
                                                                         self.source_image_mask)

    def _on_after_attack_step(self, attacked_image: Picture, *args, **kwargs):
        """
        At each step update the graph of the loss and of the PSNR
        @param attacked_image:
        @param args:
        @param kwargs:
        :return:
        """

        # compute the PSNR between the initial image
        psnr = PSNR(np.array(self.target_image, dtype=np.float32), np.array(attacked_image, dtype=np.float32))
        self.psnr_steps.append(psnr)

        super()._on_after_attack_step(attacked_image)

        plot_graph(self.loss_steps[1:], "Loss", "Attack iteration",
                   os.path.join(self.debug_folder, "loss"))
        plot_graph(self.psnr_steps[1:], "PSNR", "Attack iteration",
                   os.path.join(self.debug_folder, "psnr"))

        # write the loss and psnr into the log
        self.logger_module.info("Loss: {:.2f}".format(self.loss_steps[-1]))
        self.logger_module.info("Psnr: {:.2f}".format(psnr))

    @abstractmethod
    def _get_gradient_of_image(self, image: Picture, target: Picture, old_perturbation: Picture = None):
        """
        Function to compute and return the gradient of the image w.r.t  the given target representation.
        The old perturbation is used for computing the l2 regularization
        @param image: image on which to calculate the gradient
        @param target: target representation
        @param old_perturbation: old_perturbation that has already been applied to the image, it is used for computing the
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
        @param target_representation_source_image: image source of the target representation
        @param target_representation_source_image_mask: mask of the image source of the target representation
        :return: target representation, depending on the attack this may be a numpy array, a tuple of arrays, ...
        """
        raise NotImplementedError

    @property
    def attacked_image(self):
        """
        Compute the attacked image using the original image and the cumulative noise to reduce
        rounding artifacts caused by translating the noise from one to 3 channels and vie versa multiple times,
        still this operation here is done once so some rounding error is still present.
        Use attacked_image_monochannel to get the one channel version of the image without rounding errors
        :return:
        """
        return Picture((self.target_image - Picture(self.noise)).clip(0, 255))

    def loss(self, y_pred, y_true, perturbation=None):
        """
        Specify a loss function to drive the image we are attacking towards the target representation
        The default loss is the l2-norm
        @param y_pred: last output of the model
        @param y_true: target representation
        @param perturbation:perturbation for which to compute the regularization value
        :return: loss value
        """

        loss = tf.cast(self.loss_function(y_pred, y_true), tf.float64)
        # perturbation = tf.cast(self.regularizer_function(perturbation) * self.regularization_weight, tf.float64)

        return loss

    def loss_function(self, y_pred, y_true):
        """
        Specify a loss function to drive the image we are attacking towards the target representation
        The default loss is the l2-norm
        @param y_pred: last output of the model
        @param y_true: target representation
        :return: loss value
        """
        raise NotImplementedError

    def regularizer_function(self, perturbation=None):
        """
        Compute te regularization value to add to the loss function
        @param perturbation:perturbation for which to compute the regularization value
        :return: regularization value
        """

        # if no perturbation is given return 0
        if perturbation is None:
            return 0

        raise NotImplementedError

    @staticmethod
    def read_arguments(dataset_root) -> tuple:
        """
        Read arguments from the command line or ask for them if they are not present, validate them raising
        an exception if they are invalid, it is called by the launcher script
        @param args: args dictionary containing the arguments passed while launching the program
        :return: kwargs to pass to the attack
        """
        attack_parameters, setup_parameters = BaseIterativeAttack.read_arguments(dataset_root)
        parser = argparse.ArgumentParser()
        parser.add_argument("-a", '--alpha', default=5, type=float, help='Strength of the attack')
        args = parser.parse_known_args()[0]

        attack_parameters["alpha"] = float(args.alpha)

        return attack_parameters, setup_parameters
